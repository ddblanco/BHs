"""Guard the byte-level integrity of every digest the repository records.

Provenance hashes the bytes of a source file, line endings included. A
line-ending rewrite changes those bytes without changing the content, so it
breaks the link between a result and its inputs while leaving every diff empty.
This test exists because exactly that happened: the digests recorded by
`results/*.json` and `artifacts/manifest.json` stopped matching, and nothing in
the suite noticed -- the acceptance copy was taken from the working tree, where
the bytes were still right, while the repository stored normalised ones.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    'verify_recorded_digests', ROOT/'checks/verify_recorded_digests.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_every_recorded_digest_matches_the_bytes_on_disk():
    failures = []
    for source, target, expected in _module.recorded_digests(ROOT):
        if _module.optional_external_missing(ROOT, target, expected):
            continue
        if (target, expected) in _module.KNOWN_EXCEPTIONS:
            continue
        path = ROOT/target
        assert path.is_file(), f'{target}, recorded in {source}, is missing'
        actual = _module.sha(path.read_bytes())
        if actual != expected:
            failures.append(f'{target} (recorded in {source}): '
                            f'{_module.variant_note(path, expected)}')
    assert not failures, 'recorded digests no longer match the bytes on disk:\n' + '\n'.join(failures)


def test_known_exceptions_are_still_real_and_documented():
    """An exception must stay justified: it has to name a reason, and the file
    must still fail to match. One that starts matching has to be removed, so the
    list cannot quietly grow into a way of muting the check."""
    for (target, expected), reason in _module.KNOWN_EXCEPTIONS.items():
        assert reason.strip(), f'exception for {target} carries no reason'
        path = ROOT/target
        assert path.is_file(), f'exception names a missing file: {target}'
        assert _module.sha(path.read_bytes()) != expected, (
            f'{target} now matches its recorded digest; remove it from KNOWN_EXCEPTIONS')


def test_guard_detects_a_line_ending_rewrite(tmp_path):
    """Negative control: the exact failure mode this guard exists for."""
    import json, shutil
    (tmp_path/'results').mkdir()
    (tmp_path/'artifacts').mkdir()
    source = tmp_path/'sample.py'
    source.write_bytes(b'x = 1\r\ny = 2\r\n')
    digest = _module.sha(source.read_bytes())
    (tmp_path/'results/sample.json').write_text(
        json.dumps({'source_sha256': {'sample.py': digest}}), encoding='utf-8')
    (tmp_path/'artifacts/manifest.json').write_text('[]', encoding='utf-8')

    assert _module.main(tmp_path) == 0

    source.write_bytes(source.read_bytes().replace(b'\r\n', b'\n'))
    assert _module.main(tmp_path) == 1
    assert 'LINE ENDINGS' in _module.variant_note(source, digest)


def test_guard_checks_sources_of_nested_results(tmp_path):
    import json

    (tmp_path / 'results/nested').mkdir(parents=True)
    source = tmp_path / 'sample.py'
    source.write_bytes(b'original input\n')
    (tmp_path / 'results/nested/sample.json').write_text(json.dumps({
        'source_sha256': {'sample.py': _module.sha(source.read_bytes())}
    }), encoding='utf-8')
    assert _module.main(tmp_path) == 0
    source.write_bytes(b'changed input\n')
    assert _module.main(tmp_path) == 1


def test_git_is_configured_never_to_rewrite_bytes():
    """The guard above catches a break after it happens; this catches the setting
    that would cause one. `* -text` tells Git not to convert line endings, so
    working-tree bytes equal blob bytes and a checkout cannot invalidate a
    recorded digest. `checks/clean-copy.ps1` relies on this to take its
    acceptance copy from the working tree at all."""
    attributes = ROOT/'.gitattributes'
    assert attributes.is_file(), '.gitattributes is missing; Git may renormalise'
    rules = [line.strip() for line in attributes.read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.startswith('#')]
    assert '* -text' in rules, f'expected `* -text`, found {rules}'
    for rule in rules:
        assert 'text=auto' not in rule, (
            f'`{rule}` renormalises line endings and invalidates recorded digests')
        assert 'eol=' not in rule, (
            f'`{rule}` forces a line ending and invalidates recorded digests')


def test_only_the_documented_external_image_may_be_absent(tmp_path):
    import json
    image = 'references/data/1010.0860v1-figure-1b.png'
    recorded = __import__('json').loads(
        (ROOT/'results/egb-rotating-paper-profiles.json').read_text())
    expected = recorded['source_sha256'][image]
    (tmp_path/'results').mkdir()
    data = tmp_path/'results/image.json'
    data.write_text(json.dumps({'source_sha256': {image: expected}}))
    assert _module.main(tmp_path) == 0
    path = tmp_path/image
    path.parent.mkdir(parents=True)
    path.write_bytes(b'corrupt image')
    assert _module.main(tmp_path) == 1
    path.unlink()
    data.write_text(json.dumps({'source_sha256': {image: '0'*64}}))
    assert _module.main(tmp_path) == 1


def test_missing_scientific_source_is_still_an_error(tmp_path):
    import json
    (tmp_path/'results').mkdir()
    (tmp_path/'results/missing.json').write_text(json.dumps(
        {'source_sha256': {'src/solver.py': '0'*64}}))
    assert _module.main(tmp_path) == 1
