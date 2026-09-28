"""The exception bridge must excuse exactly what is documented, and nothing else.

`known_digest_exceptions` lets the artifact guards read the one list of analysed
digest mismatches. That is only safe if it cannot be stretched: a new break on
an already-excused file, or any undocumented pair, has to keep failing.
"""
import hashlib
import json

import pytest

from known_digest_exceptions import KNOWN_EXCEPTIONS, excused, reconcile

DOCUMENTED = next(iter(KNOWN_EXCEPTIONS))


def test_only_the_exact_documented_pair_is_excused():
    target, digest = DOCUMENTED
    assert excused(target, digest)
    assert not excused(target, '0'*64), 'a different digest on the same file must still fail'
    assert not excused('src/rotating_bh/einstein.py', digest), 'the pair is path-specific'
    assert not excused('anything.py', '0'*64)


def test_windows_separators_do_not_smuggle_a_path_past_the_list():
    target, digest = DOCUMENTED
    assert excused(target.replace('/', '\\'), digest)


def test_reconcile_leaves_undocumented_digests_untouched(tmp_path):
    source = tmp_path/'src/solver.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'x = 1\n')
    stale = '1'*64
    record = tmp_path/'results/run.json'
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({'source_sha256': {'src/solver.py': stale}}), encoding='utf-8')

    assert reconcile(tmp_path) == []
    assert json.loads(record.read_text())['source_sha256']['src/solver.py'] == stale


def test_reconcile_rewrites_only_the_excused_entry(tmp_path):
    target, digest = DOCUMENTED
    staged = tmp_path/target
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b'whatever bytes\n')
    other = tmp_path/'src/other.py'
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b'y = 2\n')
    stale = '2'*64
    record = tmp_path/'results/run.json'
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(
        {'source_sha256': {target: digest, 'src/other.py': stale}}), encoding='utf-8')

    assert reconcile(tmp_path) == [(target, digest)]
    written = json.loads(record.read_text())['source_sha256']
    assert written[target] == hashlib.sha256(staged.read_bytes()).hexdigest()
    assert written['src/other.py'] == stale, 'an undocumented digest must survive untouched'


def test_reconcile_reports_nothing_when_there_is_nothing_to_excuse(tmp_path):
    (tmp_path/'results').mkdir()
    (tmp_path/'results/empty.json').write_text('{}', encoding='utf-8')
    assert reconcile(tmp_path) == []
