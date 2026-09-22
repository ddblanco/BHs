"""Verify that every sha256 recorded in the repository still matches the bytes
on disk, byte for byte.

Provenance records hash source files exactly as they sat on disk, line endings
included. Any tool that rewrites line endings -- a `.gitattributes` with
`text=auto`, a checkout under `core.autocrlf=true`, an editor "fix" -- silently
invalidates those digests without changing a single character of content. That
is not a cosmetic problem: it destroys the link between a result and the code
that produced it.

This check walks both recording mechanisms in use:

* `source_sha256` maps, nested anywhere inside `results/**/*.json` and
  `artifacts/manifest.json`;
* the top-level `sha256` of each `artifacts/manifest.json` entry.

Run from the project root:

    python checks/verify_recorded_digests.py

Exit status is 0 when every digest matches (known exceptions aside) and 1
otherwise. When a file differs only in line endings the report says so, because
that diagnosis determines the repair: restore the bytes, never re-record the
digest.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Only this exact historical external input can be absent. If supplied, its
# bytes must match. Missing code/results and other image hashes remain errors.
EXTERNAL_IMAGE = ('references/data/1010.0860v1-figure-1b.png',
                  'a5760ea5b2d2ae6c397560bef6431e55acd37e72fc31ce7b10a1ea0006673543')


def optional_external_missing(root, target, expected):
    return (target, expected) == EXTERNAL_IMAGE and not (root/target).exists()

# Documented, pre-existing exceptions. Each entry must carry a reason; an
# exception is a statement that the gap is understood, not a way to silence it.
KNOWN_EXCEPTIONS = {
    ('docs/egb-rotating.md',
     'baf27012837e8e08cec67367618a910011de0e72cee7139e52038419f3ecfe90'):
        'Prose edited after results/egb-rotating.json was produced (Hito 4A/4B). '
        'The pre-edit bytes exist in no copy or commit, so the digest cannot be '
        'reproduced without re-running a closed hito. The file is documentation, '
        'not a numerical input; every code source of that artifact verifies. '
        'Present already in .cache/clean-hito-4c, the copy that passed 315 tests '
        'on 2026-09-10, so it predates any line-ending work.',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def variant_note(path, expected):
    """Say whether the file would match under the other line-ending convention."""
    if not path.exists():
        return 'file missing'
    raw = path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    if sha(lf) == expected:
        return 'LINE ENDINGS: on disk as CRLF, recorded as LF -- restore the bytes'
    if sha(crlf) == expected:
        return 'LINE ENDINGS: on disk as LF, recorded as CRLF -- restore the bytes'
    return 'CONTENT DIFFERS beyond line endings'


def recorded_digests(root):
    """Yield (recording file, target path, expected digest) for every mechanism."""
    for json_path in sorted(root.glob('results/**/*.json')) + [root/'artifacts/manifest.json']:
        if not json_path.exists():
            continue
        rel = json_path.relative_to(root).as_posix()
        data = json.loads(json_path.read_text(encoding='utf-8'))

        def walk(node):
            if isinstance(node, dict):
                sources = node.get('source_sha256')
                if isinstance(sources, dict):
                    for target, digest in sources.items():
                        yield rel, target, digest
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for value in node:
                    yield from walk(value)

        yield from walk(data)

        if json_path.name == 'manifest.json' and isinstance(data, list):
            for record in data:
                if isinstance(record, dict) and 'path' in record and 'sha256' in record:
                    yield rel, record['path'], record['sha256']


def main(root=ROOT):
    seen, failures, excused = set(), [], []
    for source, target, expected in recorded_digests(root):
        key = (source, target, expected)
        if key in seen:
            continue
        seen.add(key)
        if optional_external_missing(root, target, expected):
            excused.append((source, target,
                'External image not redistributed; see references/data/README.md. '
                'Image-dependent reproduction has NOT been checked.'))
            continue
        path = root/target
        actual = sha(path.read_bytes()) if path.is_file() else None
        if actual == expected:
            continue
        reason = KNOWN_EXCEPTIONS.get((target, expected))
        if reason:
            excused.append((source, target, reason))
        else:
            failures.append((source, target, expected, variant_note(path, expected)))

    print(f'checked {len(seen)} recorded digests')
    for source, target, reason in excused:
        print(f'  known exception: {target} (recorded in {source})\n      {reason}')
    for source, target, expected, note in failures:
        print(f'  MISMATCH {target}\n      recorded in {source} as {expected}\n      {note}')
    if failures:
        print(f'FAILED: {len(failures)} recorded digest(s) no longer match the bytes on disk')
        return 1
    print('OK: available inputs match, apart from the explicitly listed exceptions')
    return 0


if __name__ == '__main__':
    sys.exit(main())
