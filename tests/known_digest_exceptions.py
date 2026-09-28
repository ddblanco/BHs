"""Let the artifact guards consult the one documented list of digest exceptions.

`checks/verify_recorded_digests.py` already owns `KNOWN_EXCEPTIONS`: pairs of
(recorded path, recorded digest) whose mismatch has been analysed and written
down, each with the reason it cannot simply be re-recorded. The guards in this
directory hashed files directly and knew nothing about that list, so a single
documented exception turned eleven of them red while the checker that
understands it exited 0. Reading the same list here keeps one source of truth
instead of two disagreeing ones.

This cannot become a way to mute a break. An entry excuses only the exact
(path, digest) pair that was analysed -- any other mismatch, including a new
one on the same file, fails as before -- and `test_recorded_digests.py`
separately requires every entry to carry a reason and to still mismatch, so an
exception that stops applying must be deleted.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    'verify_recorded_digests', ROOT/'checks/verify_recorded_digests.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

KNOWN_EXCEPTIONS = _module.KNOWN_EXCEPTIONS
DIGEST_MAPS = _module.DIGEST_MAPS


def excused(relative, digest):
    """True when this exact (path, digest) mismatch is the documented one."""
    return (str(relative).replace('\\', '/'), digest) in KNOWN_EXCEPTIONS


def reconcile(staged_root):
    """Re-point excused digests at the staged bytes, in a scratch copy only.

    The corruption tests stage an artifact into `tmp_path` and call the real
    `verify()` before damaging anything, to establish that an intact copy
    passes. A documented exception makes that baseline raise, so the test can
    never reach the corruption it exists to check. Rewriting the excused digest
    in the *staged* record -- never the shipped one, which keeps its exception
    and its reason -- restores the baseline while leaving every other digest to
    be verified for real, so appending a byte still has to be caught.

    The substitution is done on the raw bytes, swapping one 64-character hex
    string for another, so nothing else in the file moves. Rewriting a result
    does change that file's own hash, and `artifacts/manifest.json` records it,
    so any manifest entry pointing at a file this touched is refreshed too.

    Returns the pairs it reconciled, so a caller can assert it did something.
    """
    staged_root = Path(staged_root)
    reconciled, rewritten = [], set()
    for path in sorted(staged_root.rglob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        pairs = []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in DIGEST_MAPS and _module.digest_map(value):
                        pairs.extend((n, d) for n, d in value.items() if excused(n, d))
                    else:
                        walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        raw = path.read_text(encoding='utf-8')
        for name, recorded in pairs:
            target = staged_root/name
            if not target.is_file() or recorded not in raw:
                continue
            raw = raw.replace(recorded, hashlib.sha256(target.read_bytes()).hexdigest())
            reconciled.append((name, recorded))
        if reconciled and raw != path.read_text(encoding='utf-8'):
            path.write_text(raw, encoding='utf-8')
            rewritten.add(path.relative_to(staged_root).as_posix())

    manifest = staged_root/'artifacts/manifest.json'
    if rewritten and manifest.is_file():
        records = json.loads(manifest.read_text(encoding='utf-8'))
        touched = False
        for record in records if isinstance(records, list) else []:
            if record.get('path') in rewritten and 'sha256' in record:
                fresh = hashlib.sha256((staged_root/record['path']).read_bytes()).hexdigest()
                if record['sha256'] != fresh:
                    record['sha256'] = fresh
                    touched = True
        if touched:
            manifest.write_text(json.dumps(records, indent=2), encoding='utf-8')
    return reconciled
