"""Verify the shipped bytes, independently of historical scientific provenance.

The inventory is a release snapshot, not a signature or a way to validate
freshly regenerated results. This command never updates it.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
INDEX = 'artifacts/distribution-sha256.json'
OPTIONAL_IMAGE = 'references/data/1010.0860v1-figure-1b.png'
TRANSIENT_DIRS = {'.git', '.venv', '.cache', '.pytest_cache', '__pycache__'}
TRANSIENT_SUFFIXES = {'.pyc', '.aux', '.log', '.out', '.toc'}


def scientific_failures(root):
    sys.path.insert(0, str(ROOT/'src'))
    from rotating_bh.provenance import validate_manifest
    manifest = root/'artifacts/manifest.json'
    try:
        validate_manifest(manifest)
        records = json.loads(manifest.read_text(encoding='utf-8'))
        limitations = json.loads((root/'artifacts/limitations.json').read_text(encoding='utf-8'))
        declared = {(row['artifact'], row['check']) for row in limitations if row['reason'].strip()}
        failing = {(r['id'], c['name']) for r in records for c in r['checks'] if not c['passed']}
        failures = []
        if failing != declared or len(declared) != len(limitations):
            failures.append('failed scientific checks differ from declared limitations')
        targets = {r['path'] for r in records}
        for record in records:
            for name in [record['path'], record['environment'], *record['inputs'], *record['prompt_refs']]:
                path = (root/name).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    failures.append(f'missing or invalid scientific input/artifact: {name}')
        for path in [*root.glob('results/**/*.json'), *root.glob('artifacts/*.png')]:
            if path.relative_to(root).as_posix() not in targets:
                failures.append(f'artifact without provenance: {path.name}')
        return failures
    except (OSError, ValueError, TypeError, KeyError) as error:
        return [f'invalid scientific inventory: {error}']


def published_files(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob('*')
            if p.is_file()
            and not any(part in TRANSIENT_DIRS or part.endswith('.egg-info')
                        for part in p.relative_to(root).parts)
            and p.suffix not in TRANSIENT_SUFFIXES
            and not p.name.endswith('.synctex.gz')
            and p.relative_to(root).as_posix() not in {INDEX, OPTIONAL_IMAGE}}


def main(root=ROOT):
    root = Path(root).resolve()
    try:
        inventory = json.loads((root/INDEX).read_text(encoding='utf-8'))
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError('inventory must be a nonempty object')
        files = published_files(root)
        failures = []
        for name, expected in inventory.items():
            target = (root/name).resolve()
            if not target.is_relative_to(root) or name not in files:
                failures.append(f'missing or invalid path: {name}')
            elif hashlib.sha256(files[name].read_bytes()).hexdigest() != expected:
                failures.append(f'changed: {name}')
        failures.extend(f'unlisted: {name}' for name in sorted(files.keys()-inventory.keys()))
        if (root/'artifacts/manifest.json').exists():
            failures.extend(scientific_failures(root))
    except (OSError, ValueError, TypeError) as error:
        print(f'FAIL: {error}')
        return 1
    for failure in failures:
        print(f'FAIL: {failure}')
    print(f'{len(inventory)} distribution files; {len(failures)} failures')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
