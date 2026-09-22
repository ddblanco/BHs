"""The publication audit must detect missing, changed and undeclared evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def checker():
    path = ROOT/'checks/verify_distribution.py'
    assert path.is_file(), 'standalone distribution audit is required'
    spec = importlib.util.spec_from_file_location('distribution', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_distribution_is_intact():
    assert checker().main(ROOT) == 0


@pytest.mark.parametrize('damage', ['changed', 'missing', 'unlisted', 'escape'])
def test_distribution_audit_rejects_corruption(tmp_path, damage):
    source = tmp_path/'sample.txt'
    source.write_bytes(b'scientific evidence\n')
    inventory = {'sample.txt': hashlib.sha256(source.read_bytes()).hexdigest()}
    (tmp_path/'artifacts').mkdir()
    index = tmp_path/'artifacts/distribution-sha256.json'
    index.write_text(json.dumps(inventory), encoding='utf-8')
    assert checker().main(tmp_path) == 0
    if damage == 'changed':
        source.write_bytes(b'altered evidence\n')
    elif damage == 'missing':
        source.unlink()
    elif damage == 'unlisted':
        (tmp_path/'unlisted.txt').write_text('unreviewed addition')
    else:
        inventory['../outside.txt'] = '0'*64
        index.write_text(json.dumps(inventory), encoding='utf-8')
    assert checker().main(tmp_path) == 1


@pytest.mark.parametrize('damage', ['missing_input', 'undeclared_failure', 'stale_failure', 'invalid_schema', 'missing_record'])
def test_scientific_inventory_rejects_broken_provenance(tmp_path, damage):
    (tmp_path/'artifacts').mkdir()
    (tmp_path/'results').mkdir()
    (tmp_path/'input.py').write_text('source')
    (tmp_path/'results/sample.json').write_text('{}')
    record = dict(id='sample', kind='result', path='results/sample.json',
                  command='python input.py', commit='historical', inputs=['input.py'],
                  parameters={}, environment='input.py', agent='author', prompt_refs=[],
                  decisions=[], checks=[dict(name='known_failure', passed=False)], status='rejected')
    limitations = [dict(artifact='sample', check='known_failure', reason='Measured disagreement')]
    manifest = tmp_path/'artifacts/manifest.json'
    declared = tmp_path/'artifacts/limitations.json'
    def save():
        manifest.write_text(json.dumps([record]))
        declared.write_text(json.dumps(limitations))
    save()
    assert checker().scientific_failures(tmp_path) == []
    if damage == 'missing_input':
        (tmp_path/'input.py').unlink()
    elif damage == 'undeclared_failure':
        limitations.clear()
    elif damage == 'stale_failure':
        record['checks'][0]['passed'] = True
    elif damage == 'invalid_schema':
        record['checks'] = 'invalid'
    else:
        (tmp_path/'results/unlisted.json').write_text('{}')
    save()
    assert checker().scientific_failures(tmp_path)
