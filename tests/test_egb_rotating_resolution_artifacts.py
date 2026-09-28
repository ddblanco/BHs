"""Exercise artifact verification on isolated copies, including corruption."""
import json
from pathlib import Path
import shutil

import pytest

from known_digest_exceptions import reconcile


@pytest.mark.parametrize('corruption', ['source', 'figure', 'cross_check'])
def test_artifact_verifier_detects_corruption(tmp_path, corruption):
    from experiments.egb_rotating_convergence import verify
    root = Path(__file__).resolve().parents[1]
    data = 'results/egb-rotating-resolution.json'
    cross = 'results/egb-rotating-cross-check.json'
    result = json.loads((root/data).read_text(encoding='utf-8'))
    cross_result = json.loads((root/cross).read_text(encoding='utf-8'))
    files = {data, cross, 'artifacts/manifest.json', 'artifacts/egb-rotating-resolution.png',
             *result['source_sha256'], *cross_result['source_sha256']}
    for name in files:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    # See tests/known_digest_exceptions.py: the shipped record's documented
    # exception is reconciled in the staged copy so the baseline can pass and
    # the corruption below is what actually gets tested.
    assert reconcile(tmp_path), 'no documented exception applied; baseline unchanged'
    verify(tmp_path)
    name = {'source': 'src/rotating_bh/egb_rotating_convergence.py',
            'figure': 'artifacts/egb-rotating-resolution.png', 'cross_check': cross}[corruption]
    with (tmp_path/name).open('ab') as target:
        target.write(b' ')
    with pytest.raises(ValueError):
        verify(tmp_path)
