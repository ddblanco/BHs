"""Artifact integrity must detect changed evidence and changed code."""
import json
from pathlib import Path
import shutil
import pytest
from experiments.egb_rotating_family_artifacts import verify, DATA, FIGURE, FAILED, FAILED_SOURCE
from known_digest_exceptions import reconcile


@pytest.mark.parametrize('corruption',['result','figure','source','manifest'])
def test_family_artifact_corruption_is_detected(tmp_path,corruption):
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root/DATA).read_text())
    files = {DATA,FIGURE,FAILED,FAILED_SOURCE,'references/notes/1010.0860-hito-4c.md',
             'artifacts/manifest.json','experiments/egb_rotating_family_artifacts.py',
             'src/rotating_bh/provenance.py',*result['sources']}
    for name in files:
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    # The shipped record carries a documented digest exception, which would
    # make this baseline raise before the test reaches the corruption it
    # exists to check. Reconcile it in the staged copy only.
    assert reconcile(tmp_path), 'no documented exception applied; baseline unchanged'
    verify(tmp_path)
    if corruption=='manifest':
        path = tmp_path/'artifacts/manifest.json'
        manifest = json.loads(path.read_text())
        for row in manifest:
            if row['path']==DATA: row['status']='rejected'
        path.write_text(json.dumps(manifest))
    else:
        name = {'result':DATA,'figure':FIGURE,'source':'src/rotating_bh/egb_rotating_observables.py'}[corruption]
        with (tmp_path/name).open('ab') as target:
            target.write(b' ')
    with pytest.raises(ValueError):
        verify(tmp_path)
