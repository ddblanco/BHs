"""Serialized polynomials must preserve independent physical evidence."""
import json
from pathlib import Path
import numpy as np
from rotating_bh.egb_rotating_saved import SavedProfile
from rotating_bh.egb_rotating_observables import measure
from rotating_bh.egb_rotating_validation import diagnose


def test_stored_egb_polynomials_preserve_charges_and_full_tensor():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'results/egb-rotating-family.json').read_text())
    row = next(r for r in data['records'] if r['label']=='resolution' and r['alpha_gb']==.1 and r['q']==.3)
    profile = SavedProfile(row)
    actual = measure(profile)
    for key in ['E','J','T_H','S']:
        np.testing.assert_allclose(actual[key],row['observables'][key],rtol=1e-10,atol=1e-10)
    assert diagnose(profile,.1,count=11)['max_tensor_residual']<1e-8
