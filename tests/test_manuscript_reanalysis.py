"""The paper's sensitivity bars must enclose the fitted alternatives it cites."""
import importlib.util
from pathlib import Path

import numpy as np


def module():
    path = Path(__file__).resolve().parents[1]/'manuscript'/'reanalysis.py'
    assert path.exists(), 'the manuscript needs a reproducible reanalysis'
    spec = importlib.util.spec_from_file_location('paper_reanalysis', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_sensitivity_contains_nonanalytic_limit_on_each_coordinate():
    t = np.geomspace(.001, .04, 12)
    states = [dict(T_H=float(x), tau=float(x), psi_gb=float(2+3*x+.7*np.sqrt(x)))
              for x in t]
    central = float(np.polynomial.polynomial.polyfit(t[:6],
        [s['psi_gb'] for s in states[:6]], 3)[0])
    result = module().sensitivity(states, central)
    assert abs(central-2) > .001
    assert abs(result['intercepts']['T_sqrt_12']-2) < 1e-10
    assert result['envelope'] >= abs(central-2)-1e-10
    assert all(abs(value-central) <= result['envelope']+1e-12
               for value in result['intercepts'].values())


def test_temperature_coordinate_change_is_included_and_inputs_untouched():
    t = np.geomspace(.002, .05, 10)
    states = [dict(T_H=float(x), tau=float(x*(1+20*x)),
                   psi_gb=float(1+2*x+10*x*x)) for x in t]
    snapshot = [dict(s) for s in states]
    result = module().sensitivity(states, 1.)
    assert states == snapshot
    assert result['points'] == 10
    assert abs(result['intercepts']['T_cubic_6']-1) < 1e-12
    assert abs(result['intercepts']['tau_cubic_6']-1) > 1e-6
    assert result['envelope'] >= abs(result['intercepts']['tau_cubic_6']-1)
