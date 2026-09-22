"""Cross-check the reserved tt equation against the independent tensor."""
from functools import lru_cache

import numpy as np
import pytest
import sympy as sp

from rotating_bh.einstein import compile_metric, curvature, gauss_bonnet, metric_from_ansatz
from rotating_bh.static_egb_equations import field_residual


@lru_cache(maxsize=1)
def _jets():
    t, r, theta, phi1, phi2 = sp.symbols('t r theta phi1 phi2', real=True)
    r0 = sp.Symbol('r0', positive=True)
    b0, b1, b2, f0, f1, f2 = sp.symbols('b0 b1 b2 f0 f1 f2')
    b = b0+b1*(r-r0)+b2*(r-r0)**2/2
    f = f0+f1*(r-r0)+f2*(r-r0)**2/2
    metric = metric_from_ansatz(r, theta, b, f, r**2, 0)
    return compile_metric(metric, (t, r, theta, phi1, phi2), (r0, b0, b1, b2, f0, f1, f2))


def tt_from_tensor(r0, b0, b1, f0, f1, alpha_gb, theta=0.6):
    """G_tt+alpha_GB H_tt for an independent (b,f,b',f') jet, b2=f2=0."""
    g, dg, ddg = _jets()(0, r0, theta, 0, 0, r0, b0, b1, 0., f0, f1, 0.)
    result = curvature(g, dg, ddg)
    gb = gauss_bonnet(g, result['inverse'], result['riemann'], result['ricci'], result['scalar'])
    field = result['einstein']+alpha_gb*gb['H']
    return field[0, 0]


@pytest.mark.parametrize('alpha_gb', [0.0, 0.05, 0.3])
@pytest.mark.parametrize('r0,b0,b1,f0,f1', [
    (1.4, 0.6, 0.9, 0.3, 1.7),   # generic, not a solution, b != f
    (2.3, 1.1, -0.2, 1.4, 0.4),
])
def test_field_residual_matches_independent_tensor(r0, b0, b1, f0, f1, alpha_gb):
    tensor_tt = tt_from_tensor(r0, b0, b1, f0, f1, alpha_gb)
    expected = field_residual(r0, f0, f1, alpha_gb)*3*b0/(2*r0**3)
    assert tensor_tt == pytest.approx(expected, abs=1e-9, rel=1e-9)


def _oracle_f_and_fprime(r0, alpha_gb):
    # Exact (sympy) derivative of the closed form, independent of any
    # finite-difference roundoff and of experiments/derive_static_egb.py.
    r, a = sp.symbols('r a', positive=True)
    q = r**2/(4*a)*(sp.sqrt(1+4*a*(4*a+2)/r**4)-1) if alpha_gb else 1/r**2
    f = 1-q
    fp = sp.diff(f, r)
    subs = {r: r0} if not alpha_gb else {r: r0, a: alpha_gb}
    return float(f.subs(subs)), float(fp.subs(subs))


@pytest.mark.parametrize('alpha_gb', [0.0, 0.02, 0.2, 1.5])
def test_field_residual_vanishes_on_the_closed_form_oracle(alpha_gb):
    # Symbolic derivative of the closed form: the reserved tt equation must
    # vanish exactly, at radii away from any future solver node.
    for r0 in (1.01, 1.3, 2.7, 9.0):
        f0, f1 = _oracle_f_and_fprime(r0, alpha_gb)
        assert field_residual(r0, f0, f1, alpha_gb) == pytest.approx(0.0, abs=1e-10)


def test_field_residual_detects_a_perturbed_profile():
    f0, f1 = _oracle_f_and_fprime(1.7, 0.1)
    assert abs(field_residual(1.7, f0*1.01, f1, 0.1)) > 1e-3
