"""Regression tests for the Hito 4A rotating EGB radial system.

These check the *generated* code (_egb_rotating_generated.py,
_egb_rotating_equations_generated.py); experiments/derive_egb_rotating.py
re-runs the full symbolic derivation and its scientific gate (action vs.
tensor equivalence, order/rank, alpha_GB=0 vs. the independent vacuum
Lagrangian) from scratch and aborts generation if any of it fails -- these
tests exist so a hand-edit of the generated files (or an environment
regression) is caught quickly, without re-deriving anything symbolically.
"""
import numpy as np
import pytest
import sympy as sp

from rotating_bh import vacuum_equations
from rotating_bh._egb_rotating_generated import rhs
from rotating_bh._egb_rotating_equations_generated import equations
from rotating_bh.einstein import compile_metric, curvature, gauss_bonnet, metric_from_ansatz

# Generic (off-shell) profiles: (r, b,bp,bpp, f,fp, h,hp,hpp, w,wp,wpp). Not
# solutions of anything -- second derivatives are independent, not derived
# from rhs() -- so the tensor-equivalence identity below is exercised
# off-shell, which is the whole point of that check.
PROFILES = [
    (1.4, 1.1, 0.6, -0.3, 0.8, -0.4, 1.6, 0.5, -0.6, 0.10, -0.20, 0.30),
    (2.2, 0.7, -0.5, 0.8, 1.3, 0.3, 0.9, -0.7, 0.4, -0.05, 0.35, -0.15),
    (1.0, 1.9, 0.2, 0.5, 0.6, -0.6, 2.1, 0.9, -0.2, 0.20, 0.10, 0.10),
    (3.0, 0.9, -0.2, -0.4, 1.7, 0.5, 0.7, 0.3, 0.6, -0.30, -0.10, 0.20),
]


@pytest.mark.parametrize('profile', PROFILES)
def test_alpha_gb_zero_matches_the_tested_vacuum_equations(profile):
    r, b, bp, bpp, f, fp, h, hp, hpp, w, wp, wpp = profile
    mine = dict(zip(('b', 'f', 'g', 'h', 'w'),
                     equations(r, (b, f, h, w), (bp, fp, hp, wp), (bpp, 0.0, hpp, wpp), 0.0)))
    vacuum = vacuum_equations.radial_equations(r, (b, f, h, w), (bp, fp, hp, wp), (bpp, 0.0, hpp, wpp))
    for key in ('b', 'f', 'g', 'h', 'w'):
        assert mine[key] == pytest.approx(vacuum[key], abs=1e-9)


@pytest.mark.parametrize('alpha_gb', [0.0, 0.02, 0.3])
@pytest.mark.parametrize('profile', PROFILES)
def test_solved_derivatives_zero_the_raw_equations(profile, alpha_gb):
    # rhs() is the algebraic solution of equations()==0 for [f',b'',h'',w''];
    # plugging it back in must reproduce that, at any alpha_GB (the second
    # derivatives from the profile are irrelevant here -- rhs() overrides
    # them with the ones that actually solve the system).
    r, b, bp, _bpp, f, _fp, h, hp, _hpp, w, wp, _wpp = profile
    fp, bpp, hpp, wpp = rhs(r, b, f, h, w, bp, hp, wp, alpha_gb)
    e_b, e_f, e_g, e_h, e_w = equations(
        r, (b, f, h, w), (bp, fp, hp, wp), (bpp, 0.0, hpp, wpp), alpha_gb)
    for name, value in (('b', e_b), ('g', e_g), ('h', e_h), ('w', e_w)):
        assert value == pytest.approx(0.0, abs=1e-8), name


@pytest.mark.parametrize('alpha_gb', [0.0, 0.05, 0.4])
@pytest.mark.parametrize('profile', PROFILES)
def test_matches_the_independent_tensor_route(profile, alpha_gb):
    """E_v == -r^2 (G+alpha_GB H)^{ij} dg_{ij}/dv, for v in b,h,w, and
    E_f == r^2 (G+alpha_GB H)^r_r / f -- the same identity already used and
    tested for the vacuum system in Hito 2 (docs/vacuum-bvp.md), extended
    here with the Gauss-Bonnet tensor. This is the equivalence gate
    AstraCheck (2026-09-09) asked for, run here on the generated code
    directly rather than re-deriving symbolically (see module docstring).
    """
    r, b, bp, bpp, f, fp, h, hp, hpp, w, wp, wpp = profile
    e_b, e_f, e_g, e_h, e_w = equations(
        r, (b, f, h, w), (bp, fp, hp, wp), (bpp, 0.0, hpp, wpp), alpha_gb)
    action = dict(b=e_b, h=e_h, w=e_w)
    theta_val = 0.6

    # Metric with b,f,h,w as *local jets* (order-2 polynomials in r-R), so
    # dg/dr correctly picks up the chain-rule terms from b',h',w' (a plain
    # symbol for b,f,h,w, as in test_vacuum_equations.py, only works there
    # because that test never differentiates that copy w.r.t. r).
    t, rc, theta, phi1, phi2 = sp.symbols('t r theta phi1 phi2', real=True)
    R = sp.symbols('R', positive=True)
    b0, b1, b2, f0, f1, h0, h1, h2, w0, w1, w2 = sp.symbols('b0 b1 b2 f0 f1 h0 h1 h2 w0 w1 w2')
    jet = lambda v0, v1, v2=0: v0+v1*(rc-R)+v2*(rc-R)**2/2
    metric = metric_from_ansatz(rc, theta, jet(b0, b1, b2), jet(f0, f1), jet(h0, h1, h2), jet(w0, w1, w2))
    params = (R, b0, b1, b2, f0, f1, h0, h1, h2, w0, w1, w2)
    evaluate = compile_metric(metric, (t, rc, theta, phi1, phi2), params)
    arrays = evaluate(0, r, theta_val, 0, 0, r, b, bp, bpp, f, fp, h, hp, hpp, w, wp, wpp)
    result = curvature(*arrays)
    gb = gauss_bonnet(arrays[0], result['inverse'], result['riemann'], result['ricci'], result['scalar'])
    field_lower = result['einstein']+alpha_gb*gb['H']
    inv = result['inverse']
    field_upper = inv @ field_lower @ inv

    # Separate plain-symbol metric (not a function of r) purely for dg/dv.
    symbols = sp.symbols('b f h w', positive=True)
    bs, fs, hs, ws = symbols
    generic = metric_from_ansatz(rc, theta, bs, fs, hs, ws)
    for name, sym in zip(('b', 'f', 'h', 'w'), symbols):
        tangent = np.array(generic.diff(sym).subs(
            {rc: r, theta: theta_val, bs: b, fs: f, hs: h, ws: w})).astype(float)
        if name == 'f':
            grr_mixed = float(inv[1, :] @ field_lower[:, 1])
            expected = r**2*grr_mixed/f
            assert e_f == pytest.approx(expected, rel=1e-8)
            continue
        expected = -r**2*float(np.einsum('ij,ij->', field_upper, tangent))
        assert action[name] == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize('alpha_gb', [0.0, 0.07, 0.3])
def test_reserved_constraint_propagates_like_the_vacuum_one(alpha_gb):
    """r^2 sqrt(b*h) * C is constant along any on-shell trajectory (C=E_f*f/r^2,
    docs/vacuum-bvp.md) -- proved in general (not just numerically) by
    experiments/derive_egb_rotating.py::check_constraint_propagation. Here:
    integrate the solved system (rhs()) from generic initial data and check
    the quantity stays constant along the flow, using only the generated
    code (a regression test, not a re-derivation).
    """
    def deriv(r, y):
        b, f, h, w, bp, hp, wp = y
        fp, bpp, hpp, wpp = rhs(r, b, f, h, w, bp, hp, wp, alpha_gb)
        return np.array([bp, fp, hp, wp, bpp, hpp, wpp])

    def rk4_step(r, y, dr):
        k1 = deriv(r, y)
        k2 = deriv(r+dr/2, y+dr/2*k1)
        k3 = deriv(r+dr/2, y+dr/2*k2)
        k4 = deriv(r+dr, y+dr*k3)
        return y+dr/6*(k1+2*k2+2*k3+k4)

    r, y = 1.5, np.array([1.2, 0.8, 1.6, 0.15, 0.5, 0.6, -0.2])
    conserved = []
    for _ in range(200):
        b, f, h, w, bp, hp, wp = y
        fp, bpp, hpp, wpp = rhs(r, b, f, h, w, bp, hp, wp, alpha_gb)
        _, e_f, _, _, _ = equations(r, (b, f, h, w), (bp, fp, hp, wp), (bpp, 0.0, hpp, wpp), alpha_gb)
        conserved.append(np.sqrt(b*h)*f*e_f)
        y = rk4_step(r, y, 1e-3)
        r += 1e-3
    conserved = np.array(conserved)
    assert np.max(np.abs(conserved-conserved[0])) < 1e-9*abs(conserved[0])
