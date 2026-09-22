from functools import lru_cache

import numpy as np
import pytest
import sympy as sp

from rotating_bh.static_egb import StaticEGB


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        StaticEGB(r_h=0.0, alpha_gb=0.1)
    with pytest.raises(ValueError):
        StaticEGB(r_h=1.0, alpha_gb=-0.1)
    with pytest.raises(ValueError):
        StaticEGB(r_h=1.0, alpha_gb=float('nan'))
    with pytest.raises(ValueError):
        StaticEGB(r_h=-1.0, alpha_gb=0.0)


def test_rejects_radii_inside_the_horizon():
    solution = StaticEGB(r_h=1.0, alpha_gb=0.1)
    with pytest.raises(ValueError):
        solution.q(0.5)
    with pytest.raises(ValueError):
        solution.q(float('inf'))


@pytest.mark.parametrize('alpha_gb', [0.0, 0.05, 0.3, 2.0])
def test_horizon_and_asymptotics(alpha_gb):
    solution = StaticEGB(r_h=1.3, alpha_gb=alpha_gb)
    functions = solution.functions(solution.r_h)
    assert functions['f'] == pytest.approx(0.0, abs=1e-12)
    assert functions['b'] == pytest.approx(0.0, abs=1e-12)
    far = solution.functions(500.0)
    assert far['f'] == pytest.approx(1.0, abs=1e-4)
    assert far['b'] == pytest.approx(1.0, abs=1e-4)
    assert far['h'] == pytest.approx(500.0**2)
    assert far['w'] == pytest.approx(0.0)
    horizon = solution.horizon_data()
    assert horizon['f1'] > 0
    assert horizon['b1'] == pytest.approx(horizon['f1'])


def test_gr_limit_is_schwarzschild_tangherlini():
    gr = StaticEGB(r_h=1.0, alpha_gb=0.0)
    r = np.array([1.2, 2.0, 5.0])
    np.testing.assert_allclose(gr.functions(r)['f'], 1-1/r**2)
    assert gr.mass_coefficient() == pytest.approx(-1.0)


def test_small_alpha_stays_close_to_gr():
    r = np.array([1.05, 1.5, 4.0])
    gr = StaticEGB(r_h=1.0, alpha_gb=0.0).functions(r)['f']
    small = StaticEGB(r_h=1.0, alpha_gb=1e-6).functions(r)['f']
    assert np.max(np.abs(small-gr)) < 1e-5


def test_q_is_monotonically_decreasing_outside_horizon():
    solution = StaticEGB(r_h=1.0, alpha_gb=0.2)
    r = np.linspace(1.0, 20.0, 50)
    q = solution.q(r)
    assert np.all(np.diff(q) < 0)
    assert q[0] == pytest.approx(1.0)


@lru_cache(maxsize=1)
def _q_of_r_symbolic():
    # Solved once with alpha kept symbolic, so sympy carries out the ± of the
    # quadratic formula algebraically instead of taking a numeric shortcut on
    # Float coefficients (which was found, while writing this test, to
    # silently re-lose the precision this reference exists to provide).
    r_h, a, r, q = sp.symbols('r_h alpha r q', positive=True)
    quadratic = sp.Eq(r**2*q+2*a*q**2, r_h**2+2*a)
    roots = sp.solve(quadratic, q)
    physical = [root for root in roots
                if sp.simplify(sp.limit(root, a, 0)-r_h**2/r**2) == 0]
    assert len(physical) == 1
    return r_h, a, r, physical[0]


@pytest.mark.parametrize('alpha_gb', [1e-4, 1e-8, 1e-12, 1e-16, 1e-20, 0.0])
@pytest.mark.parametrize('r', [1.001, 1.5, 20.0, 500.0])
def test_q_matches_high_precision_root_of_the_defining_quadratic(alpha_gb, r):
    # r^2 q+2 alpha_GB q^2=r_H^2+2 alpha_GB (see StaticEGB.q), solved at
    # 50-digit precision and picked as the GR-continuous root, is the ground
    # truth this closed form must reproduce in float64 across a log-spaced
    # sweep down to couplings tiny enough to make the naive
    # r^2/(4a)*(sqrt(1+eps)-1) form cancel two nearly-equal numbers and lose
    # all accuracy (AstraCheck, 2026-09-09).
    solution = StaticEGB(r_h=1.0, alpha_gb=alpha_gb)
    if alpha_gb == 0.0:
        reference = 1.0/r**2  # the formula's own alpha->0 limit; division by
    else:                     # a literal alpha=0 elsewhere would be 0/0.
        r_h, a, r_symbol, physical = _q_of_r_symbolic()
        reference = float(physical.evalf(50, subs={r_h: 1.0, a: alpha_gb, r_symbol: r}))
    assert solution.q(r) == pytest.approx(reference, rel=1e-9, abs=1e-300)
