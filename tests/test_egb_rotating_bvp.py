"""Regression tests for the Hito 4B rotating EGB spectral solver.

Resolution is kept low (n=16) to keep the suite fast; the equations here
are much larger than vacuum's (Jacobian entries call the generated EGB
rhs/horizon code, not a small closed form), so each solve() costs several
seconds even at this resolution -- unavoidable given the system's size, not
a bug (see docs/egb-rotating.md).
"""
import numpy as np
import pytest

from rotating_bh.egb_rotating_bvp import SolverFailure, solve
from rotating_bh.egb_rotating_validation import diagnose, tensor_residual
from rotating_bh.myers_perry import MyersPerry

N = 16


def test_alpha_zero_reproduces_myers_perry():
    """At alpha_GB=0 the system reduces exactly to vacuum; the solver's own
    equations have no special knowledge of Myers-Perry, so recovering it
    from a trivial (non-Myers-Perry) initial guess is a real check.
    """
    omega_h = 0.3
    sol = solve(omega_h, 0.0, resolution=N)
    assert sol.boundary_residual < 1e-8
    mp = MyersPerry(r_h=1.0, omega_h=omega_h)
    for r in (1.5, 2.0, 5.0, 10.0):
        zz = 1/r
        x = 1-zz
        B, F, H, W = sol.evaluate(np.array([x]))
        b = (1-zz**2)*B[0]
        f = (1-zz**2)*F[0]
        w = zz**4*W[0]
        expected = mp.functions(np.array([r]))
        assert b == pytest.approx(expected['b'][0], abs=1e-6)
        assert f == pytest.approx(expected['f'][0], abs=1e-6)
        assert w == pytest.approx(expected['w'][0], abs=1e-6)


def test_continuation_reaches_nonzero_alpha_gb():
    """The actual Hito 4B goal: a genuine alpha_GB!=0 rotating EGB point,
    reached by continuation from the (alpha_GB=0) Myers-Perry solution.
    """
    omega_h = 0.3
    sol = solve(omega_h, 0.0, resolution=N)
    for alpha_gb in (0.05, 0.1):
        sol = solve(omega_h, alpha_gb, resolution=N, previous=sol)
        assert sol.boundary_residual < 1e-8
    assert sol.alpha_gb == 0.1


def test_converged_solution_satisfies_the_independent_tensor_equations():
    """The strongest available check: G+alpha_GB H=0 via einstein.py's
    tensor evaluator, completely independent of the action/rhs() route
    used to derive and solve the system.
    """
    omega_h, alpha_gb = 0.3, 0.1
    sol = solve(omega_h, 0.0, resolution=N)
    sol = solve(omega_h, alpha_gb, resolution=N, previous=sol)
    result = diagnose(sol, alpha_gb, count=11)
    assert result['max_tensor_residual'] < 1e-6, result


@pytest.mark.parametrize('x', [0.15, 0.5, 0.85])
def test_tensor_residual_small_at_individual_points(x):
    omega_h, alpha_gb = 0.3, 0.05
    sol = solve(omega_h, 0.0, resolution=N)
    sol = solve(omega_h, alpha_gb, resolution=N, previous=sol)
    assert tensor_residual(sol, x, alpha_gb) < 1e-6


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        solve(0.3, -0.1)
    with pytest.raises(ValueError):
        solve(float('nan'), 0.1)
    with pytest.raises(ValueError):
        solve(0.3, 0.1, resolution=4)
    with pytest.raises(ValueError):
        solve(0.3, 0.1, tol=-1.0)
    with pytest.raises(ValueError):
        solve(0.3, 0.1, max_iterations=-1)


def test_reports_real_resource_exhaustion():
    with pytest.raises(SolverFailure):
        solve(0.3, 0.5, resolution=N, max_iterations=0)
