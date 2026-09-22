"""The Hito 8 solver additions: the analytic Jacobian and the predictor seed.

Two things have to be true for the rest of Hito 8 to rest on this solver. The
analytic Jacobian must be the *same matrix* the dense complex-step build
produces -- not merely good enough to converge, since a wrong Jacobian can
still converge to the right answer and would then hide a derivative error that
the linear response depends on. And the default behaviour must be untouched, so
every artifact of Hitos 4-6 keeps reproducing.
"""
import numpy as np
import pytest

from rotating_bh.egb_rotating_bvp import _cheb, _residual
from rotating_bh.egb_rotating_bvp import solve as audited_solve
from rotating_bh.egb_rotating_predictor import (analytic_jacobian, dense_jacobian,
                                                linear_response, solve)
from rotating_bh.gb_potential import VacuumSeed


def _perturbed(omega_h, n, scale=1e-3, seed=0):
    """A profile that is deliberately NOT a solution.

    At a solution many terms of the Jacobian are small and a chain-rule mistake
    can hide; on generic data every term contributes.
    """
    solution = audited_solve(omega_h, 0., resolution=n, tol=1e-11,
                             previous=VacuumSeed(omega_h))
    x, _, _ = _cheb(n)
    flat = np.asarray(solution.evaluate(x)).ravel()
    rng = np.random.default_rng(seed)
    return flat*(1+scale*rng.standard_normal(flat.size))


@pytest.mark.parametrize('omega_h,alpha_gb,n', [(.3, 0., 16), (.3, .1, 24),
                                                (.5, .05, 24)])
def test_the_analytic_jacobian_is_the_dense_one(omega_h, alpha_gb, n):
    x, _, _ = _cheb(n)
    zz = 1-x
    flat = _perturbed(omega_h, n)
    residual = lambda f: _residual(f, n, omega_h, alpha_gb, zz)
    dense = dense_jacobian(residual, flat, n)
    analytic = analytic_jacobian(flat, n, omega_h, alpha_gb, zz)
    assert np.max(np.abs(analytic-dense)) < 1e-12*np.max(np.abs(dense))


def test_a_dropped_chain_rule_term_would_be_caught():
    """The comparison has to be able to fail.

    Zeroing the derivative-argument contribution of one block is exactly the
    kind of mistake the chain-rule assembly invites, and it must show up.
    """
    n, omega_h, alpha_gb = 20, .3, .05
    x, _, _ = _cheb(n)
    zz = 1-x
    flat = _perturbed(omega_h, n)
    analytic = analytic_jacobian(flat, n, omega_h, alpha_gb, zz)
    damaged = analytic.copy()
    damaged[n:2*n, 0:n] = 0.
    dense = dense_jacobian(lambda f: _residual(f, n, omega_h, alpha_gb, zz),
                           flat, n)
    assert np.max(np.abs(damaged-dense)) > 1e-3*np.max(np.abs(dense))


@pytest.mark.parametrize('omega_h,alpha_gb', [(.3, 0.), (.4, 0.)])
def test_both_jacobians_reach_the_same_solution(omega_h, alpha_gb):
    default = solve(omega_h, alpha_gb, resolution=24, tol=1e-11,
                    previous=VacuumSeed(omega_h))
    fast = solve(omega_h, alpha_gb, resolution=24, tol=1e-11,
                 previous=VacuumSeed(omega_h), jacobian='analytic')
    sample = np.linspace(.02, .98, 41)
    assert np.max(np.abs(default.evaluate(sample)-fast.evaluate(sample))) < 1e-10


def test_the_audited_solver_is_untouched():
    """The Hito 4 entry point must still work exactly as it did.

    The additions live in a separate module precisely so that this file's
    bytes, recorded as an input of every artifact from Hito 2 on, do not move.
    """
    solution = audited_solve(.3, 0., resolution=24, tol=1e-11,
                             previous=VacuumSeed(.3))
    assert solution.method == 'spectral'
    assert solution.boundary_residual < 1e-8
    import inspect
    from rotating_bh import egb_rotating_bvp
    signature = inspect.signature(egb_rotating_bvp.solve)
    assert 'jacobian' not in signature.parameters
    assert 'seed' not in signature.parameters
    with pytest.raises(ValueError):
        solve(.3, 0., resolution=24, jacobian='nonsense')
    with pytest.raises(ValueError):
        solve(.3, 0., resolution=24, seed='nonsense')
    with pytest.raises(ValueError):
        solve(.3, .02, resolution=24, seed='predictor')


@pytest.mark.parametrize('omega_h,alpha_gb', [(.3, .02), (.3, .05)])
def test_the_linear_response_matches_a_central_difference(omega_h, alpha_gb):
    """du/dalpha against differences of actual solutions.

    The response is what every exact derivative in Hito 8 is built on, so it is
    checked against the thing it replaces rather than against itself.
    """
    n = 24
    x, _, _ = _cheb(n)
    zz = 1-x
    from rotating_bh.gb_response import predictor_ladder
    base = predictor_ladder(omega_h, alpha_gb, step=.01, resolution=n, tol=1e-12)
    flat = np.asarray(base.evaluate(x)).ravel()
    exact = linear_response(flat, n, omega_h, alpha_gb, zz)
    step = 1e-4
    neighbours = []
    for sign in (-1, 1):
        neighbour = solve(omega_h, alpha_gb+sign*step, resolution=n, tol=1e-12,
                          previous=base, jacobian='analytic')
        neighbours.append(np.asarray(neighbour.evaluate(x)).ravel())
    difference = (neighbours[1]-neighbours[0])/(2*step)
    assert np.max(np.abs(exact-difference)) < 1e-6*max(1., np.max(np.abs(exact)))


def test_the_response_in_the_spin_is_also_exact():
    n, omega_h = 24, .4
    x, _, _ = _cheb(n)
    zz = 1-x
    base = solve(omega_h, 0., resolution=n, tol=1e-12,
                 previous=VacuumSeed(omega_h), jacobian='analytic')
    flat = np.asarray(base.evaluate(x)).ravel()
    exact = linear_response(flat, n, omega_h, 0., zz, parameter='omega_h')
    step = 1e-5
    values = [np.asarray(solve(omega_h+sign*step, 0., resolution=n, tol=1e-12,
                               previous=VacuumSeed(omega_h+sign*step),
                               jacobian='analytic').evaluate(x)).ravel()
              for sign in (-1, 1)]
    difference = (values[1]-values[0])/(2*step)
    assert np.max(np.abs(exact-difference)) < 1e-5*max(1., np.max(np.abs(exact)))


def test_both_responses_come_back_together():
    n, omega_h = 20, .35
    x, _, _ = _cheb(n)
    zz = 1-x
    base = solve(omega_h, 0., resolution=n, tol=1e-12,
                 previous=VacuumSeed(omega_h), jacobian='analytic')
    flat = np.asarray(base.evaluate(x)).ravel()
    pair = linear_response(flat, n, omega_h, 0., zz,
                           parameter=('alpha_gb', 'omega_h'))
    assert pair.shape == (2, 4*n)
    for index, name in enumerate(('alpha_gb', 'omega_h')):
        single = linear_response(flat, n, omega_h, 0., zz, parameter=name)
        assert np.max(np.abs(pair[index]-single)) < 1e-12*max(
            1., np.max(np.abs(single)))
