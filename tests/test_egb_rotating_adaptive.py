"""Independent oracles for the adaptive rotating EGB route."""
from functools import lru_cache

import numpy as np
import pytest
import sympy as sp

from rotating_bh.myers_perry import MyersPerry


@lru_cache(maxsize=1)
def reference_functions():
    x = sp.symbols('x')
    q = sp.Rational(3, 10)
    a = q**2 / (1-q**2)
    z = 1-x
    fields = [(1-a*z**2)/(1+a*z**4), 1-a*z**2,
              a, q/(1-q**2+q**2*z**4)]
    return [[sp.lambdify(x, sp.diff(u, x, k), 'numpy') for u in fields]
            for k in range(3)]


def reference(x, derivative=0):
    x = np.asarray(x)
    return np.asarray([np.broadcast_to(f(x), x.shape)
                       for f in reference_functions()[derivative]])


def reference_state(x):
    return np.concatenate((reference(x), reference(x, 1)[[0, 2, 3]]))


def test_compact_oracle_matches_closed_myers_perry():
    x = np.linspace(.03, .97, 21)
    z = 1-x
    B, F, H, W = reference(x)
    closed = MyersPerry(r_h=1., omega_h=.3).functions(1/z)
    for value, name in zip([(1-z*z)*B, (1-z*z)*F,
                            (1+z**4*H)/z**2, z**4*W], ['b', 'f', 'h', 'w']):
        np.testing.assert_allclose(value, closed[name], atol=2e-14, rtol=2e-14)


def test_ode_matches_symbolic_myers_perry_derivatives():
    from rotating_bh.egb_rotating_adaptive import ode
    x = np.linspace(.02, .98, 31)
    expected = np.concatenate((reference(x, 1), reference(x, 2)[[0, 2, 3]]))
    np.testing.assert_allclose(ode(x, reference_state(x), 0.), expected,
                               atol=2e-8, rtol=2e-8)


def test_boundary_transport_converges_on_closed_myers_perry():
    from rotating_bh.egb_rotating_adaptive import boundary
    errors = []
    for cutoff in [.01, .005, .0025]:
        residual = boundary(reference_state(cutoff), reference_state(1-cutoff),
                            .3, 0., cutoff)
        assert residual.shape == (7,)
        errors.append(np.max(np.abs(residual)))
    assert errors[1] < .3*errors[0]
    assert errors[2] < .3*errors[1]
    assert errors[2] < 1e-3


def test_adaptive_recovers_myers_perry_from_perturbed_seed():
    from rotating_bh.egb_rotating_adaptive import solve
    from rotating_bh.egb_rotating_validation import diagnose

    class Seed:
        def evaluate(self, x, derivative=0):
            return reference(x, derivative)*1.001

    solution = solve(.3, 0., previous=Seed(), cutoff=.003, tol=1e-7)
    x = np.linspace(.02, .98, 151)
    assert np.max(np.abs(solution.evaluate(x)-reference(x))) < 1e-5
    assert solution.boundary_residual < 1e-8
    assert diagnose(solution, 0., count=17)['max_tensor_residual'] < 1e-5


@pytest.fixture(scope='module')
def egb_pair():
    from rotating_bh.egb_rotating_adaptive import solve
    from rotating_bh.egb_rotating_bvp import solve as spectral
    seed = spectral(.3, 0., resolution=24, tol=1e-11)
    for alpha in [.02, .05, .1]:
        seed = spectral(.3, alpha, resolution=24, tol=1e-11, previous=seed)
    return seed, solve(.3, .1, previous=seed, cutoff=.003, tol=1e-7)


def test_egb_rotating_methods_agree(egb_pair):
    spectral, adaptive = egb_pair
    x = np.linspace(.02, .98, 151)
    assert np.max(np.abs(adaptive.evaluate(x)-spectral.evaluate(x))) < 1e-5
    assert adaptive.boundary_residual < 1e-8
    assert adaptive.collocation_residual < 1e-7


def test_adaptive_egb_passes_independent_tensor_and_negative_control(egb_pair):
    from rotating_bh.egb_rotating_validation import diagnose, tensor_residual
    _, adaptive = egb_pair
    assert diagnose(adaptive, .1, count=31)['max_tensor_residual'] < 1e-5
    assert tensor_residual(adaptive, .5, .15) > 1e-4


def test_angular_current_is_conserved_by_both_methods(egb_pair):
    from rotating_bh.egb_rotating_angular import first_integral
    from rotating_bh._egb_rotating_horizon_generated import horizon
    x = np.linspace(.02, .98, 101)
    for solution in egb_pair:
        current = first_integral(x, solution.evaluate(x), solution.evaluate(x, 1), .1)
        assert np.ptp(current) < 1e-6
    spectral, _ = egb_pair
    values, first, second = [spectral.evaluate(0., k) for k in range(3)]
    Ew = horizon(*values, *first[[0, 2, 3]], first[1], second[3], .1)[3]
    assert abs(Ew) < 1e-7


def test_adaptive_derivatives_describe_the_returned_profile(egb_pair):
    _, adaptive = egb_pair
    x = np.linspace(.025, .975, 73)
    step = 1e-6
    for derivative in [1, 2]:
        difference = (adaptive.evaluate(x+step, derivative-1)
                      - adaptive.evaluate(x-step, derivative-1))/(2*step)
        np.testing.assert_allclose(difference, adaptive.evaluate(x, derivative),
                                   atol=1e-7, rtol=1e-6)
    x = adaptive.nodes[1:-1]
    jump = adaptive.evaluate(x+1e-10, 2)-adaptive.evaluate(x-1e-10, 2)
    assert np.max(np.abs(jump[[0, 2, 3]])) < 1e-7


@pytest.mark.parametrize('options', [
    {'omega_h': np.nan}, {'omega_h': True}, {'alpha_gb': -.1},
    {'alpha_gb': np.inf}, {'alpha_gb': False}, {'tol': 0}, {'tol': np.nan},
    {'resolution': 7}, {'resolution': 8.5}, {'resolution': True},
    {'cutoff': 0}, {'cutoff': .1}, {'cutoff': np.nan},
    {'max_nodes': 8}, {'max_nodes': True}, {'max_nodes': 81.5},
])
def test_rejects_invalid_solver_parameters(options):
    from rotating_bh.egb_rotating_adaptive import solve
    class Seed:
        evaluate = staticmethod(reference)
    arguments = dict(omega_h=.3, alpha_gb=.1, previous=Seed())
    arguments.update(options)
    with pytest.raises(ValueError):
        solve(**arguments)


def test_reports_real_resource_exhaustion():
    from rotating_bh.egb_rotating_adaptive import solve, SolverFailure

    class Seed:
        evaluate = staticmethod(reference)

    with pytest.raises(SolverFailure, match='nodes='):
        solve(.3, 0., previous=Seed(), resolution=8, max_nodes=8, tol=1e-10)


def test_rejects_extrapolation_and_invalid_derivative(egb_pair):
    _, adaptive = egb_pair
    for x in [0., 1., np.nan]:
        with pytest.raises(ValueError):
            adaptive.evaluate(x)
    with pytest.raises(ValueError):
        adaptive.evaluate(.5, 3)
