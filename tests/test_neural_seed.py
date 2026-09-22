"""Check the neural seed machinery before any result rests on it.

Three things have to be true before a trained network means anything: the
Myers-Perry anchor really is the alpha_gb=0 solution, the objective really is
the solver's own residual system rather than a re-derivation of it, and the
hand-written gradient really is the gradient. Each is checked against an
independent oracle, not against itself.
"""
import numpy as np
import pytest

from rotating_bh.egb_rotating_bvp import solve
from rotating_bh.neural_seed import (LAYERS, MyersPerryAnchor, Problem, Seed,
                                     TrivialSeed, anchored_fields, check_gradient,
                                     initial_parameters, multipliers, n_parameters)

OMEGA_H, ALPHA_GB = .3, .1


@pytest.fixture(scope='module')
def problem():
    return Problem(OMEGA_H, ALPHA_GB)


@pytest.mark.parametrize('omega_h', [.1, .3, .5])
def test_anchor_is_the_alpha_zero_solution(omega_h):
    """The anchor must be Myers-Perry, checked against the solver itself."""
    solution = solve(omega_h, 0., resolution=32, tol=1e-11, max_iterations=50)
    x = np.linspace(0, 1, 41)
    anchor = MyersPerryAnchor(omega_h)
    assert np.max(np.abs(anchor(x)[0] - solution.evaluate(x))) < 1e-12


def test_anchor_derivatives_match_finite_differences():
    anchor = MyersPerryAnchor(OMEGA_H)
    x = np.linspace(.1, .9, 17)
    h = 1e-5
    value, first, second = anchor(x)
    plus, minus = anchor(x+h)[0], anchor(x-h)[0]
    assert np.max(np.abs(first - (plus-minus)/(2*h))) < 1e-8
    assert np.max(np.abs(second - (plus-2*value+minus)/h**2)) < 1e-5


def test_anchor_rejects_an_extremal_or_invalid_spin():
    for bad in [1/np.sqrt(2), 1., np.nan, np.inf]:
        with pytest.raises(ValueError):
            MyersPerryAnchor(bad)


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4, 5])
def test_boundary_conditions_hold_for_any_weights(seed):
    """The five conditions built into the ansatz must hold without training.

    This is the load-bearing property: the optimiser never spends effort on the
    boundaries and no amount of training can break them. The two derivative
    conditions come from the network taking u = z^2 as its input, so they are
    tested through the real network rather than through the ansatz algebra with
    made-up derivative arrays, which would not exercise that mechanism.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0., 2., n_parameters())
    seeded = Seed(Problem(OMEGA_H, ALPHA_GB), theta)
    x = np.array([1., 0.])                       # infinity, then the horizon
    value, first = seeded.evaluate(x), seeded.evaluate(x, 1)
    assert value[0][0] == 1.                     # B = 1 at infinity
    assert value[1][0] == 1.                     # F = 1 there
    assert first[2][0] == 0.                     # dH/dx = 0 there
    assert first[3][0] == 0.                     # dW/dx = 0 there
    assert value[3][1] == OMEGA_H                # W = omega_h at the horizon


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_values_free_at_infinity_really_are_free(seed):
    """H and W have only derivative conditions at infinity; the solver never
    fixes their values there, and those values move strongly with the coupling.
    An ansatz that pinned them could not represent the solution at all -- an
    earlier one did, which is what capped the fit before this was found."""
    rng = np.random.default_rng(seed)
    problem = Problem(OMEGA_H, ALPHA_GB)
    anchor_at_infinity = MyersPerryAnchor(OMEGA_H)(np.array([1.]))[0]
    moved = set()
    for _ in range(4):
        value = Seed(problem, rng.normal(0., 2., n_parameters())).evaluate(np.array([1.]))
        moved.add((round(float(value[2][0]), 9), round(float(value[3][0]), 9)))
        assert value[0][0] == 1. and value[1][0] == 1.
    assert len(moved) == 4, 'H and W at infinity must vary with the weights'
    assert all(pair != (round(float(anchor_at_infinity[2][0]), 9),
                        round(float(anchor_at_infinity[3][0]), 9)) for pair in moved)


def test_untrained_network_is_exactly_myers_perry(problem):
    """With a zero output layer the untrained state is the anchor itself, which
    is what makes the untrained control demanding rather than a straw man."""
    zero = np.zeros(n_parameters())
    x = np.linspace(0, 1, 33)
    assert np.max(np.abs(Seed(problem, zero).evaluate(x)
                         - MyersPerryAnchor(OMEGA_H)(x)[0])) == 0.
    rng = np.random.default_rng(7)
    only_hidden = initial_parameters(rng)
    assert np.max(np.abs(Seed(problem, only_hidden).evaluate(x)
                         - MyersPerryAnchor(OMEGA_H)(x)[0])) == 0.


def test_objective_is_the_solvers_own_system(problem):
    """An accepted solution of `solve` must drive this objective's residuals to
    zero. If the objective were a re-derivation that drifted from the solver's
    system, this is where it would show."""
    solution = solve(OMEGA_H, ALPHA_GB, resolution=32, tol=1e-11, max_iterations=50)

    class Converged:
        def evaluate(self, x, derivative=0):
            return solution.evaluate(np.asarray(x, dtype=float), derivative)

    value = Converged().evaluate(problem.x)
    first = Converged().evaluate(problem.x, 1)
    second = Converged().evaluate(problem.x, 2)
    bulk, horizon, _ = problem._bulk_and_horizon(value, first, second)
    assert np.max(np.abs(bulk)) < 1e-8
    assert np.max(np.abs(horizon)) < 1e-8


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_gradient_matches_the_complex_step_oracle(problem, seed):
    """Every parameter, not a sample. The complex step is exact to rounding, so
    a disagreement is a bug in the backward pass and nothing else."""
    rng = np.random.default_rng(seed)
    theta = initial_parameters(rng) if seed else np.zeros(n_parameters())
    worst, analytic, oracle = check_gradient(problem, theta)
    assert worst < 1e-10, f'worst relative deviation {worst:.3e}'
    assert np.linalg.norm(analytic) > 0
    assert analytic.shape == (n_parameters(),)


def test_gradient_matches_the_oracle_far_from_the_anchor(problem):
    """Away from the anchor the second-derivative paths through tanh dominate;
    they are the part of the backward pass most likely to be wrong."""
    rng = np.random.default_rng(11)
    theta = rng.normal(0., .4, n_parameters())
    worst, _, _ = check_gradient(problem, theta)
    assert worst < 1e-10, f'worst relative deviation {worst:.3e}'


def test_trivial_seed_reproduces_the_solvers_own_fallback():
    """The baseline has to be the solver's real default, or the comparison is
    against something the solver never does."""
    trivial = TrivialSeed(OMEGA_H)
    x = np.linspace(0, 1, 9)
    value = trivial.evaluate(x)
    assert np.all(value[0] == 1.) and np.all(value[1] == 1.)
    assert np.all(value[2] == 0.) and np.all(value[3] == OMEGA_H)
    assert np.all(trivial.evaluate(x, 1) == 0.)
    with_seed = solve(OMEGA_H, ALPHA_GB, resolution=24, tol=1e-9,
                      previous=trivial, max_iterations=50)
    without = solve(OMEGA_H, ALPHA_GB, resolution=24, tol=1e-9, max_iterations=50)
    assert with_seed.iterations == without.iterations


def test_seed_rejects_an_unsupported_derivative_order(problem):
    with pytest.raises(ValueError):
        Seed(problem, np.zeros(n_parameters())).evaluate(np.array([.5]), derivative=3)


def test_parameter_count_matches_the_layer_sizes():
    assert n_parameters() == sum(a*b + b for a, b in zip(LAYERS[:-1], LAYERS[1:]))
    assert n_parameters((1, 3, 2)) == 1*3 + 3 + 3*2 + 2
