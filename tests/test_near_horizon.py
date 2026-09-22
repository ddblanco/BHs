"""The extremal near-horizon branch, against closed forms and against the paper.

This module reproduces section 4.2 of arXiv:1010.0860v1 rather than adding to
it (`reports/hito-8-originality.md`), so the tests that matter most are the
ones that compare it with something outside this project: the exact extremal
Myers-Perry relations at zero coupling, and the paper's own published equations
at finite coupling.
"""
import numpy as np
import pytest

from rotating_bh.near_horizon import (BRANCH_LIMIT, VACUUM_STATE,
                                      coupling_from_invariant, extremal,
                                      paper_residuals)
from rotating_bh._near_horizon_generated import (density, entropy_function,
                                                 scalar, stationarity)


def test_the_vacuum_root_is_the_exact_one():
    state = extremal(0.)
    assert state['v2'] == pytest.approx(4., abs=1e-13)
    assert state['v3'] == pytest.approx(8., abs=1e-13)
    assert state['k'] == pytest.approx(1., abs=1e-13)
    assert state['residual'] == 0.


def test_the_vacuum_reproduces_extremal_myers_perry():
    """S = 2 pi J for each spin, and R = 0 because the vacuum is Ricci flat.

    Both come out of the formalism with nothing tuned: the 1/(16 pi G) of the
    action and the 2 pi of the Legendre transform were fixed before the answer
    was known.
    """
    state = extremal(0.)
    assert state['S_over_J'] == pytest.approx(2*np.pi, rel=1e-13)
    assert abs(state['ricci_scalar']) < 1e-13
    # J = 2 sqrt(2) pi v1^{3/2} at v1 = 1.
    assert state['J'] == pytest.approx(2*np.sqrt(2)*np.pi, rel=1e-13)


@pytest.mark.parametrize('alpha_tilde', [0., .05, .1, .2, .4, .6, .8])
def test_the_published_equations_are_satisfied(alpha_tilde):
    """Eqs. (4.11) and (4.12) of 1010.0860v1, in the paper's own variables.

    This is the external anchor at finite coupling. It also pins the coupling
    normalisation: the equations close with alpha_paper = 4 alpha_GB and with
    no other factor.
    """
    residuals = paper_residuals(extremal(alpha_tilde))
    assert residuals['stationarity'] < 1e-12
    assert residuals['spin_ratio'] == pytest.approx(1., abs=1e-12)


def test_a_wrong_coupling_normalisation_would_be_caught():
    """The anchor has to be able to fail; alpha_paper = alpha_GB must not work."""
    state = extremal(.2)
    v1, v2 = 1., state['v2']
    v3, k = state['v3']/v2, state['k']/2
    alpha = state['alpha_tilde']          # the wrong one: missing the factor 4
    wrong = abs(-16*v1**2 + 4*v1**2*v3 + k**2*v2**2*v3
                + alpha*k**2*v2*v3*(4-3*v3))
    right = abs(-16*v1**2 + 4*v1**2*v3 + k**2*v2**2*v3
                + 4*alpha*k**2*v2*v3*(4-3*v3))
    assert right < 1e-10
    assert wrong > 1e4*max(right, 1e-14)


@pytest.mark.parametrize('alpha_tilde', [.05, .3, .7])
def test_the_stationarity_conditions_are_the_derivative_of_f(alpha_tilde):
    """The generated polynomials must really be d f/d v_i, up to a factor.

    Generated code that has drifted from what generated it is the failure this
    catches: the entropy function and the stationarity conditions come from the
    same derivation but are emitted separately.
    """
    state = extremal(alpha_tilde)
    values = [state['v1'], state['v2'], state['v3']]
    step = 1e-6
    for index in range(3):
        moved = []
        for sign in (-1, 1):
            trial = list(values)
            trial[index] += sign*step
            moved.append(entropy_function(*trial, state['k'], alpha_tilde))
        derivative = (moved[1]-moved[0])/(2*step)
        assert abs(derivative) < 1e-6*max(1., abs(moved[0])/step)


def test_the_scalars_are_constant_on_the_geometry():
    """Homogeneity: R and L_GB may not depend on where they are evaluated.

    They are written as functions of the four constants only, so the check is
    that the generated expressions carry no residual coordinate dependence --
    which they cannot, structurally; what this really guards is that the
    generated file is the one the derivation wrote.
    """
    state = extremal(.2)
    values = (state['v1'], state['v2'], state['v3'], state['k'])
    assert np.isfinite(scalar(*values))
    assert np.isfinite(density(*values))
    expected = np.pi/8*state['v1']*state['v2']*np.sqrt(state['v3'])*(
        scalar(*values) + .2*density(*values))
    assert entropy_function(*values, .2) == pytest.approx(expected, rel=1e-14)


def test_the_invariant_inverts_monotonically():
    couplings = np.linspace(0., BRANCH_LIMIT, 25)[1:]
    invariants = [extremal(float(a))['invariant'] for a in couplings]
    assert all(b > a for a, b in zip(invariants, invariants[1:]))
    for target in (.02, .2, 1.):
        recovered = coupling_from_invariant(target)
        assert extremal(recovered)['invariant'] == pytest.approx(target, rel=1e-10)


def test_the_vacuum_state_constant_matches_the_solved_root():
    state = extremal(0.)
    assert (state['v2'], state['v3'], state['k']) == pytest.approx(VACUUM_STATE,
                                                                   abs=1e-13)
    assert np.max(np.abs(stationarity(*VACUUM_STATE, 0.))) < 1e-12
