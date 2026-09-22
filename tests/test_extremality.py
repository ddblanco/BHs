"""The extremal-limit machinery: invariants, gates, extrapolation.

The physics of Hito 8 lives in two scale-invariant numbers, `y` and `tau`, and
in an extrapolation to `tau=0`. Each of those is a place where an error would
be invisible in the final plot, so each is tested against something that does
not depend on the measurement: the exact scaling symmetry of the system, a
function whose intercept is known, and the unchanged Hito 4 acceptance.
"""
import numpy as np
import pytest

from rotating_bh.egb_rotating_scale import scaled_diagnose, tensor_parts
from rotating_bh.extremality import (BOUNDARY_GATE, MASS_COEFFICIENT,
                                     PUBLISHED_SHIFT, RELATIVE_TENSOR_GATE,
                                     TENSOR_GATE, extrapolate, extremal_state,
                                     invariants, smarr_residual, state_at)
from rotating_bh.gb_response import predictor_ladder


def test_the_invariants_are_invariant():
    """r_H -> lambda r_H with alpha -> lambda^2 alpha must not move y or tau.

    Applied to a synthetic state rather than a solved one, because what is
    being tested is the algebra of the scaling weights: E and alpha carry
    length^2, J and S length^3, T one over length.
    """
    base = dict(alpha_gb=.1, E=2.0, J=1.1, S=6.9, T_H=.05, A_H=27.6)
    reference = invariants(base)
    for scale in (.3, 2.5):
        scaled = dict(alpha_gb=.1*scale**2, E=2.0*scale**2, J=1.1*scale**3,
                      S=6.9*scale**3, T_H=.05/scale, A_H=27.6*scale**3)
        moved = invariants(scaled)
        for key in ('y', 'tau', 'mu', 'sigma', 'x', 'j', 't_scaled', 's_scaled'):
            assert moved[key] == pytest.approx(reference[key], rel=1e-13)


def test_a_wrong_scaling_weight_would_be_caught():
    base = dict(alpha_gb=.1, E=2.0, J=1.1, S=6.9, T_H=.05, A_H=27.6)
    wrong = dict(alpha_gb=.1*4, E=2.0*4, J=1.1*8, S=6.9*8, T_H=.05/2,
                 A_H=27.6*8)
    assert invariants(wrong)['y'] == pytest.approx(invariants(base)['y'], rel=1e-13)
    # ... but getting J's weight wrong does move it.
    broken = dict(base, J=1.1*4)
    assert abs(invariants(broken)['y']-invariants(base)['y']) > 1e-3


def test_the_extrapolation_recovers_a_known_intercept():
    temperatures = np.array([.08, .05, .03, .02, .012, .007])
    values = 1.25 - 3.*temperatures + 7.*temperatures**2 - 11*temperatures**3
    result = extrapolate(temperatures, values)
    assert result['value'] == pytest.approx(1.25, abs=1e-9)
    assert result['spread'] < 1e-2


def test_the_extrapolation_reports_a_spread_it_cannot_hide():
    """A value whose degrees disagree must come back with a large spread."""
    temperatures = np.array([.08, .05, .03, .02, .012, .007])
    values = 1.25 + np.sqrt(temperatures)     # not analytic at T=0
    result = extrapolate(temperatures, values)
    assert result['spread'] > 1e-3


def test_the_relative_residual_reduces_to_the_absolute_one_when_the_scale_is_one():
    """The substituted gate must agree with the Hito 4 one where both apply.

    Away from extremality the curvature scale is O(1)-O(10), so
    relative < 1e-8 implies absolute < 1e-6; this checks the two numbers are
    consistent on an actual solution rather than taking the argument on trust.
    """
    solution = predictor_ladder(.3, .1, step=.01, resolution=32)
    checked = scaled_diagnose(solution, .1)
    assert checked['max_relative_tensor_residual']*checked['max_tensor_scale'] == \
        pytest.approx(checked['max_tensor_residual'], rel=1e-12)
    assert checked['max_tensor_residual'] < TENSOR_GATE
    assert checked['max_relative_tensor_residual'] < RELATIVE_TENSOR_GATE


def test_tensor_parts_splits_what_it_should():
    solution = predictor_ladder(.3, .1, step=.01, resolution=32)
    total, einstein, lovelock, kretschmann = tensor_parts(solution, .5, .1)
    assert total < 1e-9*max(einstein, lovelock)
    assert einstein > 0 and lovelock > 0 and kretschmann > 0


def test_the_scale_does_not_vanish_in_the_vacuum():
    """On a vacuum solution G and Ric are identically zero.

    Using them as the scale turns an exact solution into a relative error of
    one, which is exactly what the first Hito 8 production run hit. The
    Kretschmann invariant is what keeps the scale honest there.
    """
    solution = predictor_ladder(.6, 0., step=.01, resolution=48)
    total, einstein, lovelock, kretschmann = tensor_parts(solution, .5, 0.)
    assert lovelock == 0.
    assert einstein < 1e-9
    assert kretschmann > 1e-2
    checked = scaled_diagnose(solution, 0.)
    assert checked['max_relative_tensor_residual'] < RELATIVE_TENSOR_GATE


def test_state_at_refuses_an_unaccepted_solution():
    """The gate has to bite. A solution evaluated at the wrong coupling is not
    a solution of anything, and must not come back as a state."""
    solution = predictor_ladder(.3, .1, step=.01, resolution=32)
    accepted = state_at(solution)
    assert accepted['max_relative_tensor_residual'] < RELATIVE_TENSOR_GATE
    assert accepted['boundary_residual'] < BOUNDARY_GATE
    solution.alpha_gb = .3
    with pytest.raises(RuntimeError):
        state_at(solution)


def test_smarr_closes_on_a_measured_state():
    solution = predictor_ladder(.3, .1, step=.01, resolution=32)
    state = state_at(solution)
    assert state['smarr'] < 1e-6


def test_extremal_state_propagates_the_spreads():
    states = []
    for index, temperature in enumerate([.06, .04, .025, .015, .009, .005]):
        states.append(dict(alpha_gb=.1, omega_h=.6+.01*index, T_H=temperature,
                           E=2.0-temperature, J=1.1-2*temperature, A_H=27.6,
                           S=6.9-3*temperature, psi_gb=1.0+temperature,
                           max_tensor_residual=1e-9,
                           max_relative_tensor_residual=1e-11,
                           absolute_gate_passed=True, smarr=1e-9))
    result = extremal_state(states)
    assert result['E']['value'] == pytest.approx(2.0, abs=1e-9)
    assert result['mu'] == pytest.approx(2.0/1.1**(2/3), rel=1e-7)
    assert result['sigma'] == pytest.approx(6.9/1.1, rel=1e-7)
    assert result['y_spread'] >= 0 and result['mu_spread'] >= 0


def test_the_published_anchors_are_the_published_numbers():
    assert MASS_COEFFICIENT == pytest.approx(1.5*np.pi**(1/3), rel=1e-15)
    assert PUBLISHED_SHIFT == pytest.approx(np.pi, rel=1e-15)
