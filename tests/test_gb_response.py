"""The exact coupling derivatives, and the predictor's regression test.

`gb_response` duplicates the observable formulas of
`egb_rotating_observables.measure` in order to differentiate them. Duplication
that drifts is worse than no duplication, so the first tests here pin the two
together by finite differences of `measure` itself -- if the entropy's explicit
Wald term were dropped, or the constant in the mass tail carried over into its
derivative, these fail.

The last test is the one `plans/2026-09-14-hito-8.md` asks for by name: it must
fail if the continuation goes back to seeding from the previous solution.
"""
import numpy as np
import pytest

from rotating_bh.egb_rotating_bvp import SolverFailure
from rotating_bh.egb_rotating_predictor import solve
from rotating_bh.egb_rotating_observables import measure
from rotating_bh.gb_potential import VacuumSeed, perturbative_potential
from rotating_bh.gb_response import (TAIL_WINDOWS, measured_perturbative_potential,
                                     predictor_ladder, response_field,
                                     response_observables)


def _central(omega_h, alpha_gb, step, resolution=32):
    base = predictor_ladder(omega_h, alpha_gb, step=.01, resolution=resolution)
    centre = measure(base)
    neighbours = {}
    for sign in (-1, 1):
        neighbours[sign] = measure(predictor_ladder(
            omega_h, alpha_gb+sign*step, step=.01, resolution=resolution))
    derivative = {k: (neighbours[1][k]-neighbours[-1][k])/(2*step)
                  for k in ('E', 'S', 'J')}
    derivative['psi'] = (derivative['E'] - centre['T_H']*derivative['S']
                         - 2*omega_h*derivative['J'])
    return derivative


@pytest.mark.parametrize('omega_h,alpha_gb', [(.3, .1), (.5, .1)])
def test_the_exact_derivatives_match_central_differences(omega_h, alpha_gb):
    base = predictor_ladder(omega_h, alpha_gb, step=.01, resolution=32)
    exact = response_observables(base)
    approximate = _central(omega_h, alpha_gb, .002)
    for key, name in (('E', 'dE'), ('S', 'dS'), ('J', 'dJ')):
        scale = max(1., abs(exact[name]))
        assert abs(exact[name]-approximate[key]) < 2e-4*scale
    assert abs(exact['psi_gb']-approximate['psi']) < 2e-3*max(
        1., abs(exact['psi_gb']))


def test_dropping_the_explicit_wald_term_would_be_caught():
    """S = (A/4)(1+4 alpha (3-H)) depends on the coupling twice over.

    Keeping only the profile's contribution gives a smooth, plausible, wrong
    answer; the central difference sees the difference immediately.
    """
    base = predictor_ladder(.3, .1, step=.01, resolution=32)
    field = response_field(base)
    exact = response_observables(base, field)
    _, _, H_H, _ = base.evaluate(0.)
    profile_only = exact['dS'] - exact['A_H']/4*4*(3-H_H)
    approximate = _central(.3, .1, .002)
    assert abs(exact['dS']-approximate['S']) < 1e-3*abs(exact['dS'])
    assert abs(profile_only-approximate['S']) > .1*abs(approximate['S'])


@pytest.mark.parametrize('q', [.0, .3, .5, .65])
def test_the_perturbative_potential_reproduces_the_closed_form(q):
    """Psi_0(q) at alpha_GB=0, with no extrapolation anywhere.

    Hito 6 reached three or four digits here by extrapolating a ladder of small
    couplings. One linear solve at the exact Myers-Perry profile reaches eight
    or more, which is what turns a surviving hypothesis into a settled one.
    """
    measured = measured_perturbative_potential(q, resolution=48)['psi_gb']
    oracle = float(perturbative_potential(q))
    assert abs(measured-oracle) < 1e-7*max(abs(oracle), 1.)


def test_the_oracle_comparison_can_fail():
    measured = measured_perturbative_potential(.3, resolution=48)['psi_gb']
    oracle = float(perturbative_potential(.3))
    assert abs(measured-1.001*oracle) > 1e-7*abs(oracle)


def test_the_tail_windows_agree_with_each_other():
    base = predictor_ladder(.3, .1, step=.01, resolution=32)
    field = response_field(base)
    values = [response_observables(base, field, window=w)['psi_gb']
              for w in range(len(TAIL_WINDOWS))]
    assert np.ptp(values) < 1e-6*max(1., abs(values[0]))


def test_the_predictor_reaches_what_previous_seeding_cannot():
    """The regression test the Hito 8 plan asks for by name.

    `q=0.70` at `alpha_GB=0.02` fails by every route that existed before --
    single jump, fine ladder and `gb_potential.reach` -- and the addendum
    established that shrinking the step does not help, because what is wrong is
    the starting direction. If the predictor is ever removed or silently
    reverts to seeding from the previous solution, the first assertion here
    fails; if the failure it is contrasted against ever stops happening, the
    second one does, and the test stops claiming something that is no longer
    true.
    """
    rungs = np.linspace(0., .02, 5)

    def climb(seed):
        current = solve(.70, 0., resolution=32, tol=1e-11,
                        previous=VacuumSeed(.70), jacobian='analytic')
        for value in rungs[1:]:
            current = solve(.70, float(value), resolution=32, tol=1e-11,
                            previous=current, jacobian='analytic', seed=seed)
        return current

    reached = climb('predictor')
    assert abs(reached.alpha_gb-.02) < 1e-12
    assert reached.boundary_residual < 1e-8
    with pytest.raises(SolverFailure):
        climb('previous')
