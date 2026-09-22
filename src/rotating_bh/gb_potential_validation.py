"""Recompute the Hito 6 gates from the recorded evidence.

Same discipline as the Hito 4C and Hito 5 validators: gates are derived from the
numbers in the artifact, never read back from booleans the producing run stored,
and every threshold is pinned here rather than taken from the artifact, so a
later run cannot widen its own gate and still be called verified.

`oracle_confirmed` is deliberately allowed to fail. The perturbative formula it
tests is recorded in `reports/originality-precheck.md` as unvalidated; if the
measured `alpha_GB -> 0` limit disagrees with it, the honest outcome is a failing
gate and a reported refutation, not an adjusted comparison.
"""
import numpy as np

SMARR_GATE = 1e-4            # relative, at finite coupling
SMARR_VACUUM_GATE = 1e-12    # relative, at alpha_gb=0 against closed forms
CONVERGENCE_FACTOR = 2.5     # halving the step must shrink the change this much
ORACLE_SIGMA = 3.0           # how many extrapolation uncertainties count as agreement
RESOLUTION_GATE = .05        # absolute, on Psi_GB between N=32 and N=40


def _vacuum_smarr(data):
    """2E = 3TS + 6 Omega J must hold against the Hito 3 closed forms. This is
    what fixes the relation's coefficients instead of assuming them."""
    rows = data['vacuum_smarr']
    return bool(rows and all(r['relative'] < SMARR_VACUUM_GATE for r in rows))


def _step_convergence(data):
    """Psi_GB must settle as the difference step is refined. A value that keeps
    moving is not a measurement."""
    rows = data['step_refinement']
    if not rows:
        return False
    for row in rows:
        values = [entry['psi_gb'] for entry in row['ladder']]
        steps = [entry['step'] for entry in row['ladder']]
        if len(values) < 3 or not all(b < a for a, b in zip(steps, steps[1:])):
            return False
        first = abs(values[1]-values[0])
        last = abs(values[-1]-values[-2])
        if not (last*CONVERGENCE_FACTOR <= first or last < 1e-9):
            return False
    return True


def _smarr_closes(data):
    return bool(data['points'] and all(p['smarr']['relative'] < SMARR_GATE
                                       for p in data['points']))


def _gates_respected(data):
    return bool(data['points']
                and data['tensor_gate'] <= 1e-6 and data['boundary_gate'] <= 1e-8
                and all(p['max_tensor_residual'] < 1e-6 for p in data['points']))


def _oracle_confirmed(data):
    """Does the measured alpha->0 limit reproduce the unvalidated oracle?

    Judged against the extrapolation's *own* uncertainty, not against a relative
    tolerance. The oracle changes sign inside the sampled range of spins, so a
    relative criterion is ill-conditioned exactly where the interesting physics
    is: near the zero, any tolerance relative to a vanishing quantity fails for
    arithmetic reasons rather than physical ones. The spread between a linear and
    a quadratic fit of the same samples is a usable estimate of how well the
    limit is pinned down, so agreement means the disagreement is not resolved by
    the measurement.

    This also means the test can be non-discriminating -- a loose extrapolation
    agrees with almost anything. `_oracle_test_is_discriminating` reports that
    separately instead of letting a weak measurement pass as confirmation.
    """
    rows = data['vanishing_coupling']
    return bool(rows and all(
        r['absolute_deviation'] <= ORACLE_SIGMA*r['extrapolation_spread'] + 1e-9
        for r in rows))


def _oracle_test_is_discriminating(data):
    """Was the alpha->0 limit pinned down well enough for the comparison to mean
    anything? The uncertainty has to be small against the scale over which the
    oracle itself varies across the sampled spins."""
    rows = data['vanishing_coupling']
    if len(rows) < 2:
        return False
    scale = max(r['oracle'] for r in rows) - min(r['oracle'] for r in rows)
    return bool(scale > 0 and all(
        r['extrapolation_spread'] < .1*scale for r in rows))


def _resolution_independent(data):
    """Psi_GB must not depend on the spectral resolution.

    Checked at the points where the measurement is weakest, not the comfortable
    ones. The grid spans Psi_GB over about five units, so a disagreement of 0.05
    between N=32 and N=40 is already a percent of the range being measured.
    """
    rows = data.get('resolution_check', [])
    return bool(rows and all(r['absolute_difference'] < RESOLUTION_GATE for r in rows))


def _refinement_covers_the_hard_points(data):
    """The step refinement has to include the high-spin points. Refining only
    where the answer is well behaved would certify nothing about the region the
    result actually depends on."""
    rows = data['step_refinement']
    return bool(rows and max(r['omega_h'] for r in rows) >= .6)


def _sign_change_located(data):
    """The locus must be bracketed by a genuine change of sign, not interpolated
    from same-signed points.

    The opposite-sign requirement is the real check -- it is what makes a
    manufactured locus impossible -- and it stays strict. The containment is
    inclusive, because a root can legitimately land on a bracket endpoint: at
    alpha_gb = 0.2 the potential at q = 0.68 is +0.0027, so the root sits
    essentially *at* the sampled spin and the search returns that endpoint. An
    earlier strict inequality rejected that, which was the wrong assertion about
    a correctly bracketed root rather than a real defect being caught.
    """
    rows = data.get('sign_change', [])
    if not rows:
        return False
    for row in rows:
        if row['bracket_low']['psi_gb']*row['bracket_high']['psi_gb'] >= 0:
            return False
        if not row['bracket_low']['q'] <= row['q_star'] <= row['bracket_high']['q']:
            return False
    return True


def _spread_reported(data):
    """Every requested (q, alpha) point must be present, so a subset that
    behaved cannot stand in for the measurement."""
    expected = {(q, a) for q in data['spins'] for a in data['couplings']}
    present = {(p['omega_h'], p['alpha_gb']) for p in data['points']}
    return present == expected and len(expected) > 1


def checks(data):
    return [
        dict(name='smarr_exact_in_the_vacuum_limit', passed=_vacuum_smarr(data)),
        dict(name='psi_converges_under_step_refinement', passed=_step_convergence(data)),
        dict(name='smarr_closes_at_finite_coupling', passed=_smarr_closes(data)),
        dict(name='every_solution_passes_the_hito_4_gates', passed=_gates_respected(data)),
        dict(name='all_requested_points_reported', passed=_spread_reported(data)),
        dict(name='refinement_covers_the_hard_points',
             passed=_refinement_covers_the_hard_points(data)),
        dict(name='psi_independent_of_resolution', passed=_resolution_independent(data)),
        dict(name='sign_change_bracketed', passed=_sign_change_located(data)),
        dict(name='oracle_test_is_discriminating',
             passed=_oracle_test_is_discriminating(data)),
        dict(name='perturbative_oracle_confirmed', passed=_oracle_confirmed(data)),
    ]
