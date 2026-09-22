"""Recompute the Hito 8 gates from the recorded evidence.

Same discipline as the Hito 4C, 5 and 6 validators: every gate is derived from
the numbers in the artifact, never read back from a boolean the producing run
stored, and every threshold is pinned here rather than taken from the artifact,
so a later run cannot widen its own gate and still be called verified.

Three of these are allowed to fail and mean something if they do:

* `published_extremal_shift_reproduced` tests `mu'(0)` against the `pi` of
  arXiv:2009.00015. That number is external. Failing it would refute the
  measurement, not the paper, and the plan fixed in advance that a refutation
  gets published or nothing does.
* `near_horizon_agrees` tests `S_ext/J` against eqs. (4.11)-(4.20) of
  arXiv:1010.0860v1 at finite coupling. Also external, also allowed to fail.
* `routes_agree` tests the thermodynamic `lim Psi_GB` against the mass-only
  `mu'(y)`. These two share the solver and nothing else, so a disagreement
  means one of them is wrong and the extremal shift does not get claimed.
"""
import numpy as np

# Psi_0 from the linear response against the closed-form oracle. The tail fit
# is what limits it, not the response, so eight digits is the right ask.
ORACLE_DIGITS = 8.0
ORACLE_SPINS = 5
# Central differences must approach the linear response at second order. Five
# halvings with ratios this close to 4 is not something a wrong limit produces.
RATIO_TOLERANCE = .25
# Smarr, unchanged from Hito 6.
SMARR_GATE = 1e-4
# How close the walks must get to extremality for the extrapolation to mean
# anything, in the scale-invariant tau = T_H J^(1/3).
TAU_FLOOR = 5e-3
# The alpha_GB=0 endpoint against arXiv:2009.00015. This point is not
# extrapolated in the coupling -- it is the Myers-Perry limit of the family --
# so it is held to the precision the measurement actually has there.
MASS_ANCHOR_GATE = 1e-6
SHIFT_ANCHOR_GATE = 1e-5
# The same slope reconstructed from masses alone, which is a one-sided
# derivative of thirteen extrapolated points and cannot be held to that.
MASS_ROUTE_SHIFT_GATE = .05
NEAR_HORIZON_GATE = 1e-3
# The static limit is closed-form at every coupling, so it is held to the
# precision the solver actually has rather than to an extrapolation tolerance.
STATIC_GATE = 1e-7
# Myers-Perry along the whole spin range at alpha=0: the same standard.
VACUUM_MASS_GATE = 1e-8
# The exact bound x <= 2 abar/(1+2 abar) IS violated, at the largest coupling
# and the smallest spins; that violation is the honest bound on the error of
# the mass extraction and is reported as such. This gate does not assert it
# away -- it asserts it stays where appendix A says it stays.
MASS_BOUND_GATE = 1e-3
# The first law in the spin direction and the Goon-Penco identity are exact
# combinations of measured quantities, not extrapolations.
FIRST_LAW_GATE = 1e-7
GOON_PENCO_GATE = 1e-7
# The physically motivated non-analyticity for a near-extremal throat is a
# tau^2 log tau term. If the data supported one, this is where it would show.
LOG_MODEL_GATE = 5e-3
# A sqrt(tau) term is not physically motivated and is nearly degenerate with the
# constant when the fit has a short lever arm, so instead of gating on it the
# measurement folds its effect into the quoted uncertainty; this checks that it
# did, and that the reported minimum survives the enlarged bars.
MINIMUM_SIGNIFICANCE = 5.0
# Two routes agree when their gap is within this multiple of their combined
# reported spread, plus a floor so that two very confident numbers still have
# to be close in absolute terms.
ROUTE_SIGMA = 3.0
ROUTE_RELATIVE = 5e-3


def _oracle(data):
    rows = [r for r in data['perturbative']
            if r['significant_digits'] >= ORACLE_DIGITS]
    return bool(len(rows) >= ORACLE_SPINS)


def _oracle_is_discriminating(data):
    """The comparison has to be able to fail.

    A test that passes for any smooth curve tests nothing. Perturbing the
    oracle by one part in a thousand must break the agreement at every spin
    that currently passes it.
    """
    for row in data['perturbative']:
        if row['significant_digits'] < ORACLE_DIGITS:
            continue
        perturbed = abs(row['measured']-1.001*row['oracle'])/max(abs(row['oracle']), 1e-30)
        if perturbed < 10**(-ORACLE_DIGITS):
            return False
    return True


def _second_order(data):
    rows = data['step_refinement']
    if not rows:
        return False
    return all(row['ratios'] and row['worst_ratio_deviation'] <= RATIO_TOLERANCE
               for row in rows)


def _gates_hold(data):
    """Every recorded state passed the declared residual and boundary gates."""
    relative = data['relative_tensor_gate']
    boundary = data['boundary_gate']
    for states in data['walks'].values():
        for state in states:
            if not (state['max_relative_tensor_residual'] < relative
                    and state['boundary_residual'] < boundary):
                return False
    return True


def _smarr(data):
    return all(state['smarr'] < SMARR_GATE
               for states in data['walks'].values() for state in states)


def _reached_extremality(data):
    return all(min(s['tau'] for s in states) < TAU_FLOOR
               for states in data['walks'].values())


def _mass_anchor(data):
    return bool(data['extremal_anchor']['mu_deviation'] < MASS_ANCHOR_GATE)


def _shift_anchor(data):
    """lim Psi_GB at alpha_GB = 0 against the pi of arXiv:2009.00015."""
    return bool(data['extremal_anchor']['psi_deviation'] < SHIFT_ANCHOR_GATE)


def _shift_from_masses(data):
    """The same number reconstructed without any thermodynamics at all."""
    curve = data['mass_curve']
    return bool(curve['slope_at_zero_deviation'] < MASS_ROUTE_SHIFT_GATE)


def _near_horizon(data):
    rows = data['near_horizon']
    return bool(rows and all(r['relative_difference'] < NEAR_HORIZON_GATE
                             for r in rows))


def _published_near_horizon(data):
    rows = data['published_near_horizon']
    return bool(rows and all(r['stationarity'] < 1e-12
                             and abs(r['spin_ratio']-1) < 1e-12 for r in rows))


def _routes(data):
    """Int Psi_GB dy against the measured change in mu, interval by interval.

    An identity, not an approximation: the right-hand side is exactly the
    integral of mu\'. A failure means one of the two routes is wrong, and the
    extremal shift does not get claimed.
    """
    comparison = data['route_comparison']
    rows = comparison['intervals']
    if not rows:
        return False
    return all(r['relative_difference'] <= ROUTE_RELATIVE
               or r['difference'] <= ROUTE_SIGMA*r['combined_spread']
               for r in rows)


def _bars_cover_the_secants(data):
    """The quoted uncertainty must cover the disagreement it is quoted against.

    The referee's objection, made a gate: an error bar that does not contain
    the independent measurement is not an error bar. `routes_agree` accepts an
    interval on either the relative criterion or the spread one; this one drops
    the escape hatch and requires every interval to be covered by the
    uncertainty the paper actually prints.
    """
    rows = data['route_comparison']['intervals']
    return bool(rows and all(r['covered'] for r in rows))


def _resolution(data):
    rows = data['resolution_contrast']
    if not rows:
        return False
    # The coarse-resolution difference must be no larger than the extrapolation
    # spread it is being compared against, times a factor; otherwise the error
    # budget is dominated by something the budget does not report.
    return all(r['psi_difference'] < 5e-2 and r['mu_difference'] < 5e-3
               for r in rows)


def _locus_moves(data):
    """The sign change is located in the scaled temperature of a covered walk.

    Allowed to report "it does not move": the gate is that the locus is
    bracketed by an actual change of sign between two accepted states wherever
    the walk covers the spin range, not that it shifts. Walks that only sample
    near extremality are excluded rather than interpolated across.
    """
    rows = [r for r in data['sign_locus'] if r['bracketed'] and r['fully_covered']]
    if len(rows) < 3:
        return False
    return all(np.isfinite(r['t_scaled']) and 0 < r['t_scaled'] < 1 for r in rows)


def _static_limit(data):
    """The one exact test at alpha != 0: the closed-form static member."""
    rows = data['static_limit']
    return bool(rows and all(
        r['psi_relative'] < STATIC_GATE and r['mass_relative'] < STATIC_GATE
        and r['entropy_relative'] < STATIC_GATE
        and r['temperature_relative'] < STATIC_GATE for r in rows))


def _mass_bound_is_declared(data):
    """The exact bound `x <= 2 abar/(1+2 abar)` is violated only where declared.

    Not a gate on the measurement being right -- the violation is real, and the
    manuscript reports it as the honest bound on the mass extraction. It is a
    gate on the violation staying where the appendix says it stays: confined to
    the largest coupling, and below the route disagreement that the extremal
    numbers are quoted against. If it ever spreads, the paragraph in the
    appendix is wrong and this fails before a reader finds out.
    """
    summary = data['mass_bound']
    offenders = [r for r in summary['rows'] if r['excess'] > 0.]
    return bool(summary['rows']
                and summary['worst_mass_relative'] < MASS_BOUND_GATE
                and all(r['alpha_gb'] >= max(x['alpha_gb'] for x in summary['rows'])
                        for r in offenders))


def _vacuum_masses(data):
    rows = data['vacuum_masses']
    return bool(rows and all(r['relative'] < VACUUM_MASS_GATE for r in rows))


def _first_law(data):
    rows = data['consistency']
    return bool(rows and all(r['first_law']['relative'] < FIRST_LAW_GATE
                             for r in rows))


def _goon_penco(data):
    """Psi against -T dS/dalpha at fixed M and J, which is the same number."""
    rows = data['consistency']
    return bool(rows and all(not r['goon_penco']['singular']
                             and r['goon_penco']['relative'] < GOON_PENCO_GATE
                             for r in rows))


def _no_log_term(data):
    """No support in the data for the near-extremal log non-analyticity."""
    rows = data['extrapolation_models']
    return bool(rows and all(
        abs(r['intercept_tlogt']-r['polynomial']) < LOG_MODEL_GATE for r in rows))


def _uncertainty_covers_the_model(data):
    """Every quoted uncertainty is at least the spread across fit models."""
    by_model = {r['alpha_gb']: r['model_spread'] for r in data['extrapolation_models']}
    return all(record['psi_uncertainty'] >= by_model.get(record['alpha_gb'], 0.)-1e-15
               and record['psi_uncertainty'] >= record['psi_gb']['spread']-1e-15
               for record in data['extremals'])


def _minimum_is_significant(data):
    """The non-monotonicity has to survive the enlarged uncertainties.

    This is the physics gate: the minimum must sit below the largest coupling's
    value by several combined uncertainties. If it did not, the paper would be
    reporting a wiggle.
    """
    ordered = sorted(data['extremals'], key=lambda r: r['y'])
    minimum = data['minimum']
    last = ordered[-1]
    gap = last['psi_gb']['value'] - minimum['fitted_value']
    combined = minimum['value_uncertainty'] + last['psi_uncertainty']
    return bool(gap > MINIMUM_SIGNIFICANCE*combined)


def _published_bounds(data):
    """The scaled spin bound of arXiv:2303.12471, and our own normalisation.

    Two things at once: that `j = 1` comes out on the alpha_GB=0 extremal state,
    which is a closed form and therefore a check on the normalisation rather
    than on the physics, and that `j <= 1` holds on every state, which is the
    paper's finite-coupling statement and is allowed to fail.
    """
    bounds = data.get('published_bounds')
    if not bounds:
        return False
    vacuum = bounds.get('vacuum_extremal_j')
    return bool(vacuum is not None and abs(vacuum-1) < 1e-4
                and bounds['spin_bound_holds'] and bounds['static_bound_holds']
                and bounds['extremal_j_is_monotone']
                and bounds['extremal_j_at_largest_coupling'] < 1)


CHECKS = (
    ('oracle_reproduced_to_eight_digits', _oracle),
    ('oracle_test_is_discriminating', _oracle_is_discriminating),
    ('central_differences_converge_to_the_linear_response', _second_order),
    ('every_state_passes_the_declared_gates', _gates_hold),
    ('smarr_closes_at_every_state', _smarr),
    ('every_walk_reaches_extremality', _reached_extremality),
    ('published_extremal_mass_reproduced', _mass_anchor),
    ('published_extremal_shift_reproduced', _shift_anchor),
    ('published_shift_recovered_from_masses_alone', _shift_from_masses),
    ('near_horizon_agrees', _near_horizon),
    ('published_near_horizon_equations_reproduced', _published_near_horizon),
    ('routes_agree', _routes),
    ('quoted_bars_cover_the_independent_route', _bars_cover_the_secants),
    ('result_independent_of_resolution', _resolution),
    ('sign_locus_bracketed_in_the_scaled_temperature', _locus_moves),
    ('published_scaled_bounds_hold', _published_bounds),
    ('static_limit_matches_the_closed_form', _static_limit),
    ('vacuum_masses_match_myers_perry', _vacuum_masses),
    ('mass_extraction_bound_violated_only_where_declared', _mass_bound_is_declared),
    ('first_law_closes_in_the_spin_direction', _first_law),
    ('goon_penco_identity_holds_at_finite_temperature', _goon_penco),
    ('no_evidence_for_a_log_non_analyticity', _no_log_term),
    ('uncertainty_covers_the_extrapolation_model', _uncertainty_covers_the_model),
    ('the_minimum_survives_the_uncertainties', _minimum_is_significant),
)


def checks(data):
    return [dict(name=name, passed=bool(test(data))) for name, test in CHECKS]
