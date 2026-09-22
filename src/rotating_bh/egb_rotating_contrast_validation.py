"""Recompute the external-contrast gates from the recorded numbers.

Same discipline as `egb_rotating_family_validation`: a gate is derived from the
evidence in the artifact, never read back from a boolean the producing run
stored. Aggregates such as `max_absolute_difference` are recomputed from their
rows too, so a wrong aggregate cannot carry a gate on its own.
"""
import numpy as np


def _relation(data):
    """The identity must hold algebraically at machine precision and, separately,
    on the spectral solver within the project's tail-fit extraction gate. The two
    tolerances are not interchangeable: the closed-form leg may not hide behind
    the looser one."""
    rows = data['rows']
    closed = max(r['closed_form_difference'] for r in rows)
    spectral = [r['spectral_difference'] for r in rows if 'spectral_difference' in r]
    # Both tolerances are pinned here, not read from the artifact. Reading them
    # back would let a later run widen its own gate and still be called verified,
    # which is the failure this recomputation exists to prevent.
    return bool(rows and spectral
                and data['algebraic_tolerance'] <= 1e-12
                and data['extraction_tolerance'] <= 1e-6
                and closed < data['algebraic_tolerance']
                and max(spectral) < data['extraction_tolerance']
                and closed == data['max_closed_form_difference']
                and max(spectral) == data['max_spectral_difference'])


def _summed_reading_fails(data):
    """Reading the paper's J as the sum of both spins has to fail visibly.

    It can fail either by missing the relation or by pushing j^2 past 1, where
    the formula has no real value at all. Both count, and at least one of the
    sampled readings must fail, or nothing distinguishes the two conventions.
    """
    missed = data['summed_spin_reading_max_difference']
    outside = data.get('summed_spin_readings_outside_domain', 0)
    tested = data.get('summed_spin_readings_tested', 0)
    return bool(tested and (outside > 0 or (missed is not None and missed > 1e-3)))


def _critical(data):
    """The published constant and its half-unit band are pinned, so neither the
    target nor the tolerance can be edited into agreement."""
    return bool(data['published'] == 0.087396
                and data['rounding_half_unit'] == 5e-7
                and abs(data['computed']-data['published']) <= 5e-7
                and data['closed_form_vs_symbolic'] < 1e-12
                and abs(abs(data['computed']-data['published'])
                        -data['absolute_difference']) < 1e-18)


def _derivative(data):
    rows = data['rows']
    worst = max(r['absolute_difference'] for r in rows)
    # Each row is re-derived from the paper's closed form rather than trusted,
    # and the gate is pinned at 1e-10 instead of read from the artifact.
    consistent = all(
        abs(abs(r['computed']-r['paper_formula'])-r['absolute_difference']) < 1e-18
        and abs(r['paper_formula']-2/(1+r['alpha_paper'])) < 1e-15
        for r in rows)
    return bool(rows and consistent
                and data['tolerance'] <= 1e-10
                and worst < 1e-10
                and worst == data['max_absolute_difference']
                and any(r['alpha_paper'] == 2 for r in rows)
                and all(r['tensor'] < 1e-6 for r in rows))


def _monotonic(data):
    families = data['families']
    return bool(families and all(
        np.all(np.diff(f['b_prime']) < 0) and np.all(np.diff(f['f_prime']) < 0)
        for f in families))


# The published radii and the half-unit tolerances adopted for them, pinned so
# the failing comparison cannot be retired by editing either side of it.
PUBLISHED_RADII = {0: (1.059, 5e-4), 1: (1.08, 5e-3), 2: (1.104, 5e-4)}


def _radii(data):
    published = {p['alpha_paper']: p for p in data['published']}
    if set(published) != set(PUBLISHED_RADII):
        return False
    for alpha_paper, (radius, half_unit) in PUBLISHED_RADII.items():
        row = published[alpha_paper]
        if row['published_radius'] != radius or row['rounding_half_unit'] != half_unit:
            return False
        if abs(row['computed_radius']-radius) > half_unit:
            return False
    return True


def checks(data):
    """Return the gate list in the same shape the artifact stores."""
    return [
        dict(name='reduced_normalisation_pinned', sector='convention',
             passed=bool(data['normalisation']['a_H'] == '1'
                         and data['normalisation']['t_H'] == '1')),
        dict(name='einstein_reduced_relation', sector='einstein-limit',
             passed=_relation(data['einstein_reduced_relation'])),
        dict(name='einstein_each_spin_reading', sector='convention',
             passed=_summed_reading_fails(data['einstein_reduced_relation'])),
        dict(name='einstein_critical_temperature', sector='einstein-limit',
             passed=_critical(data['einstein_critical_temperature'])),
        dict(name='wald_entropy_matches_paper', sector='gauss-bonnet',
             passed=bool(data['wald_entropy']['difference'] == '0')),
        dict(name='gb_horizon_derivative', sector='gauss-bonnet',
             passed=_derivative(data['gb_horizon_derivative'])),
        dict(name='gb_horizon_derivatives_monotonic', sector='gauss-bonnet',
             passed=_monotonic(data['gb_horizon_monotonicity'])),
        dict(name='published_ergosurface_radii', sector='gauss-bonnet',
             passed=_radii(data['ergosurface_audit'])),
    ]
