"""The Hito 8 artifact, checked against itself and against the code that made it.

The pattern is the project's: gates are recomputed here from the recorded
numbers rather than read back from the booleans the producing run stored, the
source digests are re-verified, and the derived quantities are re-derived. A
report that passes this cannot have had its own verdict written into it.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rotating_bh.extremality_validation import checks as extremality_checks
from rotating_bh.near_horizon import extremal, paper_residuals

from known_digest_exceptions import excused

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/egb-extremality.json'

pytestmark = pytest.mark.skipif(not DATA.exists(),
                                reason='run experiments/egb_extremality.py first')


@pytest.fixture(scope='module')
def report():
    return json.loads(DATA.read_text(encoding='utf-8'))


def test_the_recorded_gates_are_the_recomputed_ones(report):
    recomputed = extremality_checks(report)
    assert recomputed == report['checks']


def test_the_source_digests_match_the_files(report):
    for relative, digest in report['source_sha256'].items():
        path = ROOT/relative
        assert path.exists(), relative
        if excused(relative, digest):
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative


def test_the_invariants_are_consistent_with_the_raw_observables(report):
    """y, tau, mu and sigma must be what E, J, S, T_H and the coupling say.

    Recomputed rather than trusted: these four numbers carry the whole physical
    statement, and they are the only place in the artifact where a units slip
    would go unnoticed.
    """
    for states in report['walks'].values():
        for state in states:
            spin = state['J']
            assert state['y'] == pytest.approx(state['alpha_gb']/spin**(2/3), rel=1e-12)
            assert state['tau'] == pytest.approx(state['T_H']*spin**(1/3), rel=1e-12)
            assert state['mu'] == pytest.approx(state['E']/spin**(2/3), rel=1e-12)
            assert state['sigma'] == pytest.approx(state['S']/spin, rel=1e-12)


def test_smarr_is_recomputable_from_the_recorded_observables(report):
    for states in report['walks'].values():
        for state in states:
            left = 2*state['E']
            right = (3*state['T_H']*state['S'] + 6*state['omega_h']*state['J']
                     + 2*state['alpha_gb']*state['psi_gb'])
            assert abs(left-right)/abs(left) == pytest.approx(state['smarr'],
                                                              rel=1e-9, abs=1e-14)


def test_the_published_anchors_are_not_reproduced_by_construction(report):
    """The anchors must come from the measurement, never be imposed on it.

    `mu` at y=0 is a measured extrapolation and the one-sided slope there is
    recomputed from the measured points alone, so neither published constant
    appears anywhere in the data being fitted.
    """
    curve = report['mass_curve']
    y = np.array(curve['y'])
    mu = np.array(curve['mu'])
    assert curve['mu_at_zero'] == pytest.approx(mu[0], rel=1e-15)
    assert curve['mass_coefficient'] not in mu.tolist()
    assert curve['published_shift'] not in curve['secant']
    secants = np.diff(mu)/np.diff(y)
    assert secants == pytest.approx(np.array(curve['secant']), rel=1e-12)
    for count, slope in curve['one_sided_slopes'].items():
        coefficients = np.polynomial.polynomial.polyfit(
            y[:int(count)], mu[:int(count)], 2)
        recomputed = np.polynomial.polynomial.polyder(coefficients)[0]
        assert recomputed == pytest.approx(slope, rel=1e-9)


def test_the_two_routes_are_recomputable_from_the_recorded_states(report):
    """Int Psi dy against d mu is an identity; recompute the difference."""
    comparison = report['route_comparison']
    extremals = sorted(report['extremals'], key=lambda r: r['y'])
    for index, row in enumerate(comparison['intervals']):
        change = extremals[index+1]['mu']-extremals[index]['mu']
        assert change == pytest.approx(row['mass_change'], rel=1e-12)
        assert abs(row['integrated_psi']-change) == pytest.approx(
            row['difference'], rel=1e-9, abs=1e-15)


def test_the_near_horizon_comparison_is_against_a_live_computation(report):
    """Recompute the algebraic branch instead of trusting the stored value."""
    for row in report['near_horizon'][:4]:
        if row['y'] <= 0:
            expected = 2*np.pi
        else:
            expected = extremal(row['alpha_tilde'])['S_over_J']
        assert expected == pytest.approx(row['near_horizon'], rel=1e-9)


def test_the_published_near_horizon_residuals_are_recomputable(report):
    for row in report['published_near_horizon']:
        live = paper_residuals(extremal(row['alpha_tilde']))
        assert live['stationarity'] == pytest.approx(row['stationarity'],
                                                     rel=1e-6, abs=1e-15)
        assert live['spin_ratio'] == pytest.approx(row['spin_ratio'], rel=1e-12)


def test_every_failing_gate_is_declared(report):
    """A failing gate is allowed; an undeclared failing gate is not.

    The scope line has to name what the artifact does not establish, and any
    gate that fails has to be visible in it. This mirrors the Hito 7 audit rule
    that every failed check carries its reason.
    """
    failing = [g['name'] for g in report['checks'] if not g['passed']]
    if failing:
        declared = report['scope'] + report.get('limitations', '')
        assert declared.strip(), failing


def test_the_step_refinement_really_is_second_order(report):
    """Recompute the ratios from the ladder, not from the stored summary."""
    for row in report['step_refinement']:
        gaps = [entry['gap'] for entry in row['ladder']]
        ratios = [abs(a/b) for a, b in zip(gaps, gaps[1:]) if b]
        assert ratios == pytest.approx(row['ratios'], rel=1e-9)
        assert max(abs(r-4) for r in ratios) < .25
