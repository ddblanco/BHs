"""Check the external-contrast artifact without trusting anything it asserts.

Every gate is recomputed from the recorded numbers, the manifest entry has to
agree with the artifact, and each recorded source has to still hash to what the
run saw. The corruption tests make the recomputation prove it can fail: a gate
that cannot be broken is not checking anything.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rotating_bh.egb_rotating_contrast_validation import checks as contrast_checks
from rotating_bh.provenance import validate_manifest

from known_digest_exceptions import excused

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/egb-rotating-external-contrasts.json'
IDENTIFIER = 'egb-rotating-external-contrasts'


@pytest.fixture(scope='module')
def artifact():
    return json.loads(DATA.read_text(encoding='utf-8'))


def test_recomputed_gates_match_the_stored_ones(artifact):
    assert contrast_checks(artifact) == artifact['checks']


def test_einstein_relation_holds_on_both_the_closed_form_and_the_solver(artifact):
    """The paper's omega_H^2(j^2) must hold for the Myers-Perry closed forms and
    for the spectral solver run at alpha_gb=0, or the agreement would only be
    testing the algebra against itself."""
    relation = artifact['einstein_reduced_relation']
    spectral = [r for r in relation['rows'] if 'spectral_difference' in r]
    assert len(spectral) >= 5
    for row in relation['rows']:
        predicted = 2*(1-np.sqrt(1-row['j2']))/row['j2']-1
        assert abs(row['omega_H2']-predicted) < relation['algebraic_tolerance']
        assert row['paper_formula'] == pytest.approx(predicted, abs=1e-15)
    for row in spectral:
        assert row['spectral_tensor'] < 1e-6
        assert row['spectral_difference'] < relation['extraction_tolerance']


def test_the_two_relation_tolerances_stay_separate(artifact):
    """The algebraic leg must not be allowed to hide behind the looser gate the
    spectral extraction needs, and neither gate may drift upwards."""
    relation = artifact['einstein_reduced_relation']
    assert relation['algebraic_tolerance'] == 1e-12
    assert relation['extraction_tolerance'] == 1e-6
    assert relation['max_closed_form_difference'] < relation['algebraic_tolerance']
    # The spectral leg genuinely needs the looser gate; if it ever became as
    # sharp as the algebraic one, the split should be removed rather than kept.
    assert relation['max_spectral_difference'] < relation['extraction_tolerance']
    assert relation['tolerance_note'].strip()


def test_each_spin_reading_is_what_makes_the_relation_hold(artifact):
    """Reading the paper's J as the sum of both spins must visibly fail, which is
    what fixes the convention rather than leaving it assumed."""
    relation = artifact['einstein_reduced_relation']
    missed = relation['summed_spin_reading_max_difference']
    outside = relation['summed_spin_readings_outside_domain']
    assert relation['summed_spin_readings_tested'] > 0
    assert outside > 0 or (missed is not None and missed > 1e-3)
    assert relation['max_closed_form_difference'] < 1e-12


def test_critical_temperature_agrees_to_every_published_digit(artifact):
    critical = artifact['einstein_critical_temperature']
    assert critical['published'] == 0.087396
    assert critical['rounding_half_unit'] == 5e-7
    assert abs(critical['computed']-critical['published']) <= critical['rounding_half_unit']
    assert critical['closed_form_vs_symbolic'] < 1e-12
    assert 0.2 < critical['q_star'] < 0.45


def test_gauss_bonnet_contrasts_are_not_einstein_limit_results(artifact):
    """The load-bearing claim is that the GB sector itself was contrasted, so the
    GB gates must exist, pass, and be labelled as such."""
    gb = [c for c in artifact['checks'] if c['sector'] == 'gauss-bonnet']
    assert {c['name'] for c in gb} >= {'wald_entropy_matches_paper',
                                       'gb_horizon_derivative',
                                       'gb_horizon_derivatives_monotonic'}
    assert artifact['wald_entropy']['difference'] == '0'
    assert artifact['wald_entropy']['sector'] == 'gauss-bonnet'


def test_horizon_derivative_is_checked_at_the_discrepant_coupling(artifact):
    """alpha_paper=2 is where the published radius fails. The closed-form
    horizon derivative has to be evaluated there, or the contrast would leave
    exactly the interesting coupling untested."""
    derivative = artifact['gb_horizon_derivative']
    at_two = [r for r in derivative['rows'] if r['alpha_paper'] == 2]
    assert len(at_two) == 1
    row = at_two[0]
    assert row['paper_formula'] == pytest.approx(2/(1+row['alpha_paper']), abs=1e-15)
    assert row['absolute_difference'] < 1e-10


def test_the_failed_radius_is_restated_and_not_re_toleranced(artifact):
    """The rejection must survive this artifact intact: same published values,
    same half-unit tolerances, still failing, and no erratum claimed."""
    audit = artifact['ergosurface_audit']
    published = {p['alpha_paper']: p for p in audit['published']}
    assert published[2]['published_radius'] == 1.104
    assert published[2]['rounding_half_unit'] == 0.0005
    assert not published[2]['agrees']
    assert not audit['agrees']
    assert any(not c['passed'] for c in artifact['checks'])
    assert 'erratum' not in json.dumps(artifact).lower()


def test_manifest_entry_agrees_with_the_artifact(artifact):
    validate_manifest(ROOT/'artifacts/manifest.json')
    records = json.loads((ROOT/'artifacts/manifest.json').read_text(encoding='utf-8'))
    matches = [r for r in records if r['id'] == IDENTIFIER]
    assert len(matches) == 1
    record = matches[0]
    import hashlib
    assert record['sha256'] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert record['checks'] == artifact['checks']
    # A failing gate must keep the record out of `verified`.
    assert record['status'] == ('verified' if all(c['passed'] for c in artifact['checks'])
                                else 'candidate')


def test_recorded_sources_still_hash_to_what_the_run_saw(artifact):
    import hashlib
    for name, digest in artifact['source_sha256'].items():
        path = ROOT/name
        assert path.is_file(), name
        if excused(name, digest):
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize('corruption', [
    'relation', 'critical', 'entropy', 'derivative', 'monotonic', 'radii',
    'summed_reading', 'relation_tolerance'])
def test_recomputation_detects_corruption(artifact, corruption):
    data = json.loads(json.dumps(artifact))
    if corruption == 'relation':
        data['einstein_reduced_relation']['rows'][0]['closed_form_difference'] = 1.0
    elif corruption == 'relation_tolerance':
        # Widening the algebraic gate to the extraction gate must not pass.
        data['einstein_reduced_relation']['algebraic_tolerance'] = 1e-6
        data['einstein_reduced_relation']['max_closed_form_difference'] = 1e-9
        data['einstein_reduced_relation']['rows'][0]['closed_form_difference'] = 1e-9
    elif corruption == 'critical':
        data['einstein_critical_temperature']['absolute_difference'] = 1.0
    elif corruption == 'entropy':
        data['wald_entropy']['difference'] = 'alpha_gb'
    elif corruption == 'derivative':
        data['gb_horizon_derivative']['rows'][-1]['absolute_difference'] = 1.0
    elif corruption == 'summed_reading':
        data['einstein_reduced_relation']['summed_spin_reading_max_difference'] = 0.0
        data['einstein_reduced_relation']['summed_spin_readings_outside_domain'] = 0
    elif corruption == 'monotonic':
        data['gb_horizon_monotonicity']['families'][0]['b_prime'] = [1., 2., 3., 4., 5., 6.]
    else:
        # The radii gate already fails, so nudging another entry proves nothing.
        # The corruption worth catching is the one that makes the failure go
        # away: moving the discrepant radius onto its published value, or
        # widening the tolerance that rejects it. Both must flip the gate, which
        # is detectable precisely because the gate is currently False.
        failing = next(p for p in data['ergosurface_audit']['published']
                       if p['alpha_paper'] == 2)
        failing['computed_radius'] = failing['published_radius']
    assert contrast_checks(data) != artifact['checks']


def test_the_rejected_radius_cannot_be_retired_by_widening_its_tolerance(artifact):
    data = json.loads(json.dumps(artifact))
    failing = next(p for p in data['ergosurface_audit']['published']
                   if p['alpha_paper'] == 2)
    failing['rounding_half_unit'] = 0.01
    recomputed = contrast_checks(data)
    gate = [c for c in recomputed if c['name'] == 'published_ergosurface_radii'][0]
    assert not gate['passed'], 'a widened tolerance must not retire the rejection'


def test_the_published_targets_cannot_be_edited(artifact):
    """Moving the goalposts -- changing what the paper is said to report -- has to
    fail the gate rather than satisfy it."""
    data = json.loads(json.dumps(artifact))
    failing = next(p for p in data['ergosurface_audit']['published']
                   if p['alpha_paper'] == 2)
    failing['published_radius'] = 1.102
    gate = [c for c in contrast_checks(data)
            if c['name'] == 'published_ergosurface_radii'][0]
    assert not gate['passed']

    critical = json.loads(json.dumps(artifact))
    critical['einstein_critical_temperature']['published'] = 0.0874
    gate = [c for c in contrast_checks(critical)
            if c['name'] == 'einstein_critical_temperature'][0]
    assert not gate['passed']


def test_aggregates_cannot_carry_a_gate_alone(artifact):
    """Lowering a summary number while its rows still disagree must not pass."""
    data = json.loads(json.dumps(artifact))
    data['gb_horizon_derivative']['rows'][0]['computed'] += 1.0
    recomputed = contrast_checks(data)
    assert not [c for c in recomputed if c['name'] == 'gb_horizon_derivative'][0]['passed']
