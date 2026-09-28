"""Check the Gauss-Bonnet conjugate potential machinery and its artifact.

The measurement rests on three things being true: the Smarr coefficients are the
ones the exact vacuum limit dictates, Psi_GB converges as the difference step is
refined, and no solution enters the average without passing the unchanged Hito 4
gates. Each is checked against something independent of the artifact.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rotating_bh.gb_potential import (perturbative_potential, perturbative_zero,
                                      smarr_residual)
from rotating_bh.gb_potential_validation import checks as gb_checks
from rotating_bh.myers_perry import MyersPerry
from rotating_bh.provenance import validate_manifest

from known_digest_exceptions import excused

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/egb-rotating-gb-potential.json'
IDENTIFIER = 'egb-rotating-gb-potential'


@pytest.mark.parametrize('omega_h', [.1, .3, .33, .5, .6])
def test_smarr_coefficients_are_fixed_by_the_exact_vacuum_limit(omega_h):
    """2E = 3TS + 6 Omega J must hold for Myers-Perry to machine precision.

    This is what pins the relation down. If the coefficients were wrong, the
    whole finite-coupling measurement would be measuring the wrong thing, and
    this is the only place with an exact answer to check against.
    """
    thermo = MyersPerry(r_h=1., omega_h=omega_h).thermodynamics()
    observables = dict(E=thermo['M'], J=thermo['J1'], T_H=thermo['T_H'], S=thermo['S'])
    residual = smarr_residual(observables, omega_h, 0., 0.)
    assert residual['relative'] < 1e-14


def test_a_wrong_smarr_coefficient_would_be_caught():
    """Negative control: the vacuum check has to be able to fail."""
    thermo = MyersPerry(r_h=1., omega_h=.3).thermodynamics()
    observables = dict(E=thermo['M'], J=thermo['J1'], T_H=thermo['T_H'], S=thermo['S'])
    wrong = dict(observables, S=observables['S']*1.01)
    assert smarr_residual(wrong, .3, 0., 0.)['relative'] > 1e-3


def test_the_perturbative_oracle_is_only_a_hypothesis():
    """It must be usable and it must be falsifiable: finite, sign-changing, and
    with the zero the precheck quotes. Nothing here asserts it is correct."""
    zero = perturbative_zero()
    assert zero == pytest.approx(np.sqrt(9/(16+2*np.sqrt(10))), abs=1e-15)
    assert perturbative_potential(zero) == pytest.approx(0., abs=1e-12)
    assert perturbative_potential(.3) < 0 < perturbative_potential(.68)
    assert np.isfinite(perturbative_potential(np.array([.1, .3, .5, .65]))).all()


def test_potential_refuses_a_step_that_would_touch_the_vacuum():
    """alpha_gb=0 is reached only by climbing. Using it as a difference
    neighbour is the failure the 4C addendum documents, so it is refused."""
    from rotating_bh.gb_potential import potential
    with pytest.raises(ValueError):
        potential(.3, .02, .02)
    with pytest.raises(ValueError):
        potential(.3, .02, .05)
    with pytest.raises(ValueError):
        potential(.3, .02, 0.)


pytestmark_artifact = pytest.mark.skipif(not DATA.is_file(),
                                         reason='Hito 6 artifact not generated yet')


@pytest.fixture(scope='module')
def artifact():
    if not DATA.is_file():
        pytest.skip('Hito 6 artifact not generated yet')
    return json.loads(DATA.read_text(encoding='utf-8'))


@pytestmark_artifact
def test_recomputed_gates_match_the_stored_ones(artifact):
    assert gb_checks(artifact) == artifact['checks']


@pytestmark_artifact
def test_every_solution_passed_the_hito_4_gates(artifact):
    assert artifact['tensor_gate'] <= 1e-6
    assert artifact['boundary_gate'] <= 1e-8
    for point in artifact['points']:
        assert point['max_tensor_residual'] < 1e-6


@pytestmark_artifact
def test_psi_settles_as_the_step_shrinks(artifact):
    """A value that keeps moving under refinement is not a measurement."""
    assert artifact['step_refinement']
    for row in artifact['step_refinement']:
        ladder = row['ladder']
        assert len(ladder) >= 3
        steps = [entry['step'] for entry in ladder]
        assert steps == sorted(steps, reverse=True)
        changes = [abs(b['psi_gb']-a['psi_gb']) for a, b in zip(ladder, ladder[1:])]
        assert changes[-1] < changes[0] or changes[-1] < 1e-9


@pytestmark_artifact
def test_smarr_is_reported_as_a_consistency_check_not_a_confirmation(artifact):
    """Both routes share observables; the artifact must say so rather than
    presenting the agreement as independent evidence."""
    assert 'consistency' in artifact['scope'].lower() or \
           'shares' in artifact['scope'].lower() or \
           'cross-checked' in artifact['scope'].lower()
    for point in artifact['points']:
        assert point['smarr']['relative'] < 1e-4


@pytestmark_artifact
def test_a_sign_change_is_only_recorded_when_genuinely_bracketed(artifact):
    """Interpolating between same-signed points would manufacture a locus."""
    for row in artifact['sign_change']:
        assert row['bracket_low']['psi_gb']*row['bracket_high']['psi_gb'] < 0
        assert row['bracket_low']['q'] <= row['q_star'] <= row['bracket_high']['q']
    for attempt in artifact['sign_change_attempts']:
        if not attempt.get('bracketed'):
            signs = {np.sign(s['psi_gb']) for s in attempt['samples']}
            assert len(signs) == 1, 'a real sign change was present but not bracketed'


@pytestmark_artifact
def test_novelty_is_not_claimed(artifact):
    """The existence of this potential and of its sign change have antecedents;
    the artifact has to say so itself."""
    scope = artifact['scope'].lower()
    assert 'not new' in scope or 'no claim' in scope
    assert 'publish' not in scope or 'no claim of publishability' in scope


@pytestmark_artifact
def test_manifest_entry_agrees_with_the_artifact(artifact):
    validate_manifest(ROOT/'artifacts/manifest.json')
    records = json.loads((ROOT/'artifacts/manifest.json').read_text(encoding='utf-8'))
    matches = [r for r in records if r['id'] == IDENTIFIER]
    assert len(matches) == 1
    record = matches[0]
    assert record['sha256'] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert record['checks'] == artifact['checks']
    assert record['status'] == ('verified' if all(c['passed'] for c in artifact['checks'])
                                else 'candidate')


@pytestmark_artifact
def test_recorded_sources_still_hash_to_what_the_run_saw(artifact):
    for name, digest in artifact['source_sha256'].items():
        path = ROOT/name
        assert path.is_file(), name
        if excused(name, digest):
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    assert 'src/rotating_bh/gb_potential_validation.py' in artifact['source_sha256']


@pytestmark_artifact
@pytest.mark.parametrize('corruption', [
    'vacuum', 'convergence', 'smarr', 'gates', 'missing_point', 'fake_sign_change'])
def test_recomputation_detects_corruption(artifact, corruption):
    data = json.loads(json.dumps(artifact))
    if corruption == 'vacuum':
        data['vacuum_smarr'][0]['relative'] = 1e-3
    elif corruption == 'convergence':
        for entry in data['step_refinement'][0]['ladder']:
            entry['psi_gb'] += entry['step']*1e3
    elif corruption == 'smarr':
        data['points'][0]['smarr']['relative'] = 1.
    elif corruption == 'gates':
        data['tensor_gate'] = 1.
    elif corruption == 'missing_point':
        data['points'] = data['points'][1:]
    else:
        if data['sign_change']:
            data['sign_change'][0]['bracket_high']['psi_gb'] = \
                data['sign_change'][0]['bracket_low']['psi_gb']
        else:
            data['sign_change'] = [dict(q_star=.5,
                                        bracket_low=dict(q=.4, psi_gb=-1.),
                                        bracket_high=dict(q=.6, psi_gb=-2.))]
    assert gb_checks(data) != artifact['checks']
