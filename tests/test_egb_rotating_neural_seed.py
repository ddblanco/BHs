"""Check the Hito 5 artifact without trusting anything it asserts.

Gates are recomputed from the recorded numbers, the manifest has to agree, and
the sources have to still hash to what the run saw. The corruption tests make
the recomputation prove it can fail, including on the two ways this particular
result could be dressed up: reporting only the architecture and random seed that
worked, and counting a solver that merely stopped as a solver that succeeded.
"""
import hashlib
import json
from pathlib import Path

import pytest

from rotating_bh.neural_seed_validation import checks as neural_checks
from rotating_bh.provenance import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/egb-rotating-neural-seed.json'
IDENTIFIER = 'egb-rotating-neural-seed'

pytestmark = pytest.mark.skipif(not DATA.is_file(),
                                reason='Hito 5 artifact not generated yet')


@pytest.fixture(scope='module')
def artifact():
    return json.loads(DATA.read_text(encoding='utf-8'))


def test_recomputed_gates_match_the_stored_ones(artifact):
    assert neural_checks(artifact) == artifact['checks']


def test_the_benchmark_is_actually_hard(artifact):
    """If the conventional seeds succeeded at the benchmark, nothing the network
    does there would mean anything."""
    hard = [c for c in artifact['controls'] if c['alpha_gb'] >= .5]
    assert hard
    for control in hard:
        assert not control['trivial_seed']['accepted']
        assert not control['untrained_network']['accepted']
        # and the conventional route must still work, or the baseline is broken
        assert control['continuation']['accepted']
        assert control['continuation']['solves'] > 1


def test_the_untrained_control_is_myers_perry_not_random_weights(artifact):
    """A random-weight control would be a straw man; the untrained network is
    the exact alpha_gb=0 solution, which is a demanding baseline."""
    assert artifact['controls']
    assert all(c['untrained_is_myers_perry'] for c in artifact['controls'])


def test_every_architecture_and_seed_is_reported(artifact):
    """The spread is the result about dependence on human decisions. A missing
    cell would let a best case pass as the outcome."""
    expected = {(tuple(a), s, alpha)
                for a in artifact['architectures'] for s in artifact['seeds']
                for alpha in {c['alpha_gb'] for c in artifact['cells']}}
    present = {(tuple(c['layers']), c['seed'], c['alpha_gb']) for c in artifact['cells']}
    assert present == {(tuple(a), s, al) for a, s, al in expected}


def test_acceptance_is_the_independent_gate_not_the_solvers_own_tolerance(artifact):
    """The solver can terminate on its internal residual at a point the full
    tensor check rejects; that happened during this work at 8.35e+02. Anything
    marked accepted must clear the independent gates."""
    assert artifact['tensor_gate'] <= 1e-6
    assert artifact['boundary_gate'] <= 1e-8
    for cell in artifact['cells']:
        if cell['refinement']['accepted']:
            assert cell['refinement']['max_tensor_residual'] < 1e-6
            assert cell['refinement']['boundary_residual'] < 1e-8


def test_spurious_convergences_are_recorded_rather_than_dropped(artifact):
    """Seeds that led the solver to a non-solution belong in the record."""
    assert 'spurious_convergences' in artifact
    for entry in artifact['spurious_convergences']:
        assert entry['max_tensor_residual'] > 1e-6


def test_boundary_conditions_are_exactly_zero_not_merely_small(artifact):
    for row in artifact['boundary_check']:
        assert row['B_infinity_error'] == 0.
        assert row['F_infinity_error'] == 0.
        assert row['dH_dx_infinity'] == 0.
        assert row['dW_dx_infinity'] == 0.
        assert row['W_horizon_error'] == 0.


def test_gradient_was_verified_against_the_oracle(artifact):
    assert artifact['gradient_check']
    for row in artifact['gradient_check']:
        assert row['worst_relative_deviation'] < 1e-10


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


def test_recorded_sources_still_hash_to_what_the_run_saw(artifact):
    for name, digest in artifact['source_sha256'].items():
        path = ROOT/name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    # The module defining the gates must be part of the provenance, or the
    # recorded verdicts would not be reproducible from the recorded inputs.
    assert 'src/rotating_bh/neural_seed_validation.py' in artifact['source_sha256']
    assert 'src/rotating_bh/neural_seed.py' in artifact['source_sha256']


@pytest.mark.parametrize('corruption', [
    'gradient', 'boundary', 'pinned_infinity', 'easy_benchmark', 'broken_baseline',
    'hidden_spread', 'weak_acceptance'])
def test_recomputation_detects_corruption(artifact, corruption):
    data = json.loads(json.dumps(artifact))
    if corruption == 'gradient':
        data['gradient_check'][0]['worst_relative_deviation'] = 1e-3
    elif corruption == 'boundary':
        data['boundary_check'][0]['dW_dx_infinity'] = 1e-14
    elif corruption == 'pinned_infinity':
        for row in data['boundary_check']:
            row['H_infinity'], row['W_infinity'] = 0.12, 0.37
    elif corruption == 'easy_benchmark':
        for control in data['controls']:
            control['trivial_seed']['accepted'] = True
    elif corruption == 'broken_baseline':
        data['controls'][0]['continuation']['accepted'] = False
    elif corruption == 'hidden_spread':
        keep = data['cells'][0]
        data['cells'] = [c for c in data['cells']
                         if not (c['seed'] != keep['seed']
                                 and c['alpha_gb'] == keep['alpha_gb'])]
    else:
        data['tensor_gate'] = 1.0
    assert neural_checks(data) != artifact['checks']


def test_a_failing_result_cannot_be_rescued_by_relabelling(artifact):
    """Marking a cell accepted without it clearing the independent gates must
    not turn the headline gate green."""
    data = json.loads(json.dumps(artifact))
    for cell in data['cells']:
        cell['refinement']['accepted'] = True
        cell['refinement'].setdefault('max_tensor_residual', 1.0)
        cell['refinement']['max_tensor_residual'] = 1.0
        cell['refinement'].setdefault('boundary_residual', 1.0)
        cell['refinement']['boundary_residual'] = 1.0
    recomputed = {c['name']: c['passed'] for c in neural_checks(data)}
    assert not recomputed['acceptance_uses_independent_gates']
    assert not recomputed['neural_seed_reaches_an_unreachable_coupling']
