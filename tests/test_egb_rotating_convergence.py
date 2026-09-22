"""A converged Newton flag cannot replace the physical acceptance gates."""
import pytest


def acceptable_row():
    return dict(converged=True, iterations=3, boundary_residual=1e-10,
                max_tensor_residual=1e-10, adaptive_difference=1e-9,
                min_B=.7, min_F=.8, min_h_over_r2=1.,
                horizon_b_slope=1.4, horizon_f_slope=1.6,
                infinity_residual=1e-10)


@pytest.mark.parametrize('change', [
    {'converged': False}, {'iterations': 0}, {'max_tensor_residual': 1e-3},
    {'max_tensor_residual': float('nan')}, {'adaptive_difference': 1e-3},
    {'boundary_residual': 1e-5}, {'infinity_residual': 1e-4},
    {'min_B': -1.}, {'min_F': float('nan')}, {'min_h_over_r2': 0.},
    {'horizon_b_slope': 0.}, {'horizon_f_slope': float('inf')},
])
def test_rejects_unverified_or_unresolved_profiles(change):
    from rotating_bh.egb_rotating_convergence import accepted
    row = acceptable_row()
    assert accepted(row)
    row.update(change)
    assert not accepted(row)


def test_incomplete_sweep_cannot_close_hito_4b():
    from rotating_bh.egb_rotating_convergence import closure_checks
    row = acceptable_row() | dict(resolution=40, tolerance=1e-8)
    result = dict(runs=[row], fine_resolution_difference=0.,
                  adaptive_max_tensor_residual=1e-9, seed_verified=True)
    assert not all(check['passed'] for check in closure_checks(result))


def test_seed_flag_without_supporting_evidence_is_not_accepted():
    from rotating_bh.egb_rotating_convergence import closure_checks
    result = dict(runs=[], seed_verified=True)
    checks = {c['name']: c['passed'] for c in closure_checks(result)}
    assert not checks['independently_verified_seed']


@pytest.mark.parametrize('boundary', [None, float('nan'), 1e-4])
def test_adaptive_reference_requires_its_boundary_check(boundary):
    from rotating_bh.egb_rotating_convergence import closure_checks
    result = dict(runs=[], adaptive_max_tensor_residual=1e-9,
                  adaptive_boundary_residual=boundary, adaptive_collocation_residual=1e-8)
    checks = {c['name']: c['passed'] for c in closure_checks(result)}
    assert not checks['independent_adaptive_tensor']


def test_real_study_solves_instead_of_returning_the_seed():
    from rotating_bh.egb_rotating_convergence import run_study, accepted
    result = run_study(resolutions=(12, 24), tolerances=(1e-8,), tensor_count=11)
    assert result['seed_verified']
    assert result['myers_perry_difference'] < 1e-5
    assert result['adaptive_max_tensor_residual'] < 1e-6
    coarse, fine = result['runs']
    assert fine['iterations'] > 0
    assert accepted(fine)
    assert fine['max_tensor_residual'] < coarse['max_tensor_residual']/100
    assert not all(check['passed'] for check in result['checks'])  # deliberately incomplete sweep
