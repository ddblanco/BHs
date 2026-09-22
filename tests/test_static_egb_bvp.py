import numpy as np
import pytest

from rotating_bh.static_egb import StaticEGB
from rotating_bh.static_egb_bvp import SolverFailure, solve


def _oracle_error(solution, alpha_hat):
    x = np.linspace(0.01, 1-solution.cutoff-0.01, 25)
    r = 1/(1-x)
    expected = StaticEGB(r_h=1.0, alpha_gb=alpha_hat).functions(r)['f']
    return float(np.max(np.abs(solution.evaluate(x)-expected)))


@pytest.mark.parametrize('method,kwargs', [
    ('spectral', dict(resolution=24, tol=1e-11)),
    ('adaptive', dict(tol=1e-10)),
])
@pytest.mark.parametrize('alpha_hat', [0.0, 0.02, 0.2, 1.5])
def test_matches_the_closed_form_oracle_off_nodes(method, kwargs, alpha_hat):
    solution = solve(alpha_hat, method=method, **kwargs)
    assert _oracle_error(solution, alpha_hat) < 1e-6


def test_methods_agree_with_each_other():
    spectral = solve(0.1, method='spectral', resolution=24, tol=1e-11)
    adaptive = solve(0.1, method='adaptive', tol=1e-10)
    x = np.linspace(0.01, 0.98, 30)
    assert np.max(np.abs(spectral.evaluate(x)-adaptive.evaluate(x))) < 1e-6


def test_horizon_condition_holds_and_infinity_is_a_diagnostic_not_imposed():
    solution = solve(0.15, method='spectral', resolution=20, tol=1e-11)
    assert solution.evaluate(0.0) == pytest.approx(0.0, abs=1e-10)
    assert solution.evaluate(1-solution.cutoff) == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize('alpha_hat', [0.0, 0.1, 0.5])
def test_adaptive_horizon_slope_matches_the_oracle_not_half_of_it(alpha_hat):
    # A symmetric finite difference that silently clips x-step back onto x=0
    # returns half the true one-sided slope there (AstraCheck, 2026-09-09);
    # this must match the closed-form f'(r_H)*r_H (dF/dx=df/dr*dr/dx, and
    # dr/dx=r_H at the horizon), not its half.
    expected = StaticEGB(r_h=1.0, alpha_gb=alpha_hat).horizon_data()['f1']
    solution = solve(alpha_hat, method='adaptive', tol=1e-13)
    assert solution.evaluate(0.0, 1) == pytest.approx(expected, rel=1e-4)


def test_adaptive_and_spectral_horizon_slopes_agree():
    spectral = solve(0.2, method='spectral', resolution=32, tol=1e-11)
    adaptive = solve(0.2, method='adaptive', tol=1e-13)
    assert adaptive.evaluate(0.0, 1) == pytest.approx(spectral.evaluate(0.0, 1), rel=1e-4)


def test_adaptive_derivative_near_cutoff_uses_a_one_sided_stencil():
    # Same boundary-clipping failure mode as the horizon, mirrored at
    # x=1-cutoff. dF/dx itself grows like 2(1-x) approaching infinity, so the
    # comparison point must sit much closer than the cutoff scale (1e-3) or
    # that genuine drift, not a stencil bug, would dominate the difference.
    solution = solve(0.1, method='adaptive', tol=1e-13)
    hi = 1-solution.cutoff
    nearby = solution.evaluate(hi-1e-7, 1)
    boundary = solution.evaluate(hi, 1)
    assert boundary == pytest.approx(nearby, rel=1e-3)


def test_spectral_converges_with_resolution():
    errors = [_oracle_error(solve(0.3, method='spectral', resolution=n, tol=1e-13), 0.3)
              for n in (10, 16, 24)]
    assert errors[1] < errors[0]/10
    assert errors[2] < errors[1]/10


def test_continuation_style_warm_start_from_previous_solution():
    seed = solve(0.0, method='spectral', resolution=20, tol=1e-11)
    warm = solve(0.05, method='spectral', resolution=20, tol=1e-11, previous=seed)
    assert _oracle_error(warm, 0.05) < 1e-6


@pytest.mark.parametrize('bad', [-0.1, float('nan'), float('inf')])
def test_rejects_invalid_alpha_hat(bad):
    with pytest.raises(ValueError):
        solve(bad)


def test_rejects_invalid_resource_parameters():
    with pytest.raises(ValueError):
        solve(0.1, resolution=4)
    with pytest.raises(ValueError):
        solve(0.1, tol=-1.0)
    with pytest.raises(ValueError):
        solve(0.1, cutoff=0.2)
    with pytest.raises(ValueError):
        solve(0.1, method='bogus')


def test_reports_real_resource_exhaustion():
    with pytest.raises(SolverFailure):
        solve(0.3, method='spectral', resolution=10, tol=1e-14, max_iterations=0)
