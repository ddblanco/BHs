import numpy as np
import pytest

from rotating_bh.static_egb import StaticEGB
from rotating_bh.static_egb_bvp import solve
from rotating_bh.static_egb_validation import diagnose, physical_jets, tensor_residual


@pytest.mark.parametrize('method,kwargs', [
    ('spectral', dict(resolution=32, tol=1e-11)),
    ('adaptive', dict(tol=1e-13)),
])
@pytest.mark.parametrize('alpha_gb', [0.0, 0.1, 0.5])
def test_accepted_solutions_pass_every_independent_check(method, kwargs, alpha_gb):
    solution = solve(alpha_gb, method=method, **kwargs)
    result = diagnose(solution, alpha_gb)
    assert result['accepted'], result


def test_physical_jets_match_finite_difference_of_the_oracle():
    oracle = StaticEGB(r_h=1.0, alpha_gb=0.2)
    r0, h = 2.3, 1e-4
    solution = solve(0.2, method='spectral', resolution=24, tol=1e-12)
    F, Fr, Frr = physical_jets(solution, np.array([r0]))[:, 0]
    fd_r = (oracle.functions(r0+h)['f']-oracle.functions(r0-h)['f'])/(2*h)
    fd_rr = (oracle.functions(r0+h)['f']-2*oracle.functions(r0)['f']
             + oracle.functions(r0-h)['f'])/h**2
    assert F == pytest.approx(oracle.functions(r0)['f'], abs=1e-8)
    assert Fr == pytest.approx(fd_r, abs=1e-4)
    assert Frr == pytest.approx(fd_rr, abs=1e-2)


def test_tensor_residual_is_small_on_the_numerical_solution():
    solution = solve(0.15, method='spectral', resolution=24, tol=1e-12)
    for r in (1.3, 2.0, 8.0):
        assert tensor_residual(solution, r, 0.15) < 1e-6


def test_tensor_residual_detects_a_perturbed_profile():
    solution = solve(0.15, method='spectral', resolution=24, tol=1e-12)
    baseline = tensor_residual(solution, 2.0, 0.15)
    perturbed = tensor_residual(solution, 2.0, 0.15+0.05)  # wrong alpha_GB used in the check
    assert perturbed > 100*max(baseline, 1e-12)


def test_diagnose_rejects_a_coarse_underresolved_solution():
    coarse = solve(0.4, method='spectral', resolution=8, tol=1e-2, max_iterations=2)
    result = diagnose(coarse, 0.4)
    assert not result['accepted']
