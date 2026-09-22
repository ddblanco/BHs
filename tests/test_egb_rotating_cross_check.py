"""Acceptance cannot be inferred from a converged collocation flag alone."""
import pytest


@pytest.mark.parametrize('change', [
    {'boundary_residual': 1e-4}, {'boundary_residual': float('nan')},
    {'collocation_residual': 1e-3}, {'collocation_residual': float('inf')},
    {'converged': False},
])
def test_rejects_incomplete_numerical_solution(change):
    from experiments.egb_rotating_cross_check import numerical_completion
    row = dict(converged=True, boundary_residual=1e-10,
               collocation_residual=1e-8, tolerance=1e-7)
    assert numerical_completion(row)
    row.update(change)
    assert not numerical_completion(row)
