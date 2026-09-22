import numpy as np


def test_numerical_metric_satisfies_independent_tensor_and_constraint():
    from rotating_bh.vacuum_bvp import solve
    from rotating_bh.vacuum_validation import diagnose
    result=solve(.2,method='spectral',resolution=32,tol=1e-11)
    checks=diagnose(result)
    assert checks['profile_error']<1e-6
    assert checks['einstein_residual']<1e-6
    assert checks['constraint_residual']<1e-6
    assert checks['ode_residual']<1e-6
    assert checks['positive_exterior']


def test_generic_metric_jets_detect_a_perturbation():
    from rotating_bh.vacuum_bvp import solve
    from rotating_bh.vacuum_validation import physical_jets, tensor_residual
    result=solve(.2,method='spectral',resolution=32,tol=1e-11)
    jets=physical_jets(result,1.3)
    baseline=tensor_residual(1.3,.7,jets)
    jets[:,3]*=1.01
    assert baseline<1e-6
    assert tensor_residual(1.3,.7,jets)>1e-5


def test_restoring_radius_preserves_dimensionless_tensor_norm():
    from rotating_bh.vacuum_bvp import solve
    from rotating_bh.vacuum_validation import physical_jets, tensor_residual
    result=solve(.2,method='spectral',resolution=32,tol=1e-11)
    for scale in [.7,2.]:
        jets=physical_jets(result,1.3*scale,r_h=scale)
        assert tensor_residual(1.3*scale,.7,jets,r_h=scale)<1e-6


def test_missing_tensor_sample_coverage_rejects_candidate():
    from rotating_bh.vacuum_bvp import solve
    from rotating_bh.vacuum_validation import diagnose
    result=solve(0,cutoff=.049,tol=1e-6)
    checks=diagnose(result)
    assert not checks['tensor_sample_coverage']
    assert not checks['accepted']


def test_zero_horizon_slope_is_rejected_even_when_exterior_samples_pass():
    from rotating_bh.vacuum_bvp import solve
    from rotating_bh.vacuum_validation import diagnose
    result=solve(0,method='spectral',resolution=20,tol=1e-10)
    original=result._evaluate
    amplitude=result.evaluate(0)[0]
    def altered(x,k):
        values=original(x,k).copy()
        values[0]-=amplitude*(-10000.)**k*np.exp(-10000*x)
        return values
    result._evaluate=altered
    result.boundary_residual=0.
    checks=diagnose(result)
    assert not checks['simple_horizon']
    assert not checks['accepted']
