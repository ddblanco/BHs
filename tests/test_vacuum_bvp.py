"""Physical recovery, represented derivatives, and explicit solver failure."""
import numpy as np
import pytest


def reference(x,q):
    z=1-x; a=q*q/(1-q*q)
    return np.array([(1-a*z*z)/(1+a*z**4),1-a*z*z,
                     np.full_like(x,a),q/(1-q*q+q*q*z**4)])


def test_spectral_recovers_rotating_solution_and_derivatives():
    from rotating_bh.vacuum_bvp import solve
    static=solve(0,method='spectral',resolution=20,tol=1e-9)
    result=solve(.1,method='spectral',resolution=24,tol=1e-9,previous=static)
    x=np.linspace(.02,.98,121)
    assert np.max(abs(result.evaluate(x)-reference(x,.1))) < 1e-6
    step=1e-5
    diff=(result.evaluate(x+step)-result.evaluate(x-step))/(2*step)
    assert np.max(abs(diff-result.evaluate(x,1))) < 1e-7
    diff2=(result.evaluate(x+step,1)-result.evaluate(x-step,1))/(2*step)
    assert np.max(abs(diff2-result.evaluate(x,2))) < 1e-6


def test_adaptive_recovers_rotating_solution():
    from rotating_bh.vacuum_bvp import solve
    static=solve(0,method='adaptive',resolution=32,tol=1e-7)
    result=solve(.1,method='adaptive',resolution=32,tol=1e-7,previous=static)
    x=np.linspace(.02,.98,121)
    assert np.max(abs(result.evaluate(x)-reference(x,.1))) < 1e-6


def test_adaptive_second_order_fields_are_c2_across_mesh_knots():
    from rotating_bh.vacuum_bvp import solve
    result=solve(.3,method='adaptive',resolution=32,tol=1e-6)
    x=result.nodes[1:-1]
    jump=result.evaluate(x+1e-10,2)[[0,2,3]]-result.evaluate(x-1e-10,2)[[0,2,3]]
    assert np.max(abs(jump))<1e-7
    sample=np.linspace(.02,.98,50);step=1e-5
    diff=(result.evaluate(sample+step,1)-result.evaluate(sample-step,1))/(2*step)
    assert np.max(abs(diff-result.evaluate(sample,2)))<1e-6


@pytest.mark.parametrize('q',[np.nan,np.inf,True,.71])
def test_invalid_spin(q):
    from rotating_bh.vacuum_bvp import solve
    with pytest.raises(ValueError):
        solve(q,method='spectral',resolution=12,tol=1e-6)


def test_iteration_exhaustion_is_failure():
    from rotating_bh.vacuum_bvp import solve, SolverFailure
    with pytest.raises(SolverFailure):
        solve(.3,method='spectral',resolution=12,tol=1e-12,max_iterations=0)


@pytest.mark.parametrize('options',[{'max_iterations':-1},{'max_iterations':1.2},
    {'max_iterations':True},{'max_nodes':-1},{'max_nodes':2.5},{'max_nodes':True},
    {'cutoff':True},{'cutoff':np.nan}])
def test_invalid_resource_options(options):
    from rotating_bh.vacuum_bvp import solve
    with pytest.raises(ValueError):
        solve(0,**options)


def test_returned_representation_rejects_negative_interior_between_nodes():
    from numpy.polynomial import Polynomial
    from rotating_bh.vacuum_bvp import VacuumSolution, SolverFailure
    polynomials=[Polynomial([1,-8,8]),Polynomial([1]),Polynomial([0]),Polynomial([0])]
    evaluate=lambda x,k:np.asarray([p.deriv(k)(x) for p in polynomials])
    with pytest.raises(SolverFailure):
        VacuumSolution(0,'spectral',np.array([0.,1.]),2,1e-8,0.,1,0.,evaluate)
