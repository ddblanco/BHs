"""Off-collocation checks of represented numerical metrics, independent of ODEs."""
from functools import lru_cache

import numpy as np
import sympy as sp

from .einstein import compile_metric, curvature, metric_from_ansatz
from .vacuum_equations import radial_equations
from ._vacuum_generated import rhs
from .vacuum_bvp import adaptive_boundary, horizon


def physical_jets(solution,r,*,r_h=1.):
    """Actual represented b,f,h,w and their first two radial derivatives."""
    z=r_h/np.asarray(r);x=1-z;A=1-z*z
    B,F,H,W=solution.evaluate(x)
    P,T,Q,V=solution.evaluate(x,1)
    PP,TT,QQ,VV=solution.evaluate(x,2)
    values=np.array([A*B,A*F,r_h*r_h*(z**-2+z*z*H),z**4*W/r_h])
    dx=np.array([2*z*B+A*P,2*z*F+A*T,
        r_h*r_h*(2*z**-3-2*z*H+z*z*Q),(z**4*V-4*z**3*W)/r_h])
    dxx=np.array([-2*B+4*z*P+A*PP,-2*F+4*z*T+A*TT,
        r_h*r_h*(6*z**-4+2*H-4*z*Q+z*z*QQ),
        (z**4*VV-8*z**3*V+12*z*z*W)/r_h])
    return np.array([values,z*z/r_h*dx,(z**4*dxx-2*z**3*dx)/r_h**2])


@lru_cache(maxsize=1)
def _metric_jets():
    t,r,theta,p1,p2=sp.symbols('t r theta p1 p2',real=True)
    R=sp.symbols('R',positive=True)
    parameters=sp.symbols('v0:12')
    functions=[parameters[i]+parameters[i+4]*(r-R)+parameters[i+8]*(r-R)**2/2 for i in range(4)]
    return compile_metric(metric_from_ansatz(r,theta,*functions),
                          (t,r,theta,p1,p2),(R,*parameters))


def tensor_residual(r,theta,jets,*,r_h=1.):
    """Hito 1 coordinate norm on a generic local metric jet, without MP input."""
    arrays=_metric_jets()(0,r,theta,0,0,r,*np.asarray(jets).ravel())
    tensor=curvature(*arrays)['einstein']
    scales=np.array([1,1,r,r,r])
    return float(r_h*r_h*np.max(abs(tensor/np.outer(scales,scales))))


def check_points(solution,count=401):
    # A fixed interior sample; move coincidences away from the solver's nodes.
    lower=max(.005,solution.cutoff);upper=min(.995,1-solution.cutoff)
    step=(upper-lower)/count
    points=lower+(np.arange(count)+.5)*step
    for i in range(count):
        while np.min(abs(points[i]-solution.nodes))<1e-12:
            points[i]+=step/8
    return points


def reference_profiles(x,q):
    """Analytic oracle confined to validation, never used by either solver."""
    z=1-x;a=q*q/(1-q*q)
    return np.array([(1-a*z*z)/(1+a*z**4),1-a*z*z,
                     np.full_like(x,a),q/(1-q*q+q*q*z**4)])


def diagnose(solution):
    x=check_points(solution);z=1-x
    u=solution.evaluate(x);du=solution.evaluate(x,1);ddu=solution.evaluate(x,2)
    expected=rhs(z,*np.vstack((u,du[[0,2,3]])))
    scales=[x*x*z*z,x*z,x*z*z,x*z*z]
    defects=[((du[i] if i==1 else ddu[i])-expected[i])*scales[i] for i in range(4)]
    r=1/z;jets=physical_jets(solution,r)
    variations=radial_equations(r,*jets)
    # E_f*f/r² = G^r_r. This equation is absent from both solvers.
    constraint=float(np.max(abs(variations['f']*jets[0,1]/r**2)))
    tensors=[];coverage=True
    for radius in (1.05,1.3,2.,5.,20.):
        if not solution.cutoff<=1-1/radius<=1-solution.cutoff:
            coverage=False
            continue
        local=physical_jets(solution,radius)
        for theta in (.2,.7,1.2):
            tensors.append(tensor_residual(radius,theta,local))
    positive=bool(np.all(np.isfinite(jets)) and np.all(jets[0,:3]>0))
    c=solution.cutoff
    left=solution.evaluate(c);left_d=solution.evaluate(c,1)
    right=solution.evaluate(1-c);right_d=solution.evaluate(1-c,1)
    at_horizon=left-c*left_d+c*c*solution.evaluate(c,2)/2
    simple=bool(np.all(np.isfinite(at_horizon)) and np.all(at_horizon[:2]>0)
                and 1+at_horizon[2]>0)
    boundary=None
    if simple:
        l=np.r_[left,left_d[[0,2,3]]];r=np.r_[right,right_d[[0,2,3]]]
        if c:
            boundary=float(np.max(abs(adaptive_boundary(l,r,solution.q,c))))
        else:
            boundary=float(np.max(abs(np.r_[horizon(l,left_d[1],solution.q),
                                            right[:2]-1,right_d[[2,3]]])))
    measurements={
        'profile_error':float(np.max(abs(u-reference_profiles(x,solution.q)))),
        'ode_residual':float(np.max(np.abs(defects))),
        'constraint_residual':constraint,'einstein_residual':max(tensors),
        'boundary_residual':boundary,'positive_exterior':positive,
        'tensor_sample_coverage':coverage,'simple_horizon':simple,
        'horizon_slopes':(2*at_horizon[:2]).tolist()}
    measurements['accepted']=bool(positive and coverage and simple and all(np.isfinite(measurements[k]) and measurements[k]<1e-6
        for k in ('profile_error','ode_residual','constraint_residual','einstein_residual'))
        and boundary is not None and np.isfinite(boundary) and 0<=boundary<1e-8)
    return measurements
