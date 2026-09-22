"""Independent collocation methods for the compact equal-spin vacuum system."""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.fft import dct
from scipy.integrate import solve_bvp

from ._vacuum_generated import rhs


class SolverFailure(RuntimeError):
    """A numerical attempt failed; no claim about physical existence."""


def ode(x,y):
    """First order system in B,F,H,W,Bx,Hx,Wx, excluding E_f."""
    a,b,c,d=rhs(1-x,*y)
    return np.asarray([y[4],b,y[5],y[6],a,c,d])


def horizon(y,fp,q):
    B,F,H,W,P,Q,V=y
    return np.asarray([F+H-1,
        Q-8*H/(1-H)+(1+H)**2*(V-4*W)**2/(2*B),
        fp+3*F*P/B+4*H-1.5*F*(1+H)*(V-4*W)**2/B,W-q])


def adaptive_boundary(left,right,q,cutoff):
    """Second order Taylor transport to both true endpoints.

    Derivatives here define boundary conditions only, never reported residuals.
    Complex directional differentiation gives yxx of the analytic ODE.
    """
    dl=ode(cutoff,left);dr=ode(1-cutoff,right)
    step=1e-20
    ddl=ode(cutoff+1j*step,left+1j*step*dl).imag/step
    ddr=ode(1-cutoff+1j*step,right+1j*step*dr).imag/step
    l=left-cutoff*dl+cutoff**2*ddl/2
    r=right+cutoff*dr+cutoff**2*ddr/2
    return np.concatenate((horizon(l,dl[1]-cutoff*ddl[1],q),[r[0]-1,r[5],r[6]]))


@dataclass
class VacuumSolution:
    q: float
    method: str
    nodes: np.ndarray
    initial_resolution: int
    tolerance: float
    cutoff: float
    iterations: int
    boundary_residual: float
    _evaluate: object

    def __post_init__(self):
        nodes=np.sort(self.nodes)
        points=np.unique(np.r_[nodes,(nodes[:-1]+nodes[1:])/2,
                               np.linspace(self.cutoff,1-self.cutoff,257)])
        values=self._evaluate(points,0)
        if (not np.all(np.isfinite(values)) or np.any(values[:2]<=0)
                or np.any(1+(1-points)**4*values[2]<=0)):
            raise SolverFailure('represented metric is nonfinite or loses exterior positivity')

    def evaluate(self,x,derivative=0):
        x=np.asarray(x,dtype=float)
        if derivative not in (0,1,2):
            raise ValueError('derivative must be 0, 1 or 2')
        if np.any(~np.isfinite(x)) or np.any(x<self.cutoff) or np.any(x>1-self.cutoff):
            raise ValueError('evaluation must stay inside the solved compact interval')
        return self._evaluate(x,derivative)


def _guess(x,previous):
    if previous is not None:
        # Both methods continue from the represented numerical solution.
        xx=np.clip(x,previous.cutoff,1-previous.cutoff)
        u=previous.evaluate(xx)
        du=previous.evaluate(xx,1)
        return np.vstack((u,du[[0,2,3]]))
    # A smooth perturbation of the static seed; no rotating reference import.
    perturb=.002*x*(1-x)
    u=np.zeros((7,len(x)))
    u[0]=1+perturb;u[1]=1-perturb
    u[2]=perturb
    u[4]=.002*(1-2*x);u[5]=u[4]
    return u


def _spectral(q,n,tol,previous,max_iterations):
    z=np.cos(np.pi*np.arange(n)/(n-1));x=(z+1)/2
    weights=np.ones(n);weights[[0,-1]]=2;weights*=(-1.)**np.arange(n)
    differences=z[:,None]-z[None,:]
    D=(weights[:,None]/weights[None,:])/(differences+np.eye(n))
    D-=np.diag(D.sum(axis=1));D*=2
    D2=D@D

    def residual(flat):
        u=flat.reshape(4,n);du=u@D.T;ddu=u@D2.T
        y=np.vstack((u,du[[0,2,3]]))
        interior=rhs(1-x[1:-1],*y[:,1:-1])
        out=np.empty_like(u)
        scales=[x*x*(1-x)**2,x*(1-x),x*(1-x)**2,x*(1-x)**2]
        for i in range(4):
            represented=du[i] if i==1 else ddu[i]
            out[i,1:-1]=(represented[1:-1]-interior[i])*scales[i][1:-1]
        out[:,-1]=horizon(y[:,-1],du[1,-1],q)
        out[:,0]=[u[0,0]-1,u[1,0]-1,du[2,0],du[3,0]]
        return out.ravel()

    flat=_guess(x,previous)[:4].ravel()
    for iteration in range(max_iterations+1):
        res=residual(flat)
        norm=np.max(abs(res))
        if np.isfinite(norm) and norm<tol:
            break
        if iteration==max_iterations:
            raise SolverFailure(f'spectral Newton exhausted {max_iterations} iterations; residual={norm:g}')
        jac=np.empty((4*n,4*n))
        for j in range(4*n):
            trial=flat.astype(complex);trial[j]+=1e-20j
            jac[:,j]=residual(trial).imag/1e-20
        try:
            step=np.linalg.solve(jac,-res)
        except np.linalg.LinAlgError as error:
            raise SolverFailure('singular spectral Jacobian') from error
        for power in range(16):
            trial=flat+step*2.**(-power)
            fields=trial.reshape(4,n)
            if np.min(fields[:2])<=0 or np.min(1+(1-x)**4*fields[2])<=0:
                continue
            trial_norm=np.max(abs(residual(trial)))
            if np.isfinite(trial_norm) and trial_norm<norm:
                flat=trial
                break
        else:
            raise SolverFailure(f'spectral line search failed; residual={norm:g}')
    coefficients=dct(flat.reshape(4,n),type=1,axis=1)/(n-1)
    coefficients[:,[0,-1]]*=.5
    polynomials=[Chebyshev(c,domain=[0,1]) for c in coefficients]
    derivatives=[[p.deriv(k) for p in polynomials] for k in range(3)]
    evaluate=lambda x,k:np.asarray([p(x) for p in derivatives[k]])
    return VacuumSolution(q,'spectral',x,n,tol,0.,iteration,
        float(max(np.max(abs(res.reshape(4,n)[:,0])),np.max(abs(res.reshape(4,n)[:,-1])))),evaluate)


def solve(q,*,method='adaptive',resolution=32,tol=1e-8,previous=None,
          max_iterations=30,max_nodes=10000,cutoff=.003):
    """Solve in r_H=1 units; continuation seeds are numerical representations."""
    if not isinstance(q,Real) or isinstance(q,(bool,np.bool_)) or not np.isfinite(q) or abs(q)>=1/np.sqrt(2):
        raise ValueError('q must be a finite nonextremal real scalar')
    if isinstance(resolution,bool) or not isinstance(resolution,Integral) or resolution<8:
        raise ValueError('resolution must be an integer >= 8')
    if not isinstance(tol,Real) or isinstance(tol,bool) or not np.isfinite(tol) or tol<=0:
        raise ValueError('tol must be positive and finite')
    for name,value,minimum in [('max_iterations',max_iterations,0),('max_nodes',max_nodes,2)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not isinstance(cutoff,Real) or isinstance(cutoff,(bool,np.bool_)) or not np.isfinite(cutoff) or not 0<cutoff<.05:
        raise ValueError('cutoff must lie in (0,.05)')
    if method=='spectral':
        return _spectral(q,resolution,tol,previous,max_iterations)
    if method!='adaptive':
        raise ValueError('method must be adaptive or spectral')
    x=np.linspace(cutoff,1-cutoff,resolution)
    boundary=lambda l,r:adaptive_boundary(l,r,q,cutoff)
    result=solve_bvp(ode,boundary,x,_guess(x,previous),tol=tol,bc_tol=min(tol,1e-9),max_nodes=max_nodes)
    if not result.success or np.any(~np.isfinite(result.y)):
        raise SolverFailure(f'adaptive solver failed: {result.message}')
    integral=result.sol.antiderivative()
    indices=[0,2,3];velocities=[4,5,6]
    start=integral(cutoff)[velocities]
    drift=(integral(1-cutoff)[velocities]-start
           -(result.y[indices,-1]-result.y[indices,0]))/(1-2*cutoff)

    def evaluate(x,k):
        data=result.sol(x,k)[:4].copy()
        shape=(3,)+(1,)*np.ndim(x)
        if k==0:
            data[indices]=(result.y[indices,0].reshape(shape)+integral(x)[velocities]
                           -start.reshape(shape)-(x-cutoff)*drift.reshape(shape))
        elif k==1:
            data[indices]=result.sol(x)[velocities]-drift.reshape(shape)
        else:
            data[indices]=result.sol(x,1)[velocities]
        return data

    def endpoint(x):
        return np.r_[evaluate(x,0),evaluate(x,1)[indices]]
    return VacuumSolution(q,'adaptive',result.x,resolution,tol,cutoff,result.niter,
        float(np.max(abs(boundary(endpoint(cutoff),endpoint(1-cutoff))))),evaluate)
