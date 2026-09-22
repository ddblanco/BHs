"""Adaptive collocation on a truncated compact interval (x=1-1/r).

The horizon is x=0 and infinity is x=1. Generated equations are shared
with the spectral solver; mesh, collocation and endpoint transport are not.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.integrate import solve_bvp

from ._egb_rotating_compact_generated import rhs
from ._egb_rotating_horizon_generated import horizon


def ode(x, y, alpha_gb):
    """First order system for (B,F,H,W,Bx,Hx,Wx)."""
    Bxx, Fx, Hxx, Wxx = rhs(1-x, *y, alpha_gb)
    return np.asarray([y[4], Fx, y[5], y[6], Bxx, Hxx, Wxx])


def boundary(left, right, omega_h, alpha_gb, cutoff):
    """Second order Taylor transport, using only the equations and parameters.

    The ODE has order seven: four regular-horizon conditions and three
    asymptotic ones. F(infinity)=1 is reserved as an independent check,
    as in the adaptive vacuum solver. No spectral endpoint data enter here.
    """
    dl = ode(cutoff, left, alpha_gb)
    dr = ode(1-cutoff, right, alpha_gb)
    step = 1e-20
    ddl = ode(cutoff+1j*step, left+1j*step*dl, alpha_gb).imag/step
    ddr = ode(1-cutoff+1j*step, right+1j*step*dr, alpha_gb).imag/step
    l = left-cutoff*dl+cutoff**2*ddl/2
    r = right+cutoff*dr+cutoff**2*ddr/2
    relations = horizon(*l, dl[1]-cutoff*ddl[1],
                        dl[6]-cutoff*ddl[6], alpha_gb)
    return np.r_[relations[:3], l[3]-omega_h, r[0]-1, r[5], r[6]]


class SolverFailure(RuntimeError):
    """Adaptive collocation did not produce an acceptable representation."""


@dataclass
class AdaptiveSolution:
    omega_h: float
    alpha_gb: float
    nodes: np.ndarray
    cutoff: float
    tolerance: float
    iterations: int
    boundary_residual: float
    collocation_residual: float
    _evaluate: object
    method: str = 'adaptive'

    def evaluate(self, x, derivative=0):
        x = np.asarray(x, dtype=float)
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1 or 2')
        if (np.any(~np.isfinite(x)) or np.any(x < self.cutoff)
                or np.any(x > 1-self.cutoff)):
            raise ValueError('evaluation must stay inside the solved compact interval')
        return self._evaluate(x, derivative)


def solve(omega_h, alpha_gb, *, previous, resolution=81, cutoff=.003,
          tol=1e-7, max_nodes=4000):
    """Use a supplied numerical profile only as the initial mesh guess."""
    for name, value in [('omega_h', omega_h), ('alpha_gb', alpha_gb),
                        ('tol', tol), ('cutoff', cutoff)]:
        if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)):
            raise ValueError(f'{name} must be a finite real scalar')
    if alpha_gb < 0 or tol <= 0 or not 0 < cutoff < .05:
        raise ValueError('require alpha_gb >= 0, tol > 0 and 0 < cutoff < .05')
    for name, value in [('resolution', resolution), ('max_nodes', max_nodes)]:
        if (not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
                or value < 8):
            raise ValueError(f'{name} must be an integer >= 8')
    if max_nodes < resolution:
        raise ValueError('max_nodes must be >= resolution')
    x = np.linspace(cutoff, 1-cutoff, resolution)
    y = np.vstack((previous.evaluate(x), previous.evaluate(x, 1)[[0, 2, 3]]))
    fun = lambda x, y: ode(x, y, alpha_gb)
    bc = lambda l, r: boundary(l, r, omega_h, alpha_gb, cutoff)

    def jac(x, y):
        result = np.empty((7, 7, x.size))
        for j in range(7):
            trial = y.astype(complex)
            trial[j] += 1e-20j
            result[:, j] = fun(x, trial).imag/1e-20
        return result

    result = solve_bvp(fun, bc, x, y, fun_jac=jac, tol=tol,
                       bc_tol=min(tol, 1e-9), max_nodes=max_nodes)
    if not result.success or not np.all(np.isfinite(result.y)):
        raise SolverFailure(f'{result.message}; status={result.status}; '
                            f'nodes={result.x.size}; '
                            f'rms={np.max(result.rms_residuals):.6g}; '
                            f'boundary={np.max(np.abs(bc(result.y[:, 0], result.y[:, -1]))):.6g}')

    # Integrate the represented velocities, rather than replacing reported
    # derivatives by the ODE. This gives B,H,W a C2 representation, as in
    # vacuum_bvp. The constant drift preserves the solved endpoint values.
    integral = result.sol.antiderivative()
    fields, velocities = [0, 2, 3], [4, 5, 6]
    start = integral(cutoff)[velocities]
    drift = (integral(1-cutoff)[velocities]-start
             -(result.y[fields, -1]-result.y[fields, 0]))/(1-2*cutoff)

    def evaluate(x, k):
        values = result.sol(x, k)[:4].copy()
        shape = (3,)+(1,)*np.ndim(x)
        if k == 0:
            values[fields] = (result.y[fields, 0].reshape(shape)
                              + integral(x)[velocities]-start.reshape(shape)
                              -(x-cutoff)*drift.reshape(shape))
        elif k == 1:
            values[fields] = result.sol(x)[velocities]-drift.reshape(shape)
        else:
            values[fields] = result.sol(x, 1)[velocities]
        return values

    endpoint = lambda x: np.r_[evaluate(x, 0), evaluate(x, 1)[fields]]
    boundary_residual = float(np.max(np.abs(bc(endpoint(cutoff), endpoint(1-cutoff)))))
    sample = np.unique(np.r_[result.x, (result.x[:-1]+result.x[1:])/2,
                             np.linspace(cutoff, 1-cutoff, 257)])
    values = evaluate(sample, 0)
    if (not np.all(np.isfinite(values)) or np.any(values[:2] <= 0)
            or np.any(1+(1-sample)**4*values[2] <= 0)):
        raise SolverFailure('represented metric is nonfinite or loses exterior positivity')
    return AdaptiveSolution(omega_h, alpha_gb, result.x, cutoff, tol, result.niter,
                            boundary_residual, float(np.max(result.rms_residuals)), evaluate)
