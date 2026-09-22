"""Independent collocation methods for the static EGB ODE (single unknown F).

Unlike the rotating vacuum system (Hito 2), the derivation in
experiments/derive_static_egb.py shows this system is a *first-order* ODE
that is regular at the horizon (x=0): only F(0)=0 is imposed as a boundary
condition, so both methods are exactly determined without a horizon cutoff.
`cutoff` here only regularizes the removable singularity of the compact
coordinate at spatial infinity (x=1, z=0); F(1-cutoff)~=1 is an independent
diagnostic, never an equation fed to either solver.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.fft import dct
from scipy.integrate import solve_ivp

from ._static_egb_generated import rhs


class SolverFailure(RuntimeError):
    """A numerical attempt failed; no claim about physical existence."""


def ode(x, F, alpha_hat):
    return np.asarray(rhs(1-x, F, alpha_hat))


@dataclass
class StaticEGBSolution:
    alpha_hat: float
    method: str
    nodes: np.ndarray
    initial_resolution: int
    tolerance: float
    cutoff: float
    iterations: int
    _evaluate: object

    def __post_init__(self):
        nodes = np.sort(self.nodes)
        points = np.unique(np.r_[nodes, (nodes[:-1]+nodes[1:])/2])
        values = self._evaluate(points, 0)
        if not np.all(np.isfinite(values)) or np.any(values < -1e-9) or np.any(values > 1+1e-6):
            raise SolverFailure('represented F is nonfinite or outside [0,1]')

    def evaluate(self, x, derivative=0):
        x = np.asarray(x, dtype=float)
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1 or 2')
        if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x > 1-self.cutoff):
            raise ValueError('evaluation must stay inside the solved compact interval')
        return self._evaluate(x, derivative)


def _guess(x, previous):
    if previous is not None:
        xx = np.clip(x, 0, 1-previous.cutoff)
        return previous.evaluate(xx)
    return x*(2-x)/2  # smooth, F(0)=0, order-1 seed; no rotating/EGB reference import.


def _spectral(alpha_hat, n, tol, previous, max_iterations, cutoff):
    L = 1-cutoff
    z = np.cos(np.pi*np.arange(n)/(n-1))
    x = L*(z+1)/2
    weights = np.ones(n); weights[[0, -1]] = 2; weights *= (-1.)**np.arange(n)
    differences = z[:, None]-z[None, :]
    D = (weights[:, None]/weights[None, :])/(differences+np.eye(n))
    D -= np.diag(D.sum(axis=1))
    D *= 2/L

    def residual(F):
        dF = D@F
        out = np.empty_like(F)
        out[:-1] = dF[:-1]-ode(x[:-1], F[:-1], alpha_hat)
        out[-1] = F[-1]-0.
        return out

    F = _guess(x, previous)
    for iteration in range(max_iterations+1):
        res = residual(F)
        norm = np.max(np.abs(res))
        if np.isfinite(norm) and norm < tol:
            break
        if iteration == max_iterations:
            raise SolverFailure(f'spectral Newton exhausted {max_iterations} iterations; residual={norm:g}')
        jac = np.empty((n, n))
        for j in range(n):
            trial = F.astype(complex); trial[j] += 1e-20j
            jac[:, j] = residual(trial).imag/1e-20
        try:
            step = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError as error:
            raise SolverFailure('singular spectral Jacobian') from error
        for power in range(16):
            trial = F+step*2.**(-power)
            if np.min(trial) < -1e-6 or np.max(trial) > 1+1e-3:
                continue
            trial_norm = np.max(np.abs(residual(trial)))
            if np.isfinite(trial_norm) and trial_norm < norm:
                F = trial
                break
        else:
            raise SolverFailure(f'spectral line search failed; residual={norm:g}')
    coefficients = dct(F, type=1)/(n-1)
    coefficients[[0, -1]] *= .5
    polynomial = Chebyshev(coefficients, domain=[0, L])
    derivatives = [polynomial.deriv(k) for k in range(3)]
    evaluate = lambda x, k: derivatives[k](x)
    return StaticEGBSolution(alpha_hat, 'spectral', x, n, tol, cutoff, iteration, evaluate)


def _adaptive(alpha_hat, tol, cutoff):
    result = solve_ivp(lambda x, y: ode(x, y, alpha_hat), (0., 1-cutoff), [0.],
                        method='RK45', dense_output=True, rtol=tol, atol=tol)
    if not result.success or np.any(~np.isfinite(result.y)):
        raise SolverFailure(f'adaptive integration failed: {result.message}')

    def represented(x):
        # The dense-output polynomial itself, not the ODE recomputed at x:
        # an ODE-residual check against this is a real off-node test.
        return result.sol(np.clip(x, 0., 1-cutoff))[0]

    step = 1e-5
    lo, hi = 0., 1-cutoff

    def evaluate(x, k):
        x = np.asarray(x, dtype=float)
        if k == 0:
            return represented(x)
        # A plain symmetric difference silently clips x-step (or x+step) back
        # onto the boundary, which halves the represented slope right at the
        # horizon/cutoff instead of raising (AstraCheck, 2026-09-09; caught
        # via evaluate(0.,1) returning ~half of the closed-form f'(r_H)).
        # Use one-sided, second-order-accurate stencils near each boundary
        # and a central stencil in the interior.
        near_lo = x-step < lo
        near_hi = x+step > hi
        if k == 1:
            central = (represented(x+step)-represented(x-step))/(2*step)
            forward = (-3*represented(x)+4*represented(x+step)-represented(x+2*step))/(2*step)
            backward = (3*represented(x)-4*represented(x-step)+represented(x-2*step))/(2*step)
        else:
            central = (represented(x+step)-2*represented(x)+represented(x-step))/step**2
            forward = (2*represented(x)-5*represented(x+step)+4*represented(x+2*step)
                       - represented(x+3*step))/step**2
            backward = (2*represented(x)-5*represented(x-step)+4*represented(x-2*step)
                        - represented(x-3*step))/step**2
        return np.where(near_lo, forward, np.where(near_hi, backward, central))
    return StaticEGBSolution(alpha_hat, 'adaptive', result.t, len(result.t), tol, cutoff,
                              len(result.t), evaluate)


def solve(alpha_hat, *, method='adaptive', resolution=32, tol=1e-10, previous=None,
          max_iterations=30, cutoff=1e-3):
    """Solve in r_H=1 units (alpha_hat=alpha_GB/r_H^2)."""
    if not isinstance(alpha_hat, Real) or isinstance(alpha_hat, (bool, np.bool_)) \
            or not np.isfinite(alpha_hat) or alpha_hat < 0:
        raise ValueError('alpha_hat must be a finite nonnegative real scalar')
    if isinstance(resolution, bool) or not isinstance(resolution, Integral) or resolution < 8:
        raise ValueError('resolution must be an integer >= 8')
    if not isinstance(tol, Real) or isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be positive and finite')
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')
    if not isinstance(cutoff, Real) or isinstance(cutoff, (bool, np.bool_)) \
            or not np.isfinite(cutoff) or not 0 < cutoff < .05:
        raise ValueError('cutoff must lie in (0,.05)')
    if method == 'spectral':
        return _spectral(alpha_hat, resolution, tol, previous, max_iterations, cutoff)
    if method != 'adaptive':
        raise ValueError('method must be adaptive or spectral')
    return _adaptive(alpha_hat, tol, cutoff)
