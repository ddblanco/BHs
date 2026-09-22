"""Spectral collocation solver for the general rotating EGB system (Hito 4B).

Mirrors vacuum_bvp.py's `_spectral` architecture exactly (Chebyshev-Lobatto
nodes, damped Newton with a complex-step Jacobian and line search) over the
compact amplitude variables B,F,H,W (b=(1-z^2)B, f=(1-z^2)F,
h=(1+z^4 H)/z^2, w=z^4 W, z=1/r, x=1-z -- x=0 is r->infinity, x=1 is the
horizon r=r_H=1).

Interior collocation uses `_egb_rotating_compact_generated.rhs` (verified
in derive_egb_rotating.py to reproduce `_vacuum_generated.rhs` exactly at
alpha_gb=0). The horizon (x=1) is singular for generic data in that
formula -- not a bug, the coefficient matrix of the underlying [f',b'',h'',
w''] system degenerates there (docs/egb-rotating.md) -- so that row uses
`_egb_rotating_horizon_generated.horizon` instead, four relations derived
directly in these compact variables and verified against the exact
Myers-Perry closed form. Infinity (x=0) uses B=1,F=1,Hx=0,Wx=0, the same
four conditions vacuum_bvp.py's own spectral solver imposes there (checked
in derive_egb_rotating.py to also hold for this system's leading
asymptotic order; alpha_gb does not enter until 1/r^6, past what these four
probe). W(x=1)=omega_h (the horizon angular velocity) replaces what would
otherwise be a fourth, non-independent horizon relation -- confirmed
empirically: using this in place of the raw 'w' relation reproduces
Myers-Perry to the solver's tolerance at alpha_gb=0, and the raw 'w'
relation was found to genuinely determine Wxx (not to be redundant), so it
is not an extra condition being dropped, just not the one used here.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.fft import dct

from ._egb_rotating_compact_generated import rhs
from ._egb_rotating_horizon_generated import horizon as horizon_relations


class SolverFailure(RuntimeError):
    """A numerical attempt failed; no claim about physical existence."""


def _residual(flat, n, omega_h, alpha_gb, zz):
    u = flat.reshape(4, n)
    du = u@_D(n).T
    ddu = u@_D2(n).T
    B, F, H, W = u
    P, Fx, Q, V = du
    out = np.empty_like(u)
    scales = [zz*zz*(1-zz)**2, zz*(1-zz), zz*(1-zz)**2, zz*(1-zz)**2]
    interior = rhs(zz[1:-1], B[1:-1], F[1:-1], H[1:-1], W[1:-1],
                   P[1:-1], Q[1:-1], V[1:-1], alpha_gb)
    for i in range(4):
        represented = du[i] if i == 1 else ddu[i]
        out[i, 1:-1] = (represented[1:-1]-interior[i])*scales[i][1:-1]
    out[:, 0] = [B[0]-1, F[0]-1, du[2, 0], du[3, 0]]
    out[:, -1] = list(horizon_relations(B[-1], F[-1], H[-1], W[-1], P[-1], Q[-1], V[-1],
                                         Fx[-1], ddu[3, -1], alpha_gb))[:3]+[W[-1]-omega_h]
    return out.ravel()


_cache = {}


def _cheb(n):
    if n not in _cache:
        z = np.cos(np.pi*np.arange(n)/(n-1))
        x = (z+1)/2
        weights = np.ones(n); weights[[0, -1]] = 2; weights *= (-1.)**np.arange(n)
        differences = z[:, None]-z[None, :]
        D = (weights[:, None]/weights[None, :])/(differences+np.eye(n))
        D -= np.diag(D.sum(axis=1)); D *= 2
        _cache[n] = (x, D, D@D)
    return _cache[n]


def _D(n):
    return _cheb(n)[1]


def _D2(n):
    return _cheb(n)[2]


@dataclass
class EGBRotatingSolution:
    omega_h: float
    alpha_gb: float
    method: str
    nodes: np.ndarray
    initial_resolution: int
    tolerance: float
    iterations: int
    boundary_residual: float
    _evaluate: object

    def __post_init__(self):
        nodes = np.sort(self.nodes)
        points = np.unique(np.r_[nodes, (nodes[:-1]+nodes[1:])/2])
        values = self._evaluate(points, 0)
        if not np.all(np.isfinite(values)) or np.any(values[:2] <= 0):
            raise SolverFailure('represented B or F is nonfinite or non-positive')
        zz = 1-points
        h_over_r2 = 1+zz**4*values[2]  # h/r^2 = 1+z^4 H, must stay positive
        if np.any(h_over_r2 <= 0):
            raise SolverFailure('represented metric loses exterior positivity (h/r^2<=0)')

    def evaluate(self, x, derivative=0):
        """Return B,F,H,W (or their x-derivatives) at compact coordinate x.

        b=(1-z^2)B, f=(1-z^2)F, h=(1+z^4 H)/z^2, w=z^4 W, z=1-x, r=1/z.
        """
        x = np.asarray(x, dtype=float)
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1 or 2')
        if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
            raise ValueError('evaluation must stay inside [0,1]')
        return self._evaluate(x, derivative)


def _guess(x, previous, omega_h):
    if previous is not None:
        return previous.evaluate(x)
    # A smooth perturbation of the (alpha_gb=0, omega_h) Myers-Perry
    # solution would be a better guess, but omega_h is only known at solve
    # time; callers should pass `previous` from a nearby accepted solution
    # (continuation) for anything past the trivial case below.
    u = np.zeros((4, len(x)))
    u[0] = 1.0; u[1] = 1.0; u[3] = omega_h
    return u


def _spectral(omega_h, alpha_gb, n, tol, previous, max_iterations):
    x, D, D2 = _cheb(n)
    zz = 1-x

    def residual(flat):
        return _residual(flat, n, omega_h, alpha_gb, zz)

    flat = _guess(x, previous, omega_h).ravel()
    for iteration in range(max_iterations+1):
        res = residual(flat)
        norm = np.max(np.abs(res))
        if np.isfinite(norm) and norm < tol:
            break
        if iteration == max_iterations:
            raise SolverFailure(f'spectral Newton exhausted {max_iterations} iterations; residual={norm:g}')
        jac = np.empty((4*n, 4*n))
        for j in range(4*n):
            trial = flat.astype(complex); trial[j] += 1e-20j
            jac[:, j] = residual(trial).imag/1e-20
        try:
            step = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError as error:
            raise SolverFailure('singular spectral Jacobian') from error
        for power in range(20):
            trial = flat+step*2.**(-power)
            fields = trial.reshape(4, n)
            if np.min(fields[:2]) <= 0:
                continue
            trial_norm = np.max(np.abs(residual(trial)))
            if np.isfinite(trial_norm) and trial_norm < norm:
                flat = trial
                break
        else:
            raise SolverFailure(f'spectral line search failed; residual={norm:g}')
    coefficients = dct(flat.reshape(4, n), type=1, axis=1)/(n-1)
    coefficients[:, [0, -1]] *= .5
    polynomials = [Chebyshev(c, domain=[0, 1]) for c in coefficients]
    derivatives = [[p.deriv(k) for p in polynomials] for k in range(3)]
    evaluate = lambda x, k: np.asarray([p(x) for p in derivatives[k]])
    boundary_residual = float(max(np.max(np.abs(res.reshape(4, n)[:, 0])),
                                   np.max(np.abs(res.reshape(4, n)[:, -1]))))
    return EGBRotatingSolution(omega_h, alpha_gb, 'spectral', x, n, tol, iteration,
                                boundary_residual, evaluate)


def solve(omega_h, alpha_gb, *, resolution=32, tol=1e-9, previous=None, max_iterations=50):
    """Solve in r_H=1 units (alpha_gb=alpha_GB/r_H^2, omega_h=r_H*Omega_H).

    Only the spectral method is implemented (Hito 4B, first pass); no
    adaptive/solve_bvp method yet -- see docs/egb-rotating.md.
    """
    if not isinstance(omega_h, Real) or isinstance(omega_h, (bool, np.bool_)) or not np.isfinite(omega_h):
        raise ValueError('omega_h must be a finite real scalar')
    if not isinstance(alpha_gb, Real) or isinstance(alpha_gb, (bool, np.bool_)) \
            or not np.isfinite(alpha_gb) or alpha_gb < 0:
        raise ValueError('alpha_gb must be a finite nonnegative real scalar')
    if isinstance(resolution, bool) or not isinstance(resolution, Integral) or resolution < 8:
        raise ValueError('resolution must be an integer >= 8')
    if not isinstance(tol, Real) or isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be positive and finite')
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) \
            or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')
    return _spectral(omega_h, alpha_gb, resolution, tol, previous, max_iterations)
