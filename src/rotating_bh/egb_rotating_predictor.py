"""Analytic Jacobian and linear-response continuation for the EGB solver (Hito 8).

This lives beside `egb_rotating_bvp` rather than inside it, and that is a
provenance decision, not a stylistic one. Every artifact of Hitos 2-6 records
the sha256 of `egb_rotating_bvp.py` as an input; editing that file -- even
purely additively, even with the default path bit-identical -- breaks the link
between those results and the bytes that produced them, and the honest repair
would be to rerun hours of closed milestones. So the audited module is left
untouched and everything new is here.

What is *not* duplicated: the residual, the boundary conditions, the compact
representation, the acceptance object. All of those are imported. What is
duplicated is the damped-Newton driver, about twenty lines, because the
Jacobian is built inside its loop. `tests/test_egb_rotating_jacobian.py` pins
the two drivers together by checking they reach the same solution.

Two additions:

**The analytic Jacobian.** The solver builds its Newton matrix from 4n
scalar-perturbation residual evaluations, each a call into the 3.5k-line
generated system. The same matrix follows from the chain rule out of seven
*vectorised* complex-step partials of `rhs` and nine of `horizon`, which is
sixteen times faster at N=32 and barely grows with resolution. It is the
identical matrix, checked against the dense build on deliberately
non-solution data, where a chain-rule mistake shows and at a solution it
would not.

**The predictor seed.** Newton starts from
`u_prev + dalpha du/dalpha + domega du/domega` instead of `u_prev`, with the
responses from `J du/dp = -dR/dp` and `dR/dp` by complex step in the
parameter. (The residual is not affine in `alpha_GB`: the generated system is
already solved for the highest derivatives, so the coupling enters
rationally.) Near extremality `|du/dalpha|` reaches about 30, which is why the
previous solution is a bad seed there and why shrinking the step does not
help -- the starting *direction* is what is wrong, not the distance. See
`reports/hito-6-addendum.md`.
"""
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.fft import dct

from ._egb_rotating_compact_generated import rhs
from ._egb_rotating_horizon_generated import horizon as horizon_relations
from .egb_rotating_bvp import (EGBRotatingSolution, SolverFailure, _cheb, _D, _D2,
                               _guess, _residual)

__all__ = ['analytic_jacobian', 'dense_jacobian', 'linear_response',
           'predicted_seed', 'solve']

_STEP = 1e-20
# Which argument of `rhs(z,B,F,H,W,P,Q,V,alpha)` each unknown block feeds.
# Block j is the field u[j]; it enters as a value always, and as a first
# derivative for j=0 (P=Bx), j=2 (Q=Hx) and j=3 (V=Wx). Block j=1 (F) has no
# derivative argument: the system is solved for Fx, not for Fxx.
_VALUE_ARGUMENT = {0: 0, 1: 1, 2: 2, 3: 3}
_DERIVATIVE_ARGUMENT = {0: 4, 2: 5, 3: 6}


def analytic_jacobian(flat, n, omega_h, alpha_gb, zz):
    """The Newton matrix of `egb_rotating_bvp._residual`, by the chain rule."""
    D, D2 = _D(n), _D2(n)
    u = flat.reshape(4, n)
    du = u@D.T
    ddu = u@D2.T
    B, F, H, W = u
    P, Fx, Q, V = du
    scales = [zz*zz*(1-zz)**2, zz*(1-zz), zz*(1-zz)**2, zz*(1-zz)**2]
    inner = slice(1, -1)
    arguments = [B[inner], F[inner], H[inner], W[inner], P[inner], Q[inner], V[inner]]
    partials = []
    for a in range(7):
        perturbed = [np.asarray(v, dtype=complex).copy() for v in arguments]
        perturbed[a] += _STEP*1j
        partials.append([np.asarray(o).imag/_STEP
                         for o in rhs(zz[inner], *perturbed, alpha_gb)])
    jac = np.zeros((4*n, 4*n))
    count = n-2
    index = np.arange(count)
    for i in range(4):
        rows = slice(i*n+1, i*n+n-1)
        for j in range(4):
            block = np.zeros((count, n))
            if i == 1 and j == 1:
                block += D[inner, :]
            if i != 1 and j == i:
                block += D2[inner, :]
            block[index, index+1] -= partials[_VALUE_ARGUMENT[j]][i]
            if j in _DERIVATIVE_ARGUMENT:
                block -= partials[_DERIVATIVE_ARGUMENT[j]][i][:, None]*D[inner, :]
            jac[rows, j*n:(j+1)*n] = block*scales[i][inner][:, None]
    # x=1 endpoint (node 0, r -> infinity): B=1, F=1, Hx=0, Wx=0.
    jac[0*n, 0*n] = 1.0
    jac[1*n, 1*n] = 1.0
    jac[2*n, 2*n:3*n] = D[0, :]
    jac[3*n, 3*n:4*n] = D[0, :]
    # x=0 endpoint (node n-1, the horizon): three generated relations, whose
    # nine arguments reach the unknowns as values, first derivatives (P,Q,V,Fx)
    # and one second derivative (Wxx).
    horizon_arguments = [B[-1], F[-1], H[-1], W[-1], P[-1], Q[-1], V[-1],
                         Fx[-1], ddu[3, -1]]
    chain = [(0, None), (1, None), (2, None), (3, None),
             (0, D[-1, :]), (2, D[-1, :]), (3, D[-1, :]),
             (1, D[-1, :]), (3, D2[-1, :])]
    for a, (block, vector) in enumerate(chain):
        perturbed = [complex(v) for v in horizon_arguments]
        perturbed[a] += _STEP*1j
        derivative = [complex(o).imag/_STEP
                      for o in horizon_relations(*perturbed, alpha_gb)]
        for i in range(3):
            if vector is None:
                jac[i*n+n-1, block*n+n-1] += derivative[i]
            else:
                jac[i*n+n-1, block*n:(block+1)*n] += derivative[i]*vector
    jac[3*n+n-1, 3*n+n-1] = 1.0  # W(horizon) = omega_h
    return jac


def dense_jacobian(residual, flat, n):
    """The same matrix the audited solver builds, column by column."""
    jac = np.empty((4*n, 4*n))
    for j in range(4*n):
        trial = flat.astype(complex)
        trial[j] += _STEP*1j
        jac[:, j] = residual(trial).imag/_STEP
    return jac


def _parameter_derivative(flat, n, omega_h, alpha_gb, zz, parameter):
    """dR/dparameter at fixed unknowns, by complex step in that parameter."""
    trial = flat.astype(complex)
    if parameter == 'alpha_gb':
        return _residual(trial, n, omega_h, complex(alpha_gb, _STEP), zz).imag/_STEP
    if parameter == 'omega_h':
        return _residual(trial, n, complex(omega_h, _STEP), alpha_gb, zz).imag/_STEP
    raise ValueError("parameter must be 'alpha_gb' or 'omega_h'")


def linear_response(flat, n, omega_h, alpha_gb, zz, jacobian='analytic',
                    parameter='alpha_gb'):
    """du/dparameter along the solution branch, from one linear solve.

    Differentiating R(u;p)=0 gives J du/dp = -dR/dp. Both pieces are exact to
    machine precision, so this needs no Newton iteration and no extrapolation:
    at alpha_GB=0, evaluated on the closed-form Myers-Perry profile, du/dalpha
    is the perturbative response itself.

    The same Jacobian serves both parameters, so asking for both costs one
    factorisation and two back-substitutions rather than two Jacobians.
    """
    if jacobian == 'analytic':
        jac = analytic_jacobian(flat, n, omega_h, alpha_gb, zz)
    else:
        jac = dense_jacobian(lambda f: _residual(f, n, omega_h, alpha_gb, zz),
                             flat, n)
    parameters = (parameter,) if isinstance(parameter, str) else tuple(parameter)
    columns = np.column_stack([-_parameter_derivative(flat, n, omega_h, alpha_gb,
                                                      zz, p) for p in parameters])
    try:
        solved = np.linalg.solve(jac, columns)
    except np.linalg.LinAlgError as error:
        raise SolverFailure('singular Jacobian in the linear response') from error
    return solved[:, 0] if isinstance(parameter, str) else solved.T


def predicted_seed(flat, previous, omega_h, alpha_gb, zz, n, jacobian='analytic'):
    """Transport the previous solution along the branch tangent.

    `u_prev + dalpha du/dalpha + domega du/domega`, with both gaps measured
    from wherever `previous` actually sits. A step in one parameter alone is the
    common case; allowing both lets one continuation move diagonally towards
    extremality at finite coupling, which is where neither parameter can be
    advanced far on its own.

    The response is evaluated in the *target* discretisation, on the previous
    profile sampled at the new nodes, so changing resolution costs nothing
    extra and the predictor stays a seed -- the gates still decide.
    """
    base_alpha = getattr(previous, 'alpha_gb', None)
    base_omega = getattr(previous, 'omega_h', None)
    if base_alpha is None or base_omega is None:
        raise ValueError('the predictor needs a previous solution carrying its '
                         'alpha_gb and omega_h')
    gaps = [float(alpha_gb)-float(base_alpha), float(omega_h)-float(base_omega)]
    wanted = [p for p, g in zip(('alpha_gb', 'omega_h'), gaps) if g != 0.]
    if not wanted:
        return flat
    responses = linear_response(flat, n, float(base_omega), float(base_alpha), zz,
                                jacobian, parameter=wanted)
    for value, gap in zip(responses, [g for g in gaps if g != 0.]):
        flat = flat + gap*value
    return flat


def solve(omega_h, alpha_gb, *, resolution=32, tol=1e-9, previous=None,
          max_iterations=50, jacobian='analytic', seed='previous'):
    """`egb_rotating_bvp.solve` with the analytic Jacobian and predictor seeding.

    Same residual, same boundary rows, same representation, same acceptance
    object; only the Newton driver is written out again, because the Jacobian is
    built inside its loop. With `jacobian='complex-step'` and `seed='previous'`
    this is the audited solver's algorithm line for line, which is what
    `tests/test_egb_rotating_jacobian.py` checks.
    """
    if alpha_gb < 0:
        raise ValueError('alpha_gb must be non-negative')
    if jacobian not in ('complex-step', 'analytic'):
        raise ValueError("jacobian must be 'complex-step' or 'analytic'")
    if seed not in ('previous', 'predictor'):
        raise ValueError("seed must be 'previous' or 'predictor'")
    if seed == 'predictor' and previous is None:
        raise ValueError("seed='predictor' requires a previous solution")
    n = int(resolution)
    x, _, _ = _cheb(n)
    zz = 1-x

    def residual(vector):
        return _residual(vector, n, omega_h, alpha_gb, zz)

    flat = _guess(x, previous, omega_h).ravel()
    if seed == 'predictor':
        flat = predicted_seed(flat, previous, omega_h, alpha_gb, zz, n, jacobian)
    for iteration in range(max_iterations+1):
        res = residual(flat)
        norm = np.max(np.abs(res))
        if np.isfinite(norm) and norm < tol:
            break
        if iteration == max_iterations:
            raise SolverFailure(f'spectral Newton exhausted {max_iterations} '
                                f'iterations; residual={norm:g}')
        if jacobian == 'analytic':
            jac = analytic_jacobian(flat, n, omega_h, alpha_gb, zz)
        else:
            jac = dense_jacobian(residual, flat, n)
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
    evaluate = lambda point, k: np.asarray([p(point) for p in derivatives[k]])
    boundary = float(max(np.max(np.abs(res.reshape(4, n)[:, 0])),
                         np.max(np.abs(res.reshape(4, n)[:, -1]))))
    return EGBRotatingSolution(omega_h, alpha_gb, 'spectral', x, n, tol, iteration,
                               boundary, evaluate)
