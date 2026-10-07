"""The extremal branch, constructed directly at T = 0 (units r_H = g_H = 1),
with the production solver extremal_rq.py.

At each coupling alpha the solution is computed at two resolutions; the
observables, the invariants and the slope are recorded with their
resolution differences, together with the near-horizon data.

Slope by implicit differentiation of the discrete system (no finite
difference in alpha): A dq/dalpha = -dR/dalpha, then
    M_a = -(3 pi/8) dQ1(1)/dalpha,  J_a by complex step through p_w,
    mu'(y) = (M_a - (2/3) M J_a/J) / (1 - (2/3) alpha J_a/J)
          = dM_ext/dalpha|_J   (scale invariance).
Two identities are evaluated, not imposed:
    first law at T = 0 along the branch:  Psi_FL = M_a - 2 Omega_H J_a ,
    Smarr at T = 0:                      Psi_S  = (M - 3 Omega_H J)/alpha ;
mu' = Psi_FL holds if and only if Smarr holds.

Acceptance of a step: residual < 5e-8, constraint < 1e-6 (spurious aliasing
roots have ~1e-3), and the last Chebyshev coefficients of every field below
1e-6 of the first (resolved).

Resumable: branch_rq.json is rewritten after every accepted coupling with the
Chebyshev coefficients of both solutions; a rerun continues from the last.

Run: python branch_rq.py [alpha_max] [N_low] [N_high]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from extremal_rq import ExtremalRQ, myers_perry_q, invariants, H_STEP
from _pw_generated_p0 import J_of

warnings.simplefilter('ignore', RuntimeWarning)
HERE = Path(__file__).resolve().parent
OUT = HERE/'branch_rq.json'
CH = np.polynomial.chebyshev


def slope(e, flat, alpha):
    n = e.n
    A = e.jacobian(flat, alpha)
    dR = e.residual(flat.astype(complex), alpha+1j*H_STEP).imag/H_STEP
    dq = np.linalg.solve(A, -dR).reshape(4, n)
    M_a = -3*np.pi/8*dq[0, -1]
    m = int(np.argmin(np.abs(e.x-0.5)))
    qc = flat.reshape(4, n)+1j*H_STEP*dq
    J = e.qjets(qc)
    Jc = J_of(e.x[m], J[0][m], J[1][m], J[3][m], J[4][m], J[6][m], J[7][m], J[9][m], J[10][m],
              alpha+1j*H_STEP)
    J_a = float(np.imag(Jc)/H_STEP)
    Omega_a = float(e.qmats[0][0]@dq[3])
    return dict(M_a=float(M_a), J_a=J_a, Omega_a=Omega_a)


def record(e, flat, alpha, info):
    o = e.observables(flat, alpha)
    inv = invariants(o)
    out = dict(o, **inv, **info)
    if alpha > 0:
        d = slope(e, flat, alpha)
        M, J, Om = o['M'], o['J'], o['Omega_H']
        out.update(d, mu_prime=float((d['M_a']-2/3*M*d['J_a']/J)/(1-2/3*alpha*d['J_a']/J)),
                   psi_first_law=float(d['M_a']-2*Om*d['J_a']),
                   psi_smarr=float((M-3*Om*J)/alpha))
    tails = [float(np.max(np.abs(c[-4:]))/max(np.max(np.abs(c)), 1e-300)) for c in e.coefficients(flat)]
    out['tails'] = tails
    return out


def robust_solve(e, alpha, guess):
    """Newton, then, if the Chebyshev tails show grid-scale noise (> 1e-7),
    drop the top quarter of the coefficients and solve again; keep the
    solution with the smaller constraint violation. Grid-scale modes are
    weakly constrained at the round-off floor, and Newton can stop with
    ~1e-6 noise in them; the filtered restart removes it (diag_polish.py)."""
    sol, info = e.solve(alpha, guess)
    rec = record(e, sol, alpha, info)
    if max(rec['tails']) > 1e-7:
        keep = 3*e.n//4
        start = np.array([CH.chebval(1-2*e.x, c[:keep]) for c in e.coefficients(sol)]).ravel()
        sol2, info2 = e.solve(alpha, start)
        rec2 = record(e, sol2, alpha, info2)
        if rec2['constraint'] < rec['constraint'] and rec2['residual'] < 5e-8:
            sol, info, rec = sol2, info2, rec2
    return sol, rec


CONSTRAINT_MAX = 1e-6        # phase 2 (large alpha) uses 1e-5: see main()
RESIDUAL_MAX = 5e-8          # round-off floor grows with N; phase 4 (N >= 144) uses 2e-7


def tangent(e, flat, alpha):
    """dq/dalpha at an accepted solution (implicit differentiation)."""
    e.set_weights(flat, alpha)
    A = e.jacobian(flat, alpha)
    dR = e.residual(flat.astype(complex), alpha+1j*H_STEP).imag/H_STEP
    return np.linalg.solve(A, -dR)


def acceptable(o):
    # the residual floor at N = 96 is ~1e-8 (rows next to infinity); spurious
    # roots are caught by the constraint and the Chebyshev tails
    return (o['residual'] < RESIDUAL_MAX and o['constraint'] < CONSTRAINT_MAX and max(o['tails']) < 1e-5)


def resample(coeffs, e):
    return np.array([CH.chebval(1-2*e.x, c) for c in coeffs]).ravel()


def main(alpha_max=10.0, n_low=48, n_high=72, constraint_max=None, reset_n=False, residual_max=None):
    """Adaptive: the high-resolution solve is seeded from the accepted
    low-resolution solution (continuing a high-resolution solution on its own
    can trap it in a noisy root), and when the constraint fails with clean
    tails both resolutions are raised (N_low += 16, N_high += 24) instead of
    the step being cut. Each row records the resolutions it used."""
    global CONSTRAINT_MAX, RESIDUAL_MAX
    if residual_max is not None:
        RESIDUAL_MAX = residual_max
    if constraint_max is not None:
        CONSTRAINT_MAX = constraint_max
    if OUT.exists():
        data = json.loads(OUT.read_text())
    else:
        data = dict(n_low=n_low, n_high=n_high, rows=[])
    if data['rows']:
        last = data['rows'][-1]
        if not reset_n:
            n_low, n_high = last.get('n_low', data['n_low']), last.get('n_high', data['n_high'])
        e_lo, e_hi = ExtremalRQ(n_low), ExtremalRQ(n_high)
        alpha, step = last['alpha'], last['step']
        flat_lo = resample(last['coeffs_low'], e_lo)
        flat_hi = resample(last['coeffs_high'], e_hi)
        print(f'resuming at alpha = {alpha} with N = {n_low}/{n_high}')
    else:
        e_lo, e_hi = ExtremalRQ(n_low), ExtremalRQ(n_high)
        alpha, step = 0., 2e-3
        flat_lo, flat_hi = myers_perry_q(n_low), myers_perry_q(n_high)
        lo = record(e_lo, flat_lo, 0., dict(residual=0., iterations=0))
        hi = record(e_hi, flat_hi, 0., dict(residual=0., iterations=0))
        data['rows'].append(dict(alpha=0., step=step, low=lo, high=hi, n_low=n_low, n_high=n_high,
                                 coeffs_low=[c.tolist() for c in e_lo.coefficients(flat_lo)],
                                 coeffs_high=[c.tolist() for c in e_hi.coefficients(flat_hi)]))
    # tangent predictor q + da dq/dalpha for both resolutions: without it,
    # Newton from the previous solution falls into spurious roots once
    # alpha ~ 1 (diag_escalate.py)
    t_lo = tangent(e_lo, flat_lo, alpha) if alpha > 0 else np.zeros_like(flat_lo)
    t_hi = tangent(e_hi, flat_hi, alpha) if alpha > 0 else np.zeros_like(flat_hi)
    while alpha < alpha_max-1e-12:
        target = min(alpha+step, alpha_max)
        da = target-alpha
        sol_lo, lo = robust_solve(e_lo, target, flat_lo+da*t_lo)
        if not acceptable(lo):
            if lo['constraint'] >= CONSTRAINT_MAX and max(lo['tails']) < 1e-6 and lo['residual'] < 5e-8 and n_low < 200:
                coeffs = e_lo.coefficients(flat_lo)
                n_low, n_high = n_low+16, n_high+24
                e_lo, e_hi = ExtremalRQ(n_low), ExtremalRQ(n_high)
                flat_lo = resample(coeffs, e_lo)
                flat_lo, _ = robust_solve(e_lo, alpha, flat_lo)
                flat_hi = resample(e_lo.coefficients(flat_lo), e_hi)
                flat_hi, _ = robust_solve(e_hi, alpha, flat_hi)
                t_lo, t_hi = tangent(e_lo, flat_lo, alpha), tangent(e_hi, flat_hi, alpha)
                print(f'   resolution raised to N = {n_low}/{n_high} at alpha = {alpha:.6f}')
                continue
            step /= 2
            print(f'   step rejected (low) at alpha = {target:.6f}; step -> {step:.2e}')
            if step < 1e-6:
                print(f'stopped: no acceptable solution beyond alpha = {alpha:.6f}')
                break
            continue
        sol_hi, hi = robust_solve(e_hi, target, flat_hi+da*t_hi)
        if not acceptable(hi):           # second chance: seed from the low solution
            sol_hi, hi = robust_solve(e_hi, target, resample(e_lo.coefficients(sol_lo), e_hi))
        if not acceptable(hi):
            step /= 2
            print(f'   step rejected (high) at alpha = {target:.6f}: C {hi["constraint"]:.0e} '
                  f'tails {max(hi["tails"]):.0e}; step -> {step:.2e}')
            if step < 1e-6:
                print(f'stopped: no acceptable solution beyond alpha = {alpha:.6f}')
                break
            continue
        flat_lo, flat_hi, alpha = sol_lo, sol_hi, target
        t_lo, t_hi = tangent(e_lo, flat_lo, alpha), tangent(e_hi, flat_hi, alpha)
        data['rows'].append(dict(alpha=alpha, step=step, low=lo, high=hi, n_low=n_low, n_high=n_high,
                                 constraint_max=CONSTRAINT_MAX, residual_max=RESIDUAL_MAX,
                                 coeffs_low=[c.tolist() for c in e_lo.coefficients(sol_lo)],
                                 coeffs_high=[c.tolist() for c in e_hi.coefficients(sol_hi)]))
        OUT.write_text(json.dumps(data))
        print(f"alpha={alpha:.5f} N={n_low}/{n_high} y={hi['y']:.7f} x={hi['x']:.5f} j={hi['j']:.6f} "
              f"mu={hi['mu']:.10f} (dN {abs(hi['mu']-lo['mu']):.0e}) mu'={hi['mu_prime']:.7f} "
              f"(dN {abs(hi['mu_prime']-lo['mu_prime']):.0e}) Psi_S={hi['psi_smarr']:.7f} "
              f"Psi_FL={hi['psi_first_law']:.7f} C={hi['constraint']:.0e} tail={max(hi['tails']):.0e}", flush=True)
        step = min(step*1.4, 0.02*max(alpha, 0.4))     # relative steps at large alpha


if __name__ == '__main__':
    # python branch_rq.py alpha_max n_low n_high [constraint_max reset]
    args = sys.argv[1:]
    kw = {}
    if len(args) > 3:
        kw = dict(constraint_max=float(args[3]), reset_n=len(args) > 4 and args[4] == 'reset',
                  residual_max=float(args[5]) if len(args) > 5 else None)
    main(*([float(args[0])] if args else []), *(int(a) for a in args[1:3]), **kw)
