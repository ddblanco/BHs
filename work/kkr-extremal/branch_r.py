"""The extremal branch, constructed directly at T = 0, r_H = g_H = 1.

For each coupling alpha (units r_H = 1, the manuscript's convention) the
extremal solution is solved at two resolutions; observables, invariants and
the slope are recorded together with their resolution differences.

Slope by implicit differentiation of the discrete system (no finite
difference in the coupling): A du/dalpha = -dR/dalpha, then
    mu'(y) = (M_a - (2/3) M J_a/J) / (1 - (2/3) alpha J_a/J),
which by scale invariance is dM_ext/dalpha|_J. Two independent identities are
evaluated, not imposed:
    first law at T = 0 along the branch:  Psi_FL = M_a - 2 Omega_H J_a ,
    Smarr at T = 0:                      Psi_S  = (M - 3 Omega_H J)/alpha ;
mu' = Psi_FL holds iff Smarr holds (algebra in the write-up).

Resumable: results are appended to branch_r.json after every coupling, with
the Chebyshev coefficients of the solution, and a rerun continues from the
last recorded coupling.

Run: python branch_r.py [alpha_max] [N_low] [N_high]
"""
import json
import sys
from pathlib import Path

import numpy as np

from extremal_r_solver import ExtremalR, invariants, myers_perry, cheb
from _pw_generated_p0 import J_of

HERE = Path(__file__).resolve().parent
OUT = HERE/'branch_r.json'
H = 1e-30


def slope(e, flat, alpha):
    """Implicit derivatives of M, J along the branch at fixed r_H = 1."""
    n = e.n
    A = e.jacobian(flat, alpha)
    dR = e.residual(flat.astype(complex), alpha+1j*H).imag/H
    du = np.linalg.solve(A, -dR)
    u = flat.reshape(4, n)
    dP1 = du.reshape(4, n)[0]
    M_a = -3*np.pi/4*(e.D2@dP1)[-1]/4
    m = n//2                                   # J and J_a at one fixed node
    uc = (flat+1j*H*du).reshape(4, n)
    D = e.D
    Jc = J_of(e.x[m], uc[0, m], (D@uc[0])[m], uc[1, m], (D@uc[1])[m],
              uc[2, m], (D@uc[2])[m], uc[3, m], (D@uc[3])[m], alpha+1j*H)
    J_m = J_of(e.x[m], u[0, m], (D@u[0])[m], u[1, m], (D@u[1])[m],
               u[2, m], (D@u[2])[m], u[3, m], (D@u[3])[m], alpha)
    J_a = np.imag(Jc)/H
    Omega_a = du.reshape(4, n)[3, 0]
    return dict(M_a=float(M_a), J_a=float(J_a), J_mid=float(np.real(J_m)), Omega_a=float(Omega_a))


def record(e, flat, alpha):
    o = e.observables(flat, alpha)
    inv = invariants(o)
    d = slope(e, flat, alpha)
    M, J, Om = o['M'], o['J'], o['Omega_H']
    mu_prime = ((d['M_a']-2/3*M*d['J_a']/J)/(1-2/3*alpha*d['J_a']/J))
    psi_fl = d['M_a']-2*Om*d['J_a']
    psi_smarr = (M-3*Om*J)/alpha if alpha > 0 else float('nan')
    return dict(o, **inv, **d, mu_prime=float(mu_prime), psi_first_law=float(psi_fl),
                psi_smarr=float(psi_smarr))


def main(alpha_max=2.0, n_low=96, n_high=128):
    data = json.loads(OUT.read_text()) if OUT.exists() else dict(n_low=n_low, n_high=n_high, rows=[])
    assert data['n_low'] == n_low and data['n_high'] == n_high, 'resolutions differ from the saved run'
    e_lo, e_hi = ExtremalR(n_low), ExtremalR(n_high)
    if data['rows']:
        last = data['rows'][-1]
        alpha = last['alpha']
        flat_lo = np.array([np.polynomial.chebyshev.chebval(1-2*e_lo.x, c) for c in last['coeffs_low']]).ravel()
        flat_hi = np.array([np.polynomial.chebyshev.chebval(1-2*e_hi.x, c) for c in last['coeffs_high']]).ravel()
        step = last['step']
    else:
        alpha, step = 0., 2e-3
        flat_lo, flat_hi = myers_perry(n_low), myers_perry(n_high)
        rows = []
        for e, f in ((e_lo, flat_lo), (e_hi, flat_hi)):
            f, info = e.solve(0., f)
        r_lo = dict(e_lo.observables(flat_lo, 0.), **invariants(e_lo.observables(flat_lo, 0.)))
        data['rows'].append(dict(alpha=0., step=step, low=r_lo, high=r_lo,
                                 coeffs_low=[c.tolist() for c in e_lo.coefficients(flat_lo)],
                                 coeffs_high=[c.tolist() for c in e_hi.coefficients(flat_hi)]))
    while alpha < alpha_max-1e-12:
        target = min(alpha+step, alpha_max)
        ok = True
        new = []
        for e, f in ((e_lo, flat_lo), (e_hi, flat_hi)):
            sol, info = e.solve(target, f)
            if info['residual'] > 1e-10:
                ok = False
                break
            new.append(sol)
        if not ok:
            step /= 2
            if step < 1e-6:
                print(f'stopped: no convergence beyond alpha = {alpha:.6f}')
                break
            continue
        flat_lo, flat_hi = new
        alpha = target
        lo, hi = record(e_lo, flat_lo, alpha), record(e_hi, flat_hi, alpha)
        row = dict(alpha=alpha, step=step, low=lo, high=hi,
                   coeffs_low=[c.tolist() for c in e_lo.coefficients(flat_lo)],
                   coeffs_high=[c.tolist() for c in e_hi.coefficients(flat_hi)])
        data['rows'].append(row)
        OUT.write_text(json.dumps(data))
        print(f"alpha={alpha:.5f} y={hi['y']:.6f} x={hi['x']:.5f} j={hi['j']:.6f} "
              f"mu={hi['mu']:.10f} (d {abs(hi['mu']-lo['mu']):.0e}) mu'={hi['mu_prime']:.7f} "
              f"(d {abs(hi['mu_prime']-lo['mu_prime']):.0e}) Psi_S={hi['psi_smarr']:.7f} "
              f"Psi_FL={hi['psi_first_law']:.7f} C={hi['constraint']:.0e}")
        step = min(step*1.5, 0.05)


if __name__ == '__main__':
    args = [float(a) for a in sys.argv[1:]]
    main(*(args[:1]), *(int(a) for a in args[1:]))
