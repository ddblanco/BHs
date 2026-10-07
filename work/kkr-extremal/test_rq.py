"""Gate and resolution study for the production solver extremal_rq.py.

Asserted at alpha = 0: residual at extremal Myers-Perry (Q = 0) at round-off;
chain-rule Jacobian = dense complex-step Jacobian; Newton from a smooth
perturbation returns Q = 0; M = 3 pi/4, J = pi sqrt(2)/4, Omega_H = 1/sqrt 2,
S = 2 pi J, j = 1, a_H = s = 1/2.
Resolution study at alpha = 0.05, 0.2, 0.5 (r_H = 1), printed next to the
manuscript's T -> 0 extrapolation at the same coupling (not asserted).

Run: python test_rq.py  ->  convergence_rq.json
"""
import json
import warnings
from pathlib import Path

import numpy as np

from extremal_rq import ExtremalRQ, myers_perry_q, invariants

warnings.simplefilter('ignore', RuntimeWarning)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def gate():
    for n in (24, 40):
        e = ExtremalRQ(n)
        q0 = myers_perry_q(n)
        E = e.rows(q0.reshape(4, n), 0.)
        assert np.max(np.abs(E)) < 1e-8, np.max(np.abs(E))
        p = q0+1e-2*np.sin(np.arange(4*n))
        A, B = e.jacobian(p, 0.), e.dense_jacobian(p, 0.)
        err = np.max(np.abs(A-B)/np.maximum(np.abs(B).max(axis=1, keepdims=True), 1e-300))
        assert err < 1e-10, err
        print(f'N={n}: residual at MP {np.max(np.abs(E)):.1e}; Jacobian check {err:.1e}')
    n = 48
    e = ExtremalRQ(n)
    x = np.tile(e.x, 4)
    # a larger perturbation can reach a spurious aliasing root of the
    # discrete system (non-decaying Chebyshev tail, constraint ~1e-3)
    guess = 1e-4*np.sin(2*x+np.repeat(np.arange(4), n))
    sol, info = e.solve(0., guess)
    o = e.observables(sol, 0.)
    inv = invariants(o)
    print('Newton from perturbed MP:', info, f'max |Q| {np.max(np.abs(sol)):.1e}')
    print(f"  M {o['M']:.15f} vs {3*np.pi/4:.15f}; J {o['J']:.15f} vs {np.pi*np.sqrt(2)/4:.15f}")
    assert np.max(np.abs(sol)) < 1e-7
    # the residual floor (~4e-10) bounds |Q| ~ 3e-8 here; tolerances follow it
    assert abs(o['M']-3*np.pi/4) < 1e-7 and abs(o['J']-np.pi*np.sqrt(2)/4) < 1e-7
    assert abs(o['Omega_H']-1/np.sqrt(2)) < 1e-7 and abs(o['S']-2*np.pi*o['J']) < 1e-7
    assert abs(inv['j']-1) < 1e-7 and abs(inv['aH']-.5) < 1e-7 and abs(inv['s']-.5) < 1e-7
    print('alpha = 0 gate: OK')


def manuscript():
    d = json.loads((ROOT/'results/egb-extremality.json').read_text())
    return {r['alpha_gb']: r for r in d['extremals']}


def study():
    ref = manuscript()
    report = []
    for target in (0.05, 0.2, 0.5):
        e = ExtremalRQ(48)
        flat = myers_perry_q(48)
        for a in np.linspace(0, target, 41)[1:]:
            flat, info = e.solve(a, flat)
            assert info['residual'] < 1e-8, (a, info)   # floor ~1e-10 from rows next to infinity
        rows = []
        m = ref[target]
        print(f"alpha={target}: manuscript (T->0 extrapolation) y={m['y']:.9f} mu={m['mu']:.9f} "
              f"(+-{m['mu_spread']:.0e}) sigma={m['sigma']:.7f}")
        for n in (32, 48, 64, 96, 128, 160):
            flat = e.interpolate(flat, n)
            e = ExtremalRQ(n)
            flat, info = e.solve(target, flat)
            o = e.observables(flat, target)
            inv = invariants(o)
            rows.append(dict(n=n, **info, **o, **inv))
            print(f"   N={n:3d} res {info['residual']:.0e} M {o['M']:.13f} J {o['J']:.13f} "
                  f"y {inv['y']:.12f} mu {inv['mu']:.12f} sigma {inv['sigma']:.10f} "
                  f"C {o['constraint']:.0e} spreadJ {o['J_spread']:.0e} row {o['replaced_rows'][0]:.0e}")
        report.append(dict(alpha=target, rows=rows, manuscript=dict(y=m['y'], mu=m['mu'],
                                                                     mu_spread=m['mu_spread'],
                                                                     sigma=m['sigma'])))
    (HERE/'convergence_rq.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    gate()
    study()
