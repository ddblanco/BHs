"""Gate and resolution study for extremal_r_solver.py.

Asserted at alpha = 0: residual at extremal Myers-Perry at round-off level,
chain-rule Jacobian = dense complex-step Jacobian (row-relative), Newton from
a smooth perturbation returns Myers-Perry, M = 3 pi/4, J = pi sqrt(2)/4 (both
from p_w and from the r^-4 tail), Omega_H = 1/sqrt 2, S = 2 pi J, j = 1,
a_H = s = 1/2.
Then the resolution study at three couplings (not asserted; printed and saved).

Run: python test_r.py  ->  convergence_r.json
"""
import json
from pathlib import Path

import numpy as np

from extremal_r_solver import ExtremalR, invariants, myers_perry

HERE = Path(__file__).resolve().parent


def gate():
    for n in (24, 40):
        e = ExtremalR(n)
        mp = myers_perry(n)
        E = e.raw(mp.reshape(4, n), 0.)
        assert np.max(np.abs(E[:, 1:-1])) < 1e-8, np.max(np.abs(E))
        p = mp+0.01*np.sin(np.arange(4*n))*np.tile(e.x*(1-e.x), 4)
        A, B = e.jacobian(p, 0.), e.dense_jacobian(p, 0.)
        err = np.max(np.abs(A-B)/np.maximum(np.abs(B).max(axis=1, keepdims=True), 1e-300))
        assert err < 1e-9, err
        print(f'N={n}: residual at MP {np.max(np.abs(E[:, 1:-1])):.1e}; Jacobian check {err:.1e}')
    n = 40
    e = ExtremalR(n)
    x = np.tile(e.x, 4)
    guess = myers_perry(n)+1e-3*np.sin(2*x+np.repeat(np.arange(4), n))*x*(1-x)
    sol, info = e.solve(0., guess)
    o = e.observables(sol, 0.)
    inv = invariants(o)
    dev = np.max(np.abs(sol-myers_perry(n)))
    print('Newton from perturbed MP:', info, f'deviation {dev:.1e}, cond {e.condition(sol, 0.):.1e}')
    print(f"  M {o['M']:.15f} vs {3*np.pi/4:.15f}; J {o['J']:.15f} J_inf {o['J_inf']:.15f} vs {np.pi*np.sqrt(2)/4:.15f}")
    assert dev < 1e-10
    # M needs P1''(1): round-off floor ~N^4 eps; J comes from p_w (no derivative at infinity)
    assert abs(o['M']-3*np.pi/4) < 1e-7 and abs(o['J']-np.pi*np.sqrt(2)/4) < 1e-10
    assert abs(o['Omega_H']-1/np.sqrt(2)) < 1e-9 and abs(o['S']-2*np.pi*o['J']) < 1e-9
    assert abs(inv['j']-1) < 1e-7 and abs(inv['aH']-.5) < 1e-7 and abs(inv['s']-.5) < 1e-7
    print('alpha = 0 gate: OK')
    return sol


def study(start):
    report = []
    for target in (0.05, 0.2, 0.5):
        e = ExtremalR(40)
        flat = start
        for a in np.linspace(0, target, 41)[1:]:
            flat, info = e.solve(a, flat)
            assert info['residual'] < 1e-10, (a, info)
        rows = []
        for n in (32, 48, 64, 96, 128, 160):
            flat = e.interpolate(flat, n)
            e = ExtremalR(n)
            flat, info = e.solve(target, flat)
            o = e.observables(flat, target)
            inv = invariants(o)
            rows.append(dict(n=n, **info, **o, **inv))
            print(f"alpha={target} N={n:3d} res {info['residual']:.0e} M {o['M']:.13f} "
                  f"J {o['J']:.13f} (inf {o['J_inf']:.9f}, spread {o['J_spread']:.0e}) "
                  f"mu {inv['mu']:.12f} y {inv['y']:.12f} C {o['constraint']:.0e} "
                  f"P1'(1) {o['P1_slope_inf']:.0e} rows {o['replaced_rows'][0]:.0e},{o['replaced_rows'][1]:.0e}")
        report.append(dict(alpha=target, rows=rows))
    (HERE/'convergence_r.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    start = gate()
    study(start)
