"""Resolution study of the KKR-gauge extremal solver.

At a few couplings, solve at N = 24 ... 96 (each seeded by interpolating the
previous resolution) and report M, J, mu, y, the constraint E_f and the decay
of the Chebyshev coefficients of P1, P2, P3, W. The near-horizon analysis
(nearhorizon.py) predicts a term rho^gamma ~ xi^gamma with non-integer gamma,
so the coefficients should decay algebraically, roughly like n^(-2 gamma - 1).

Run: python convergence_kkr.py  ->  convergence_kkr.json
"""
import json
from pathlib import Path

import numpy as np

from kkr_solver import KKR, invariants, myers_perry

HERE = Path(__file__).resolve().parent


def walk_to(alpha_target, n=40, steps=None):
    k = KKR(n)
    flat = myers_perry(n)
    alphas = steps if steps is not None else np.linspace(0, alpha_target, 9)[1:]
    for a in alphas:
        flat, info = k.solve(a, flat)
        assert info['residual'] < 1e-9, (a, info)
    return k, flat


def main():
    report = []
    for alpha in (0.04, 0.2, 0.6):
        k, flat = walk_to(alpha, steps=np.geomspace(1e-3, alpha, 14))
        rows = []
        for n in (24, 32, 40, 48, 64, 80, 96, 128):
            guess = k.interpolate(flat, n)
            k = KKR(n)
            flat, info = k.solve(alpha, guess)
            o = k.observables(flat, alpha)
            inv = invariants(o)
            coeffs = k.coefficients(flat)
            tail = [float(np.max(np.abs(c[-6:]))) for c in coeffs]
            rows.append(dict(n=n, residual=info['residual'], condition=info['condition'],
                             M=o['M'], J=o['J'], Omega_H=o['Omega_H'], S=o['S'],
                             constraint=o['constraint'], tail=tail, **inv))
            print(f"alpha={alpha} N={n:3d}: res {info['residual']:.1e} cond {info['condition']:.1e} "
                  f"M {o['M']:.13f} J {o['J']:.13f} mu {inv['mu']:.13f} C {o['constraint']:.1e} "
                  f"tails {np.array(tail)}")
        report.append(dict(alpha=alpha, rows=rows))
    (HERE/'convergence_kkr.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
