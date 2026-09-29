"""Independent check of stored profiles against the full EGB equations and charges.

For every profile file in out/: evaluate E_ab=G_ab+alpha H_ab (egb_tensor.py, derived
here from the metric) at 12 off-grid radii and two angles, relative to the largest of
|G^a_b|, |alpha H^a_b| and sqrt|Kretschmann| at that point (mixed components); and re-extract E,J,T,S without tail fits.
"""
import json
import sys
from pathlib import Path

import numpy as np

from egb_tensor import field_equations
from profiles import Profile

XS = np.array([0.013, 0.05, 0.11, 0.2, 0.31, 0.43, 0.52, 0.64, 0.77, 0.86, 0.93, 0.97])


def residual(prof, alpha):
    worst = 0.
    for x in XS:
        rv = 1/(1-x)
        vals = {k: float(v) for k, v in prof.radial(rv).items()}
        for thv in (0.37, 1.05):
            E, G, aH, K, L, g = field_equations(rv, thv, vals, alpha, metric=True)
            gi = np.linalg.inv(g)
            mixed = [gi@X for X in (E, G, aH)]   # E^a_b: coordinate-invariant scale
            scale = max(np.max(np.abs(mixed[1])), np.max(np.abs(mixed[2])), np.sqrt(abs(K)))
            worst = max(worst, np.max(np.abs(mixed[0]))/scale)
    return worst


if __name__ == '__main__':
    summary = {}
    for path in sys.argv[1:]:
        d = json.loads(Path(path).read_text())
        if 'couplings' in d:          # profiles-omega0.33.json
            items = [(float(a), d['omega_h'], v['coefficients'], v['state'])
                     for a, v in d['couplings'].items()]
        else:                         # walk-*.json
            rows = {f"{r['omega_h']:.12f}": r for r in d['rows'] if 'mine' in r}
            items = [(d['alpha'], float(k), c, rows[k]['stored'])
                     for k, c in d['profiles'].items()]
        for alpha, omega, coeffs, ref in items:
            prof = Profile(coeffs)
            res = residual(prof, alpha)
            obs = prof.observables(alpha, omega)
            rel = {k: abs(obs[k]-ref[k])/abs(ref[k]) for k in ('E', 'J', 'T_H', 'S')}
            assert abs(obs['Omega_H']-omega) < 1e-9, (obs['Omega_H'], omega)
            summary.setdefault(alpha, []).append((omega, res, rel))
    print(f"{'alpha':>6} {'states':>6} {'max EGB rel.res.':>17} {'max|dE|/E':>10} {'max|dJ|/J':>10} {'max|dT|/T':>10} {'max|dS|/S':>10}")
    for alpha in sorted(summary):
        rows = summary[alpha]
        print(f"{alpha:6g} {len(rows):6d} {max(r[1] for r in rows):17.2e} "
              + ' '.join(f"{max(r[2][k] for r in rows):10.2e}" for k in ('E', 'J', 'T_H', 'S')))
    worst = max(r[1] for rows in summary.values() for r in rows)
    print(f'worst relative EGB residual over all profiles: {worst:.2e}')
