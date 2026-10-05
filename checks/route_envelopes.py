"""Psi and mu = M/J^(2/3) extrapolation envelopes on the same 20-fit list.

The manuscript's Psi envelope (manuscript/reanalysis.py:sensitivity, applied in
manuscript/make_figures.py) varies coordinate (T, tau), window (6, up to 12
coldest) and model (degree 1-3, sqrt, v^2 log v). The stored mu_spread uses only
the three original degree checks. This script gives mu the same list, pairing M
and J fit by fit, and compares the two routes per interval and cumulatively.
Run from the repository root: python3 checks/route_envelopes.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'manuscript'))
from reanalysis import WINDOW, _cold, _design, sensitivity  # noqa: E402


def intercepts(states, field):
    cold, out = _cold(states), {}
    values = np.array([s[field] for s in cold])
    for name, key in (('T', 'T_H'), ('tau', 'tau')):
        raw = np.array([s[key] for s in cold])
        for requested in (WINDOW, 12):
            n = min(requested, len(cold))
            x = raw[:n]/raw[n-1]
            for model in (1, 2, 3, 'sqrt', 'log'):
                c, *_ = np.linalg.lstsq(_design(x, model), values[:n], rcond=None)
                out[f'{name}_{model}_{n}'] = c[0]
    return out


data = json.loads((ROOT/'results/egb-extremality.json').read_text())
extremals = {r['alpha_gb']: r for r in data['extremals']}
rows = []
for key, states in data['walks'].items():
    r = extremals[float(key)]
    psi = max(r['psi_gb']['spread'], sensitivity(states, r['psi_gb']['value'])['envelope'])
    # Same list as the manuscript's Psi envelope, reproduced here for mu.
    assert len(intercepts(states, 'psi_gb')) == len(sensitivity(states, 0.)['intercepts'])
    M, J = intercepts(states, 'E'), intercepts(states, 'J')
    mu = max(r['mu_spread'], max(abs(M[k]/J[k]**(2/3)-r['mu']) for k in M))
    rows.append((r['y'], psi, r['mu_spread'], mu))
rows.sort()
y, psi, mu_deg, mu = (np.array(c) for c in zip(*rows))

print('y        psi_env   mu_spread(degrees)  mu_env(20 fits)')
for row in rows:
    print('{:.4f}  {:.2e}  {:.2e}            {:.2e}'.format(*row))
print('\ninterval         psi_env  secant_env  ratio')
for i in range(len(y)-1):
    gap = y[i+1]-y[i]
    p, s = (psi[i]+psi[i+1])/2, (mu[i]+mu[i+1])/gap
    print(f'{y[i]:.4f}-{y[i+1]:.4f}  {p:.2e} {s:.2e}   {p/s:.2g}')
print('\ny        cum_intPsi  mu_env   ratio')
total = 0.
for i in range(1, len(y)):
    total += (y[i]-y[i-1])*(psi[i]+psi[i-1])/2
    print(f'{y[i]:.4f}  {total:.2e}    {mu[i]:.2e} {total/mu[i]:.2g}')
