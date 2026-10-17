"""Fit the strong-coupling expansion m(s) = 3s/4 + sum_k a_k sigma^k, sigma = s^(-1/2),
to the numerical extremal branch of work/kkr-extremal (analysis.json), using M and,
independently, Omega_H:  Omega_H sqrt(alpha) = (1/3) sum_k a_k (1+k/2) sigma^(k-1)."""
import json, math, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch']
S, m, om = [], [], []
for r in rows:
    a, M, J = r['alpha'], r['M'], r['J']
    if a <= 0: continue
    S.append(math.pi**(2/3)*a/J**(2/3)); m.append(M/(math.pi**(1/3)*J**(2/3))); om.append(r['Omega_H']*math.sqrt(a))
S, m, om = map(np.array, (S, m, om))
sig = S**-0.5
out = {}
for lo in (2.0, 2.5, 3.0):
    sel = S >= lo
    for K in (3, 4, 5):
        ks = range(1, K+1)
        A = np.vstack([np.array([sig[sel]**k for k in ks]).T,
                       np.array([(1+k/2)/3*sig[sel]**(k-1) for k in ks]).T])
        y = np.concatenate([m[sel]-0.75*S[sel], om[sel]])
        co = np.linalg.lstsq(A, y, rcond=None)[0]
        rms = float(np.sqrt(np.mean((A@co-y)**2)))
        print(f's>={lo} K={K}: a_k = {np.round(co,4)}  rms {rms:.1e}  (n={sel.sum()})')
        out[f'{lo}_{K}'] = dict(a=co.tolist(), rms=rms)
    # constrained: a1 = 4/3
    for K in (3, 4, 5):
        ks = range(2, K+1)
        A = np.vstack([np.array([sig[sel]**k for k in ks]).T,
                       np.array([(1+k/2)/3*sig[sel]**(k-1) for k in ks]).T])
        y = np.concatenate([m[sel]-0.75*S[sel]-4/3*sig[sel], om[sel]-2/3])
        co = np.linalg.lstsq(A, y, rcond=None)[0]
        print(f'   a1=4/3 fixed, K={K}: a_2.. = {np.round(co,4)} rms {np.sqrt(np.mean((A@co-y)**2)):.1e}')
print('max s on branch:', S.max(), ' sigma:', sig.min())
Path('strong_fit.json').write_text(json.dumps(out, indent=1))
