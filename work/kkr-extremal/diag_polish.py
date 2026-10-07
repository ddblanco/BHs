import json, numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ
from branch_rq import record
CH = np.polynomial.chebyshev
d = json.loads(open('branch_rq.json').read())
row = max(d['rows'], key=lambda r: r['high']['constraint'])
a = row['alpha']
print('noisiest row alpha', a, 'C', row['high']['constraint'], 'tails', row['high']['tails'])
e = ExtremalRQ(72)
f = np.array([CH.chebval(1-2*e.x, c) for c in row['coeffs_high']]).ravel()
for k in range(3):
    f, info = e.solve(a, f)
    r = record(e, f, a, info)
    print(f'polish {k}: res {info["residual"]:.1e} C {r["constraint"]:.1e} tails {max(r["tails"]):.1e} mu {r["mu"]:.10f}')
# low-pass: refit with the top quarter of coefficients removed, then re-solve
c = e.coefficients(f)
f2 = np.array([CH.chebval(1-2*e.x, ci[:54]) for ci in c]).ravel()
f2, info = e.solve(a, f2)
r = record(e, f2, a, info)
print(f'filtered start: res {info["residual"]:.1e} C {r["constraint"]:.1e} tails {max(r["tails"]):.1e} mu {r["mu"]:.10f}; low-res mu {row["low"]["mu"]:.10f}')
