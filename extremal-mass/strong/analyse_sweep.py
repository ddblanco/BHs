"""Strong-coupling analysis of the two-sided sweep (sweep.json):
m(s) = 3s/4 + a1 s^-1/2 + a2 s^-1 + a3 s^-3/2 + ...   (sigma = s^-1/2)
1. free fits of (a1, a2, ...) on s >= s_min, several truncations;
2. fits with a1 = 4/3 fixed (BTZ universality, btz_universality.py);
3. comparison with the conditional a2 = 0.456526735220 of reports/strong-coupling-endpoint.pdf."""
import json, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
data = sorted(json.loads((HERE/'sweep.json').read_text()), key=lambda d: d['s'])
s = np.array([d['s'] for d in data]); m = np.array([d['m'] for d in data])
err = np.array([max([abs(v) for v in d.get('dm_variants', [0])]+[1e-12]) for d in data])
print(f'{len(s)} points, s in [{s.min():.3g}, {s.max():.3g}]')
for d in data:
    print(f"  s={d['s']:10.5g} m={d['m']:.12f}  (m-3s/4)*sqrt(s)={(d['m']-0.75*d['s'])*np.sqrt(d['s']):.10f}  variants {['%.1e' % v for v in d.get('dm_variants', [])]}")
sig = s**-0.5
y = m-0.75*s
out = {}
for smin in sorted({v for v in (10, 20, 40, 80, 150) if (s >= v).sum() >= 6}):
    sel = s >= smin
    for K in (3, 4, 5, 6):
        if sel.sum() <= K: continue
        A = np.array([sig[sel]**k for k in range(1, K+1)]).T
        c = np.linalg.lstsq(A, y[sel], rcond=None)[0]
        print(f's>={smin} K={K}: a = {np.array2string(c, precision=7)}')
        out[f'free_{smin}_{K}'] = c.tolist()
    for K in (2, 3, 4, 5):
        if sel.sum() <= K: continue
        A = np.array([sig[sel]**k for k in range(2, K+2)]).T
        c = np.linalg.lstsq(A, y[sel]-4/3*sig[sel], rcond=None)[0]
        print(f'   a1=4/3: s>={smin} K={K}: a2.. = {np.array2string(c, precision=7)}')
        out[f'fixed_{smin}_{K}'] = c.tolist()
(HERE/'analyse_sweep.json').write_text(json.dumps(out, indent=1))
print('conditional prediction a2 = 0.456526735220')
