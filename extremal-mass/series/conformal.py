"""Resummation test of the conjecture that the nearest singularities of m(s) are s*, conj(s*).
Map: t = (s - x*)/y*,  w = t/(1 + sqrt(1+t^2))  sends the s-plane cut along Re s = x*, |Im s| >= y*
onto |w| < 1.  The exact series d_1..d_6 is re-expanded in (w - w0), w0 = w(0), and evaluated beyond
|s*| = 0.0551; compared with the spectral branch (work/kkr-extremal/analysis.json)."""
import json, math
from pathlib import Path
import mpmath as mp
import numpy as np
mp.mp.dps = 40
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = json.loads((HERE/'coefficients.json').read_text())
d = [mp.mpf('1.5')]+[mp.mpf(1)]+[mp.mpf(C[str(n)]['value']) for n in range(2, 7)]
xs, ys = mp.mpf('0.0059231771'), mp.mpf('0.0548188529')
s_of_w = lambda w: xs+ys*2*w/(1-w**2)
w_of_s = lambda s: (lambda t: t/(1+mp.sqrt(1+t**2)))((s-xs)/ys)
w0 = w_of_s(0)
N = len(d)-1
# Taylor coefficients of s(w0 + v) - s(w0) in v, then compose with the s-series
sv = mp.taylor(lambda v: s_of_w(w0+v), 0, N)          # sv[0] = 0
def compose(order):
    # coefficients c_k of m(s(w0+v)) = sum_n d_n (sum_j sv_j v^j)^n, truncated at v^order
    c = [mp.mpf(0)]*(order+1)
    powr = [mp.mpf(1)]+[mp.mpf(0)]*order
    for n in range(0, N+1):
        for k in range(order+1):
            c[k] += d[n]*powr[k]
        newp = [mp.mpf(0)]*(order+1)
        for i in range(order+1):
            if powr[i] == 0: continue
            for j in range(1, order+1-i):
                newp[i+j] += powr[i]*sv[j]
        powr = newp
    return c
cw = compose(N)
rows = json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch']
num = []
for r in rows:
    a, M, J = r['alpha'], r['M'], r['J']
    if a <= 0: continue
    num.append((math.pi**(2/3)*a/J**(2/3), M/(math.pi**(1/3)*J**(2/3))))
num = np.array(sorted(num))
print('w0 =', mp.nstr(w0, 8))
print(' s       numeric       plain(6)      conformal K=3..6')
for s in [0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5]:
    mnum = np.interp(s, num[:, 0], num[:, 1])
    plain = sum(d[n]*mp.mpf(s)**n for n in range(N+1))
    v = w_of_s(mp.mpf(s))-w0
    conf = [sum(cw[k]*v**k for k in range(K+1)) for K in range(3, N+1)]
    print(f'{s:5.2f}  {mnum:.8f}  {mp.nstr(plain, 9):>12s}  ' + '  '.join(mp.nstr(c, 9) for c in conf), f'  |v|={mp.nstr(abs(v), 3)}')
