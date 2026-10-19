"""Closed-form approximant for m(s) on 0 <= s < infinity, built on the exact ends.

Ansatz:  m(s) = 3s/4 + N(s)/sqrt(D(s)),  N = sum_{k<=n} c_k s^k,  D = 1 + sum_{k=1}^{2n+1} e_k s^k.
- s -> infinity: N/sqrt(D) -> (c_n/sqrt(e_{2n+1})) s^-1/2; we impose c_n^2 = (4/3)^2 e_{2n+1}  (a1 = 4/3).
- s -> 0: Taylor coefficients imposed exactly up to s^p (p <= 6) from series/coefficients.json.
Remaining parameters: least squares on the numerical m(s) (spectral branch + two-sided sweep),
in relative error.  Reports max relative error on all data and the implied strong-coupling a2."""
import json, sys
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
dser = json.loads((HERE.parent/'series/coefficients.json').read_text())
d = [1.5, 1.0]+[float(dser[str(k)]['value']) for k in range(2, 7)]


def data():
    pts = []
    for r in json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch']:
        a, M, J = r['alpha'], r['M'], r['J']
        if a > 0:
            pts.append((np.pi**(2/3)*a/J**(2/3), M/(np.pi**(1/3)*J**(2/3)), 'spectral'))
    sw = HERE/'sweep.json'
    if sw.exists():
        for r in json.loads(sw.read_text()):
            pts.append((r['s'], r['m'], 'twoside'))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), [p[2] for p in pts]


def model(params, s, n):
    c = params[:n+1]; e = np.r_[1., params[n+1:]]
    N = np.polyval(c[::-1], s); D = np.polyval(e[::-1], s)
    return 0.75*s+N/np.sqrt(D)


def taylor(params, n, p):
    """Taylor coefficients of the model at s = 0 up to s^p (numerically, via series algebra)"""
    c = np.r_[params[:n+1], np.zeros(p+1)][:p+1]
    e = np.r_[1., params[n+1:], np.zeros(p+1)][:p+1]
    # 1/sqrt(D) series
    inv = np.zeros(p+1); inv[0] = 1.
    # g = D^(-1/2): g' D = -(1/2) g D'  -> recursion
    for k in range(1, p+1):
        acc = 0.
        for j in range(1, k+1):
            acc += (-(0.5)*j-(k-j))*e[j]*inv[k-j]
        inv[k] = acc/k
    out = np.convolve(c, inv)[:p+1]
    out[1] += 0.75
    return out


def fit(n, p, s, m, w_strong=True):
    npar = (n+1)+(2*n+1)

    def resid(x):
        r = [(model(x, s, n)-m)/m]
        t = taylor(x, n, p)
        r.append(1e3*(t-np.array(d[:p+1]))/np.maximum(1, np.abs(d[:p+1])))
        if w_strong:
            r.append([1e3*(x[n]/np.sqrt(abs(x[-1]))-4/3)])
        return np.concatenate(r)
    best = None
    rng = np.random.default_rng(0)
    for trial in range(40):
        x0 = np.r_[1.5, rng.uniform(0, 3, n), rng.uniform(0, 3, 2*n+1)]
        try:
            res = least_squares(resid, x0, method='lm', max_nfev=20000)
        except Exception:
            continue
        if np.all(np.isfinite(res.fun)) and (best is None or res.cost < best.cost):
            best = res
    return best


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    s, m, src = data()
    res = fit(n, p, s, m)
    x = res.x
    rel = (model(x, s, n)-m)/m
    print(f'n={n} p={p}: params = {np.array2string(x, precision=10)}')
    print(f'max |rel err| = {np.max(np.abs(rel)):.2e} over {len(s)} points, s in [{s.min():.3g}, {s.max():.3g}]')
    # implied strong coupling coefficients
    big = np.logspace(4, 6, 20)
    y = (model(x, big, n)-0.75*big)
    A = np.array([big**-0.5, big**-1, big**-1.5]).T
    print('implied a1, a2, a3 =', np.linalg.lstsq(A, y, rcond=None)[0])
    print('Taylor at 0:', taylor(x, n, 6), ' exact:', np.round(d, 6))
