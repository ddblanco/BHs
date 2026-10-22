"""Global approximant for m(s) on 0 < s < infinity.

y(s) = m(s) - 3s/4 = sum_{k=1..K} c_k t^k,   t = (1 + s/kappa)^(-1/2).
t is analytic at s = 0 and t ~ sqrt(kappa/s) at large s, so y has a Taylor series at s = 0 and a
series in s^(-1/2) at infinity.  Exact constraints imposed linearly (for given kappa):
  y(0) = 3/2,  y'(0) = 1/4,  c_1 sqrt(kappa) = 4/3  (a1),  optionally c_2 kappa = a2.
kappa is scanned; the remaining c_k are a weighted least-squares fit to the numerical branch
(spectral s <= 2, two-sided sweep, BVP, multiple shooting).  Writes global2.json."""
import json, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A2 = 0.456526735220      # conditional value (reports/strong-coupling-endpoint.pdf)


def data():
    pts = []
    for r in json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch']:
        if r['alpha'] > 0:
            s = np.pi**(2/3)*r['alpha']/r['J']**(2/3)
            if s <= 2:
                pts.append((s, r['M']/(np.pi**(1/3)*r['J']**(2/3)), 'spec'))
    pts += [(d['s'], d['m'], 'sweep') for d in json.loads((HERE/'sweep.json').read_text())]
    pts += [(d['s'], d['m'], 'bvp') for d in json.loads((HERE/'bvp_results.json').read_text())
            if d['status'] == 0 and d['s'] < 9.1]
    pts += [(d['s'], d['m'], 'ms') for d in json.loads((HERE/'ms_results.json').read_text()) if d['ok']]
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), [p[2] for p in pts]


def design(s, kappa, K):
    t = (1+s/kappa)**-0.5
    return np.array([t**k for k in range(1, K+1)]).T


def fit(s, m, kappa, K, fix_a2):
    """constrained LSQ for c_1..c_K; relative weights"""
    y = m-0.75*s
    A = design(s, kappa, K)
    # constraints C c = d
    C = [np.ones(K),                                            # y(0) = sum c_k = 3/2
         np.array([-k/(2*kappa) for k in range(1, K+1)]),       # y'(0) = sum c_k (-k/2kappa) = 1/4
         np.eye(K)[0]*np.sqrt(kappa)]                           # c1 sqrt(kappa) = 4/3
    d = [1.5, 0.25, 4/3]
    if fix_a2:
        C.append(np.eye(K)[1]*kappa); d.append(A2)
    C, d = np.array(C), np.array(d)
    w = 1/m
    Aw, yw = A*w[:, None], y*w
    n = len(d)
    KKT = np.block([[2*Aw.T@Aw, C.T], [C, np.zeros((n, n))]])
    rhs = np.concatenate([2*Aw.T@yw, d])
    c = np.linalg.solve(KKT, rhs)[:K]
    rel = (A@c+0.75*s)/m-1
    return c, rel


if __name__ == '__main__':
    s, m, src = data()
    print(len(s), 'points, s in', s.min(), s.max(), {k: src.count(k) for k in set(src)})
    best = {}
    for fix in (False, True):
        for K in range(4, 11):
            res = []
            for kappa in np.geomspace(0.05, 20, 120):
                c, rel = fit(s, m, kappa, K, fix)
                res.append((np.max(np.abs(rel)), kappa, c))
            e, kappa, c = min(res, key=lambda r: r[0])
            a2 = c[1]*kappa
            print(f'fix_a2={fix} K={K} kappa={kappa:.4f} max rel err={e:.2e} a2={a2:.5f}')
            best[f'{fix}_{K}'] = dict(kappa=kappa, c=list(c), maxrel=e, a2=a2)
    (HERE/'global2.json').write_text(json.dumps(best, indent=1))
