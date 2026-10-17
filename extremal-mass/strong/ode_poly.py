"""Monomial tables of the six coefficients A, B, C (equations eb, eh) of the
reduced fixed-J system  A f' + B h'' + C = 0, as polynomials in (r, f, h, hp)
with coefficients polynomial in alpha.  Cached in ode_poly.pkl.

Fast evaluators:
  coeffs(r, f, h, hp, alpha)        numpy arrays (vectorised)
  coeffs_tps(R, f, h, hp, alpha)    truncated power series (tps.TPS)
"""
import sys
import pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'code'))
NAMES = ('A_eb', 'B_eb', 'C_eb', 'A_eh', 'B_eh', 'C_eh')


def build():
    import sympy as s
    from reduced_system import r, alpha, f, h, p, q, t, C2
    E = pickle.load(open(HERE/'reduced_ode.pkl', 'rb'))
    tables = {}
    for name in ('eb', 'eh'):
        P = s.Poly(s.expand(E[name].subs(C2, 32)), p, t)
        for lab, mon in (('A', (1, 0)), ('B', (0, 1)), ('C', (0, 0))):
            X = s.Poly(P.coeff_monomial(p**mon[0]*t**mon[1]), r, f, h, q, alpha)
            tables[lab+'_'+name] = [(tuple(m[:4]), m[4], int(c)) for m, c in X.terms()]
    pickle.dump(tables, open(HERE/'ode_poly.pkl', 'wb'))
    return tables


def load():
    p = HERE/'ode_poly.pkl'
    return pickle.load(open(p, 'rb')) if p.exists() else build()


TABLES = load()


def _collapse(alpha):
    """monomials in (r,f,h,hp) with numeric coefficients at this alpha"""
    out = {}
    for nm in NAMES:
        d = {}
        for m, ka, c in TABLES[nm]:
            d[m] = d.get(m, 0.)+c*alpha**ka
        out[nm] = [(m, c) for m, c in d.items() if c != 0]
    return out


_CACHE = {}


def collapsed(alpha):
    key = alpha if not isinstance(alpha, complex) else ('c', alpha)
    if key not in _CACHE:
        _CACHE[key] = _collapse(alpha)
    return _CACHE[key]


def maxdeg():
    return [max(m[i] for nm in NAMES for m, _, _ in TABLES[nm]) for i in range(4)]


def coeffs(r, f, h, hp, alpha):
    C = collapsed(alpha)
    md = maxdeg()
    pw = [[np.ones_like(v)] for v in (r, f, h, hp)]
    for i, v in enumerate((r, f, h, hp)):
        for k in range(md[i]):
            pw[i].append(pw[i][-1]*v)
    out = []
    for nm in NAMES:
        acc = 0.
        for m, c in C[nm]:
            acc = acc+c*pw[0][m[0]]*pw[1][m[1]]*pw[2][m[2]]*pw[3][m[3]]
        out.append(acc)
    return out


def coeffs_tps(R, f, h, hp, alpha):
    from tps import TPS
    C = collapsed(alpha)
    md = maxdeg()
    pw = []
    for i, v in enumerate((R, f, h, hp)):
        lst = [TPS(np.r_[1., np.zeros(TPS.K)].astype(v.c.dtype))]
        for k in range(md[i]):
            lst.append(lst[-1]*v)
        pw.append(lst)
    # group by (r,f) then (h,hp) products cached
    cache = {}
    out = []
    K = TPS.K+1
    for nm in NAMES:
        acc = np.zeros(K, dtype=np.result_type(R.c.dtype, f.c.dtype, h.c.dtype, hp.c.dtype))
        for m, c in C[nm]:
            key = m
            if key not in cache:
                a = cache.get(('rf', m[0], m[1]))
                if a is None:
                    a = pw[0][m[0]]*pw[1][m[1]]
                    cache[('rf', m[0], m[1])] = a
                b = cache.get(('hq', m[2], m[3]))
                if b is None:
                    b = pw[2][m[2]]*pw[3][m[3]]
                    cache[('hq', m[2], m[3])] = b
                cache[key] = (a*b).c
            acc = acc+c*cache[key]
        out.append(TPS(acc))
    return out


if __name__ == '__main__':
    T = build()
    for k, v in T.items():
        print(k, len(v), 'max degrees', [max(m[0][i] for m in v) for i in range(4)], 'alpha', max(m[1] for m in v))
