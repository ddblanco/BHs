"""Independent evaluator of E_{ab} = G_{ab} + alpha H_{ab} for the equal-spin ansatz.

Written for the 2026-09-25 review; it shares no code with src/rotating_bh.
The metric is built from eq. (4) of the manuscript in coordinates (t,r,theta,phi1,phi2);
the Riemann tensor is derived symbolically with b,f,h,w as undefined functions of r,
then lambdified and contracted numerically. H_{ab} is the Lanczos tensor
    H_ab = 2(R R_ab - 2 R_ac R^c_b - 2 R_acbd R^cd + R_a^cde R_bcde) - g_ab L_GB/2.
"""
import pickle
from pathlib import Path

import numpy as np
import sympy as sp

CACHE = Path(__file__).with_name('.riemann_cache.pkl')
r, th = sp.symbols('r theta', positive=True)
names = ['b', 'f', 'h', 'w']
funcs = {n: sp.Function(n)(r) for n in names}
# value, first and second derivative symbols
S = {(n, k): sp.Symbol(f'{n}{k}') for n in names for k in range(3)}


def metric():
    b, f, h, w = (funcs[n] for n in names)
    s, c = sp.sin(th), sp.cos(th)
    g = sp.zeros(5, 5)
    g[0, 0] = -b + h*w**2
    g[0, 3] = g[3, 0] = -h*w*s**2
    g[0, 4] = g[4, 0] = -h*w*c**2
    g[1, 1] = 1/f
    g[2, 2] = r**2
    g[3, 3] = r**2*s**2*c**2 + h*s**4
    g[4, 4] = r**2*s**2*c**2 + h*c**4
    g[3, 4] = g[4, 3] = -r**2*s**2*c**2 + h*s**2*c**2
    return g


def _to_symbols(expr):
    rep = {}
    for n in names:
        F = funcs[n]
        rep[sp.Derivative(F, (r, 2))] = S[(n, 2)]
    expr = expr.subs(rep)
    rep = {sp.Derivative(funcs[n], r): S[(n, 1)] for n in names}
    expr = expr.subs(rep)
    return expr.subs({funcs[n]: S[(n, 0)] for n in names})


def riemann_lower():
    """R_{abcd} as a dict of sympy expressions in (r, theta, b0..w2)."""
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    X = [sp.Symbol('t'), r, th, sp.Symbol('p1'), sp.Symbol('p2')]
    g = metric()
    gi = sp.simplify(g.inv())
    dg = [[[sp.diff(g[a, b], X[c]) for c in range(5)] for b in range(5)] for a in range(5)]
    Gam = [[[sp.simplify(sum(gi[a, d]*(dg[d][b][c] + dg[d][c][b] - dg[b][c][d])
                             for d in range(5))/2)
             for c in range(5)] for b in range(5)] for a in range(5)]
    # R^a_{bcd} = d_c Gam^a_{db} - d_d Gam^a_{cb} + Gam^a_{ce} Gam^e_{db} - Gam^a_{de} Gam^e_{cb}
    Rup = {}
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(c+1, 5):
                    e_ = (sp.diff(Gam[a][d][b], X[c]) - sp.diff(Gam[a][c][b], X[d])
                          + sum(Gam[a][c][e]*Gam[e][d][b] - Gam[a][d][e]*Gam[e][c][b]
                                for e in range(5)))
                    Rup[a, b, c, d] = e_
    Rlow = {}
    for a in range(5):
        for b in range(a+1, 5):
            for c in range(5):
                for d in range(c+1, 5):
                    if (c, d) < (a, b):
                        continue
                    e_ = sum(g[a, e]*Rup[e, b, c, d] for e in range(5))
                    e_ = sp.simplify(_to_symbols(e_))
                    if e_ != 0:
                        Rlow[a, b, c, d] = e_
    CACHE.write_bytes(pickle.dumps(Rlow))
    return Rlow


_compiled = None


def _compile():
    global _compiled
    if _compiled is None:
        Rlow = riemann_lower()
        args = [r, th] + [S[(n, k)] for n in names for k in range(3)]
        keys = list(Rlow)
        fn = sp.lambdify(args, [Rlow[k] for k in keys], 'numpy')
        gfn = sp.lambdify(args, _to_symbols(metric()), 'numpy')
        _compiled = (keys, fn, gfn)
    return _compiled


def full_riemann(rv, thv, vals):
    """Numeric R_{abcd} (5^4 array) and g_{ab} at one point.

    vals: dict with keys like 'b0','b1','b2', ... (value and r-derivatives).
    """
    keys, fn, gfn = _compile()
    args = [rv, thv] + [vals[f'{n}{k}'] for n in names for k in range(3)]
    comps = fn(*args)
    R = np.zeros((5, 5, 5, 5))
    for (a, b, c, d), v in zip(keys, comps):
        for (p, q, s_, t_), sign in [((a, b, c, d), 1), ((b, a, c, d), -1),
                                    ((a, b, d, c), -1), ((b, a, d, c), 1)]:
            R[p, q, s_, t_] = sign*v
            R[s_, t_, p, q] = sign*v
    g = np.array(gfn(*args), dtype=float)
    return R, g


def field_equations(rv, thv, vals, alpha, metric=False):
    """Return (E_ab, G_ab, alpha*H_ab, Kretschmann, L_GB) at one point."""
    R, g = full_riemann(rv, thv, vals)
    gi = np.linalg.inv(g)
    Ric = np.einsum('acbd,cd->ab', R, gi)
    Rs = np.einsum('ab,ab->', gi, Ric)
    Rud = np.einsum('ae,ebcd->abcd', gi, R)            # R^a_bcd
    Rupall = np.einsum('ae,bf,cg,dh,efgh->abcd', gi, gi, gi, gi, R)
    RicUp = gi@Ric@gi
    K = np.einsum('abcd,abcd->', R, Rupall)
    RicSq = np.einsum('ab,ab->', Ric, RicUp)
    LGB = Rs**2 - 4*RicSq + K
    G = Ric - g*Rs/2
    term = (Rs*Ric - 2*Ric@gi@Ric - 2*np.einsum('acbd,cd->ab', R, RicUp)
            + np.einsum('acde,bfgh,cf,dg,eh->ab', R, R, gi, gi, gi))
    H = 2*term - g*LGB/2
    if metric:
        return G + alpha*H, G, alpha*H, K, LGB, g
    return G + alpha*H, G, alpha*H, K, LGB


if __name__ == '__main__':
    import time
    t0 = time.time()
    Rl = riemann_lower()
    print(f'{len(Rl)} independent nonzero R_abcd components, {time.time()-t0:.1f}s')
