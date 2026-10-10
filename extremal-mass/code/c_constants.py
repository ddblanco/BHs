"""Regularised horizon values C_w of the hyperlog words, and their MZV identification.

C_w = constant term (u^0 ell^0) of G_w at x=1+u, ell=log u.  Computed by
matching at x=3/2 the series at infinity (z=1/x) with the horizon series.
Identification: PSLQ against the MZV basis of the word's weight
(weight 2: z2; 3: z3; 4: z4; 5: z5, z2 z3; 6: z6, z3^2).  This is a numerical
identification at 100 digits, with a 1e-80 residual threshold, not a proof.
"""
from pathlib import Path
import datetime, itertools, json, platform, sys
import mpmath as mp

HERE = Path(__file__).resolve().parent
mp.mp.dps = 110
NI, NH = 620, 360
XM = mp.mpf(3)/2
ZM, UM = 1/XM, XM-1
LM = mp.log(UM)


def words(wmax):
    return [w for n in range(1, wmax+1) for w in itertools.product((0, 1), repeat=n) if w[-1] == 1]


def inf_series(ws):
    S = {}
    for w in sorted(ws, key=len):
        if len(w) == 1:
            S[w] = [mp.mpf(0)]+[-mp.mpf(1)/k for k in range(1, NI+1)]
        elif w[0] == 0:
            g = S[w[1:]]; S[w] = [mp.mpf(0)]+[-g[k]/k for k in range(1, NI+1)]
        else:
            g = S[w[1:]]; acc = mp.mpf(0); out = [mp.mpf(0)]
            for k in range(1, NI+1):
                acc += g[k]; out.append(-acc/k)
            S[w] = out
    return S


def integ(series):
    """series: dict j -> list a_m (coefficient of u^(m+shift) ell^j), with key ('m', j) form.
    Input: dict (m, j) -> coeff for integrand u^m ell^j; returns integral, same form."""
    out = {}
    for (m, j), c in series.items():
        if c == 0: continue
        if m == -1:
            out[(0, j+1)] = out.get((0, j+1), 0)+c/(j+1)
            continue
        # int u^m ell^j = u^(m+1) sum_i (-1)^i j!/(j-i)! ell^(j-i)/(m+1)^(i+1)
        fac = mp.mpf(1)
        for i in range(j+1):
            key = (m+1, j-i)
            out[key] = out.get(key, 0)+c*(-1)**i*fac/mp.mpf(m+1)**(i+1)
            fac *= (j-i)
    return out


def hor_series(ws, C):
    E = {}
    for w in sorted(ws, key=len):
        if len(w) == 1:
            e = {(0, 1): mp.mpf(1)}
            for k in range(1, NH+1): e[(k, 0)] = -(-1)**(k+1)/mp.mpf(k)
            E[w] = e; continue
        g = E[w[1:]]
        if w[0] == 1:
            integrand = {(m-1, j): c for (m, j), c in g.items()}
        else:
            integrand = {}
            for (m, j), c in g.items():
                for k in range(0, NH-m+1):
                    integrand[(m+k, j)] = integrand.get((m+k, j), 0)+c*(-1)**k
        e = {k: v for k, v in integ(integrand).items() if k[0] <= NH}
        e[(0, 0)] = e.get((0, 0), 0)+C.get(w, 0)
        E[w] = e
    return E


def step(g, a):
    if a == 1:
        integrand = {(m-1, j): c for (m, j), c in g.items()}
    else:
        integrand = {}
        for (m, j), c in g.items():
            for k in range(0, NH-m+1):
                integrand[(m+k, j)] = integrand.get((m+k, j), 0)+c*(-1)**k
    return {k: v for k, v in integ(integrand).items() if k[0] <= NH}


def evalE(e):
    return mp.fsum(c*UM**m*LM**j for (m, j), c in e.items())


def main(wmax=5):
    ws = words(wmax)
    S = inf_series(ws)
    C, E = {}, {}
    for w in sorted(ws, key=len):
        E.update(hor_series([w], C) if len(w) == 1 else {w: step(E[w[1:]], w[0])})
        if len(w) == 1: C[w] = mp.mpf(0); continue
        Gm = mp.fsum(S[w][k]*ZM**k for k in range(1, NI+1))
        C[w] = Gm-evalE(E[w])
        E[w][(0, 0)] = E[w].get((0, 0), 0)+C[w]
    # control: G_(0,1)(x) = Li2(1/x) exactly -> C_01 = zeta(2)
    assert abs(C[(0, 1)]-mp.zeta(2)) < mp.mpf(10)**-90
    z = {k: mp.zeta(k) for k in range(2, 7)}
    basis = {2: [('z2', z[2])], 3: [('z3', z[3])], 4: [('z4', z[4])], 5: [('z5', z[5]), ('z2*z3', z[2]*z[3])],
             6: [('z6', z[6]), ('z3**2', z[3]**2)]}
    ident = {}
    with mp.workdps(100):
        for w in ws:
            if len(w) == 1 or abs(C[w]) < mp.mpf(10)**-95: ident[w] = '0'; continue
            b = basis[len(w)]
            rel = mp.pslq([C[w]]+[v for _, v in b], maxcoeff=10**8, maxsteps=10**6)
            assert rel and rel[0] != 0, (w, rel)
            val = -sum(mp.mpf(c)*v for c, (_, v) in zip(rel[1:], b))/rel[0]
            assert abs(val-C[w]) < mp.mpf(10)**-80, (w, rel)
            ident[w] = '+'.join(f'({-c}/{rel[0]})*{n}' for c, (n, _) in zip(rel[1:], b) if c) or '0'
    rows = {'C'+''.join(map(str, w)): dict(value=mp.nstr(C[w], 60), mzv=ident[w]) for w in ws}
    for w in ws:
        print('C'+''.join(map(str, w)), ident[w], mp.nstr(C[w], 25), flush=True)
    (HERE/'c_constants.json').write_text(json.dumps(dict(
        date=datetime.datetime.now().astimezone().isoformat(), command=[sys.executable, *sys.argv],
        environment=dict(python=platform.python_version(), mpmath=mp.__version__),
        method='match series at infinity (620 terms) and horizon (360 terms) at x=3/2, 110 digits; PSLQ to 1e-80',
        control='C01 = zeta(2) asserted to 1e-90', constants=rows), indent=1)+'\n')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
