"""Large-r expansion of the reduced fixed-J system, numerically, z = 1/r:
    f = Ft(z) = 1 + sum a_k z^k,   h = z^-2 Ht(z),  Ht = 1 + sum b_k z^k.
Then  f' = -z^2 Ft',  h' = z^-1 Qt,  Qt = 2 Ht - z Ht',  h'' = Qt - z Qt'.
Each equation is multiplied by z^N (N = max total negative power) to become a power series.
probe(): which coefficient orders constrain which unknowns (free data: mu = -a_2, b_4)."""
import numpy as np
from tps import TPS
from ode_poly import TABLES, NAMES


def eq_series(a, b, alpha, K, Nshift=None):
    """a, b: coefficient arrays (index k -> z^k) of Ft, Ht (a[0] = b[0] = 1)"""
    TPS.K = K
    dt = np.result_type(np.asarray(a).dtype, np.asarray(b).dtype)
    Ft = TPS(np.asarray(a, dtype=dt)); Ht = TPS(np.asarray(b, dtype=dt))
    z = TPS(np.array([0., 1.], dtype=dt))
    Qt = 2*Ht-z*Ht.deriv()
    fp_ = -(z*z)*Ft.deriv()          # f'
    hpp_ = Qt-z*Qt.deriv()           # h''
    # powers
    md = [max(m[i] for nm in NAMES for m, _, _ in TABLES[nm]) for i in range(4)]
    pw = {}
    for name, base, n in (('f', Ft, md[1]), ('h', Ht, md[2]), ('q', Qt, md[3])):
        lst = [TPS(np.r_[1., np.zeros(K)].astype(dt))]
        for k in range(n):
            lst.append(lst[-1]*base)
        pw[name] = lst
    if Nshift is None:
        Nshift = max(m[0]+2*m[2]+m[3] for nm in NAMES for m, _, _ in TABLES[nm])
    out = {}
    for nm in NAMES:
        acc = np.zeros(K+1, dtype=dt)
        cache = {}
        for m, ka, c in TABLES[nm]:
            e = Nshift-(m[0]+2*m[2]+m[3])
            if e > K:
                continue
            key = (m[1], m[2], m[3])
            if key not in cache:
                cache[key] = (pw['f'][m[1]]*pw['h'][m[2]]*pw['q'][m[3]]).c
            acc[e:] += c*alpha**ka*cache[key][:K+1-e]
        out[nm] = TPS(acc)
    Eb = out['A_eb']*fp_+out['B_eb']*hpp_+out['C_eb']
    Eh = out['A_eh']*fp_+out['B_eh']*hpp_+out['C_eh']
    return Eb.c, Eh.c


def probe(alpha=1.0, mu=2.5, b4=0.7, K=60):
    rng = np.random.default_rng(0)
    a = np.zeros(K+1); b = np.zeros(K+1); a[0] = b[0] = 1.
    a[2] = -mu; b[4] = b4
    a[3:] = 0.1*rng.standard_normal(K-2); b[5:] = 0.1*rng.standard_normal(K-4)
    a[1] = 0.05; b[1] = 0.02; b[2] = 0.03; b[3] = 0.01
    Eb, Eh = eq_series(a.astype(complex), b.astype(complex), alpha, K)
    lowb = next(i for i in range(K) if abs(Eb[i]) > 1e-12*np.max(np.abs(Eb)))
    lowh = next(i for i in range(K) if abs(Eh[i]) > 1e-12*np.max(np.abs(Eh)))
    print('lowest nonzero orders: Eb', lowb, 'Eh', lowh)
    print('Nshift', max(m[0]+2*m[2]+m[3] for nm in NAMES for m, _, _ in TABLES[nm]))
    for nm, arr, ks in (('a', a, range(1, 10)), ('b', b, range(1, 10))):
        for k in ks:
            aa = a.astype(complex); bb = b.astype(complex)
            (aa if nm == 'a' else bb)[k] += 1e-20j
            db, dh = eq_series(aa, bb, alpha, K)
            db = np.imag(db)/1e-20; dh = np.imag(dh)/1e-20
            fb = next((i for i in range(K) if abs(db[i]) > 1e-9*max(1, np.max(np.abs(db)))), None)
            fh = next((i for i in range(K) if abs(dh[i]) > 1e-9*max(1, np.max(np.abs(dh)))), None)
            print(f'{nm}{k}: first in Eb at {fb} ({db[fb] if fb is not None else 0:.3g}), Eh at {fh} ({dh[fh] if fh is not None else 0:.3g})')


if __name__ == '__main__':
    probe()


def solve_asym(alpha, mu, b4, M=14, K=None, guess=None, verbose=False):
    """coefficients a_1..a_M (a_2 = -mu fixed), b_1..b_M (b_4 = b4 fixed) by Gauss-Newton on
    Eb[20..19+M+2], Eh[2..M+3]; returns a, b and the residual."""
    K = K or (M+40)
    ia = [k for k in range(2, M+1, 2) if k != 2]
    ib = [k for k in range(2, M+1, 2) if k != 4]
    nu = len(ia)+len(ib)
    u = np.zeros(nu) if guess is None else np.array(guess, float)

    def unpack(v):
        a = np.zeros(K+1, dtype=v.dtype); b = np.zeros(K+1, dtype=v.dtype)
        a[0] = b[0] = 1; a[2] = -mu; b[4] = b4
        a[ia] = v[:len(ia)]; b[ib] = v[len(ia):]
        return a, b

    def eqs(v):
        a, b = unpack(v)
        Eb, Eh = eq_series(a, b, alpha, K)
        return np.r_[Eb[20:20+M+1], Eh[2:2+M+1]]
    for it in range(30):
        e = np.real(eqs(u.astype(complex)))
        Jm = np.empty((len(e), nu))
        for i in range(nu):
            vc = u.astype(complex); vc[i] += 1e-20j
            Jm[:, i] = np.imag(eqs(vc))/1e-20
        sc = np.max(np.abs(Jm), axis=1); sc[sc == 0] = 1
        step = np.linalg.lstsq(Jm/sc[:, None], e/sc, rcond=None)[0]
        u = u-step
        if verbose:
            print(it, np.max(np.abs(step)), np.max(np.abs(e/sc)))
        if np.max(np.abs(step)) < 1e-15*max(1, np.max(np.abs(u))):
            break
    a, b = unpack(u)
    e = np.real(eqs(u.astype(complex)))
    return a[:M+1].real, b[:M+1].real, float(np.max(np.abs(e/sc))), u


if __name__ == '__main__' and True:
    for alpha, mu, b4 in [(0.0, 2.0, 1.0), (1.49, 4.05, 0.3)]:
        a, b, res, u = solve_asym(alpha, mu, b4, M=12, verbose=False)
        print(f'alpha={alpha} mu={mu} b4={b4}: res {res:.2e}')
        print('  a:', np.array2string(a, precision=6))
        print('  b:', np.array2string(b, precision=6))


def solve_seq(alpha, mu, b4, M=16, K=None, newton=3):
    """level-by-level: level k (even) unknowns (a_k, b_k) from (Eb[k+19], Eh[k+1]);
    k = 2: a_2 = -mu free, b_2 from Eh[3];  k = 4: b_4 free, a_4 from Eb[23]."""
    K = K or (M+24)
    a = np.zeros(K+1, complex); b = np.zeros(K+1, complex)
    a[0] = b[0] = 1; a[2] = -mu; b[4] = b4
    for k in range(2, M+1, 2):
        unk = [('a', k), ('b', k)]
        rows = [('b', k+19), ('h', k+1)]
        if k == 2:
            unk, rows = [('b', 2)], [('h', 3)]
        if k == 4:
            unk, rows = [('a', 4)], [('b', 23)]
        for it in range(newton):
            def ev(v):
                aa, bb = a.copy(), b.copy()
                for (w, i), val in zip(unk, v):
                    (aa if w == 'a' else bb)[i] = val
                Eb, Eh = eq_series(aa, bb, alpha, k+20)
                return np.array([(Eb if w == 'b' else Eh)[i] for w, i in rows])
            v0 = np.array([(a if w == 'a' else b)[i].real for w, i in unk])
            e = np.real(ev(v0.astype(complex)))
            Jm = np.empty((len(rows), len(unk)))
            for j in range(len(unk)):
                vc = v0.astype(complex); vc[j] += 1e-20j
                Jm[:, j] = np.imag(ev(vc))/1e-20
            v = v0-np.linalg.solve(Jm, e)
            for (w, i), val in zip(unk, v):
                (a if w == 'a' else b)[i] = val
    Eb, Eh = eq_series(a, b, alpha, M+20)
    chk = dict(Eb=np.abs(Eb[19:M+20]).max()/max(1, np.abs(Eb).max()), Eh=np.abs(Eh[:M+2]).max()/max(1, np.abs(Eh).max()))
    return a[:M+1].real, b[:M+1].real, chk


if __name__ == '__main__':
    for alpha, mu, b4 in [(0.0, 2.0, 1.0), (0.0, 2.3, 0.7), (1.49, 4.05, 0.3), (25., 40., 3.)]:
        a, b, chk = solve_seq(alpha, mu, b4, M=16)
        print(f'SEQ alpha={alpha} mu={mu} b4={b4}: {chk}')
        print('  a:', np.array2string(a[::2], precision=6)); print('  b:', np.array2string(b[::2], precision=6))
