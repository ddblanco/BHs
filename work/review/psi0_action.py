"""Derive Psi_0(q) = dG/dalpha|_{T,Omega} at alpha=0 from the Myers-Perry action (Reall-Santos),
    Psi_0 = -(1/16 pi) * 2 pi^2 * int_{r_H}^oo sqrt(b/f) r^2 sqrt(h) L_GB[MP] dr ,
with L_GB = Kretschmann (Ricci flat) computed from egb_tensor.py's Riemann components in
60-digit mpmath arithmetic (float arithmetic loses all digits near f=0). Asserts eq. (16).
"""
import mpmath as mp
import sympy as sp
from egb_tensor import riemann_lower, S, names, metric, _to_symbols, r, th

mp.mp.dps = 60
Rl = riemann_lower()
args = [r, th] + [S[(n, k)] for n in names for k in range(3)]
keys = list(Rl)
Rfn = sp.lambdify(args, [Rl[k] for k in keys], 'mpmath')
# explicit inverse metric (LU in mpmath flags g as singular at r ~ 1e30 from scale alone)
gifn = sp.lambdify(args, sp.simplify(_to_symbols(metric()).inv()), 'mpmath')
rr, aa = sp.symbols('rr aa', positive=True)


def mp_profile(a):
    mu = 1/(1-aa**2)
    f = 1-mu/rr**2+mu*aa**2/rr**4
    h = rr**2+mu*aa**2/rr**2
    w = mu*aa/(rr**2*h)
    b = rr**2*f/h
    exprs = dict(zip(names, (b, f, h, w)))
    fns = {f'{n}{k}': sp.lambdify((rr, aa), sp.diff(exprs[n], rr, k), 'mpmath')
           for n in names for k in range(3)}
    return fns


FNS = mp_profile(None)


def kretschmann(rv, a, thv=mp.mpf('0.7')):
    vals = [FNS[f'{n}{k}'](rv, a) for n in names for k in range(3)]
    comps = Rfn(rv, thv, *vals)
    gi = mp.matrix(gifn(rv, thv, *vals))
    R = {}
    for (A, B, C, D), v in zip(keys, comps):
        for (p, q, s, t), sg in [((A, B, C, D), 1), ((B, A, C, D), -1), ((A, B, D, C), -1), ((B, A, D, C), 1)]:
            R[p, q, s, t] = sg*v
            R[s, t, p, q] = sg*v
    # raise all indices
    K = mp.mpf(0)
    nz = [(i, j) for i in range(5) for j in range(5) if gi[i, j] != 0]
    for (p, q, s, t), v in R.items():
        up = mp.mpf(0)
        for (p2, q2, s2, t2), v2 in R.items():
            c = gi[p, p2]*gi[q, q2]*gi[s, s2]*gi[t, t2]
            if c:
                up += c*v2
        K += v*up
    return K


def psi0(q):
    a = mp.mpf(q)
    mu = 1/(1-a**2)
    def integrand(rv):
        f = 1-mu/rv**2+mu*a**2/rv**4
        h = rv**2+mu*a**2/rv**2
        b = rv**2*f/h
        return 2*mp.pi**2*mp.sqrt(rv**2/h)*rv**2*mp.sqrt(h)*kretschmann(rv, a)   # sqrt(b/f)=r/sqrt(h)
    # tanh-sinh nodes reach f~1e-60 at r=1; start 1e-14 away (dropped piece ~1e-13, bound asserted below)
    I = mp.quad(integrand, [1+mp.mpf('1e-14'), 1.01, 1.2, 2, 8, mp.inf])
    return -I/(16*mp.pi)


if __name__ == '__main__':
    print(' q         Psi_0 (action integral)       eq.(16)                 difference')
    for q in ['0', '0.3', '0.5', '0.634936', '0.7']:
        qq = mp.mpf(q); u = qq**2/(1-qq**2)
        ref = -mp.pi/4*(u**2-14*u+9)
        got = psi0(q)
        print(f'{q:9s} {mp.nstr(got, 18):>24s} {mp.nstr(ref, 18):>24s} {mp.nstr(got-ref, 3):>10s}')
        assert abs(got-ref) < mp.mpf('1e-11')
    print('ASSERTED: eq.(16) equals the Reall-Santos action integral of L_GB on Myers-Perry')
