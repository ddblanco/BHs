"""Recursive exact solver for the extremal fixed-J exterior to order alpha^N.

Fields are polynomials in hyperlog words g_w(x=r^2) with QQ(r) coefficients.
At each order n:
  * the reduced RHS (angular momentum eliminated, C2=32 fixed) evaluated on the
    order-(n-1) truncation gives the sources of the linear system;
  * h_n = particular (three exact quadratures of the factorised master
    operator) + K1 y3 + K2 y2 + K3 y1;
  * f_n, log(b)_n and v_n=w_n/sqrt2 follow algebraically / by one quadrature;
  * constants: h_n -> 0, b->1, w->0 at infinity; with x_H = 1+sum X_k alpha^k,
    the horizon-centred expansions F(u)=f(x_H+u), H(u)=h(x_H+u) must have no
    poles, no u^0 log terms, and F must start at u^2 (double zero).
Horizon expansions keep the regularised word values C_w as symbols.
"""
from pathlib import Path
import datetime, json, platform, sys, time, itertools
import sympy as s
from sympy.polys.rings import ring
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reduced_system import r, alpha, reduced_equations, f as fS, h as hS, p as pS, q as qS, t as tS, u as uS, v as vS, C2
from radial_series import compile_exact
import hyperlog as HL
from hyperlog import x

# maximal word weight kept (argv[2]); a longer word raises KeyError in GEN
WMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ZERO_, ONE_ = s.Integer(0), s.Integer(1)
WORDS = [w for n in range(1, WMAX+1) for w in itertools.product((ZERO_, ONE_), repeat=n) if w[-1] == ONE_]
NAME = {w: 'g'+''.join(str(a) for a in w) for w in WORDS}
NW = len(WORDS)
DOMR = s.QQ.frac_field(r)
# Horizon constants C_w enter through their MZV identifications (c_constants.json,
# PSLQ at 100 digits): every C_w is a polynomial in Z2=zeta2, Z3=zeta3, Z5=zeta5
# (zeta4 = 2/5 Z2^2, zeta6 = 8/35 Z2^3).  Results are exact conditional on them.
CSYMLIST = list(s.symbols('Z2 Z3 Z5'))
NC = len(CSYMLIST)
RING, *GENS = ring([NAME[w] for w in WORDS]+[str(c) for c in CSYMLIST], DOMR)
GEN = dict(zip(WORDS, GENS[:NW]))
u_, z_, ell = s.symbols('u z ell', positive=True)
_Z2, _Z3, _Z5 = CSYMLIST
_MZV = dict(z2=_Z2, z3=_Z3, z5=_Z5, z4=s.Rational(2, 5)*_Z2**2, z6=s.Rational(8, 35)*_Z2**3)
_CJ = json.loads((HERE/'c_constants.json').read_text())['constants']
CSYM = {w: s.expand(s.sympify(_CJ['C'+NAME[w][1:]]['mzv'], locals=_MZV)) for w in WORDS}
ZVAL = {_Z2: s.pi**2/6, _Z3: s.zeta(3), _Z5: s.zeta(5)}
OUT = []


def say(t): print(t, flush=True); OUT.append(t)


def rc(expr):
    """sympy expr in r (and C symbols, polynomially) -> ring element"""
    e = s.expand(s.sympify(expr))
    if not e.free_symbols & set(CSYMLIST):
        return RING(DOMR.from_sympy(e))
    poly = s.Poly(e, *CSYMLIST)
    out = RING.zero
    for mon, c in poly.terms():
        m = RING.one
        for i, k in enumerate(mon):
            if k: m *= GENS[NW+i]**k
        out += RING(DOMR.from_sympy(s.cancel(c)))*m
    return out


def cmon_expr(cm):
    return s.Mul(*[CSYMLIST[i]**k for i, k in enumerate(cm) if k])


def cmon_ring(cm):
    m = RING.one
    for i, k in enumerate(cm):
        if k: m *= GENS[NW+i]**k
    return m


def xc_to_r(c):
    return DOMR.from_sympy(HL.E(c).subs(x, r**2))


def from_F(fx, cm=None):
    out = RING.zero
    for w, c in fx.t.items():
        out += RING(xc_to_r(c))*(GEN[w] if w else 1)
    return out*cmon_ring(cm) if cm else out


_LIN = {}


def lin_monomial(mon):
    if mon in _LIN: return _LIN[mon]
    acc = HL.F({(): 1})
    for i, e in enumerate(mon):
        for _ in range(e):
            acc = HL.mulF(acc, HL.F({WORDS[i]: 1}))
    _LIN[mon] = acc
    return acc


def linearize(p):
    """dict (word, C-monomial) -> DOMR coefficient (canonical, shuffle-reduced)."""
    out = {}
    for mon, c in p.terms():
        wm, cm = mon[:NW], tuple(mon[NW:])
        if sum(wm) <= 1:
            w = () if sum(wm) == 0 else WORDS[wm.index(1)]
            out[(w, cm)] = out.get((w, cm), DOMR.zero)+c
        else:
            for w, cc in lin_monomial(wm).t.items():
                out[(w, cm)] = out.get((w, cm), DOMR.zero)+c*xc_to_r(cc)
    return {k: c for k, c in out.items() if c}


def iszero(p): return not linearize(p)


DGEN = {}
for w in WORDS:
    a = w[0]
    if len(w) == 1:
        DGEN[w] = rc(2*a/(r*(r*r-a)))
    else:
        DGEN[w] = rc(2*r/(r*r-a))*GEN[w[1:]]


def coeff_diff(c):
    return DOMR.from_sympy(s.diff(DOMR.to_sympy(c), r))


def dr(p):
    out = RING.zero
    for mon, c in p.terms():
        cd = coeff_diff(c)
        m = RING.one
        for i, e in enumerate(mon):
            if e: m *= GENS[i]**e
        if cd: out += RING(cd)*m
        for i, e in enumerate(mon[:NW]):
            if e:
                rest = RING.one
                for j, ee in enumerate(mon):
                    if ee: rest *= GENS[j]**(ee-(1 if j == i else 0))
                out += RING(c)*e*rest*DGEN[WORDS[i]]
    return out


def to_Fs(p, odd=False):
    """ring element -> dict C-monomial -> F in x (even in r; odd: divided by 2r first)."""
    res = {}
    for (w, cm), c in linearize(p).items():
        e = DOMR.to_sympy(c)
        if odd: e = e/(2*r)
        e = s.cancel(e)
        assert s.cancel(e-e.subs(r, -r)) == 0, 'parity'
        res[cm] = res.get(cm, HL.F())+HL.F({w: HL.R(s.cancel(e.subs(r, s.sqrt(x))))})
    return res


def to_F(p, odd=False):
    d = to_Fs(p, odd)
    assert set(d) <= {tuple([0]*NC)}, 'transcendental constants present'
    return d.get(tuple([0]*NC), HL.F())


def integ_r(p):
    """Antiderivative in r of an odd ring element."""
    return sum((from_F(HL.integrate(fx), cm) for cm, fx in to_Fs(p, odd=True).items()), RING.zero)


# ---------------- truncated alpha series ----------------
NORD = 2


class Ser:
    def __init__(self, *c):
        self.c = tuple((c[i] if hasattr(c[i], 'ring') else rc(c[i])) if i < len(c) else RING.zero for i in range(NORD))
    @staticmethod
    def co(x_): return x_ if isinstance(x_, Ser) else Ser(x_)
    def __add__(self, o): o = self.co(o); return Ser(*(a+b for a, b in zip(self.c, o.c)))
    __radd__ = __add__
    def __neg__(self): return Ser(*(-a for a in self.c))
    def __sub__(self, o): return self+-self.co(o)
    def __rsub__(self, o): return self.co(o)+-self
    def __mul__(self, o):
        o = self.co(o)
        return Ser(*(sum((self.c[i]*o.c[n-i] for i in range(n+1)), RING.zero) for n in range(NORD)))
    __rmul__ = __mul__
    def inverse(self):
        a = self.c[0]
        assert a.is_ground and a
        ai = RING(DOMR.one/a.LC) if a.LC else None
        out = [ai]
        for n in range(1, NORD):
            out.append(-ai*sum((self.c[i]*out[n-i] for i in range(1, n+1)), RING.zero))
        return Ser(*out)
    def __truediv__(self, o): return self*self.co(o).inverse()
    def __rtruediv__(self, o): return self.co(o)*self.inverse()
    def __pow__(self, n):
        n = int(n)
        if n < 0: return self.inverse()**(-n)
        out = Ser(1)
        for _ in range(n): out = out*self
        return out


def unit_pow(X, p):
    """(1+X)^p for X with zero constant term."""
    out, term = Ser(1), Ser(1)
    for n in range(1, NORD):
        term = term*X
        out = out+term*s.binomial(p, n)
    return out


def ser_exp(X):
    out, term = Ser(1), Ser(1)
    for n in range(1, NORD):
        term = term*X*s.Rational(1, n)
        out = out+term
    return out


# ---------------- expansions ----------------
def inf_word_series(n):
    """Exact series in z=1/x of every word up to z^n."""
    S = {}
    for w in sorted(WORDS, key=len):
        if len(w) == 1:
            S[w] = [s.Integer(0)]+[-s.Rational(1, k) for k in range(1, n+1)]
        else:
            g = S[w[1:]]
            if w[0] == 0:
                S[w] = [s.Integer(0)]+[-g[k]/k for k in range(1, n+1)]
            else:
                acc, out = s.Integer(0), [s.Integer(0)]
                for k in range(1, n+1):
                    acc += g[k]; out.append(-acc/k)
                S[w] = out
    return S


def ser_inf(p, n, WS):
    """Even ring element -> polynomial in z (Laurent, truncated at z^n)."""
    tot = 0
    for (w, cm), c in linearize(p).items():
        e = s.cancel(DOMR.to_sympy(c).subs(r, s.sqrt(1/z_)))*cmon_expr(cm)
        ce = s.series(e, z_, 0, n+1).removeO()
        sw = 1 if not w else sum(WS[w][k]*z_**k for k in range(1, len(WS[w])))
        tot += s.expand(ce*sw)
    tot = s.expand(tot)
    return sum(tot.coeff(z_, k)*z_**k for k in range(-WMAX*4, n+1))


def integ_uell(m, j):
    """Integral of u^m ell^j du (ell=log u), no constant."""
    if m == -1: return ell**(j+1)/(j+1)
    if j == 0: return u_**(m+1)/(m+1)
    return u_**(m+1)*ell**j/(m+1)-s.Rational(j, m+1)*integ_uell(m, j-1)


def trunc_u(e, n):
    e = s.expand(e)
    if e == 0: return e
    return s.Add(*[t for t in (e.args if e.is_Add else [e]) if split_term(t)[1] <= n])


def hor_word_series(n, wmax=WMAX):
    """Expansion of every word at x=1+u through u^n; constants C_w symbolic."""
    E = {}
    log1u = sum((-1)**(k+1)*u_**k/k for k in range(1, n+2))
    for w in sorted(WORDS, key=len):
        if len(w) == 1:
            E[w] = ell-log1u
            continue
        g = s.expand(E[w[1:]])
        if w[0] == 0:
            g = s.expand(g*sum((-u_)**k for k in range(0, n+2)))
            g = trunc_u(g, n)
        else:
            g = s.expand(g/u_)
        out = CSYM[w]
        for t in (g.args if g.is_Add else [g]):
            c, m, j = split_term(t)
            if m <= n-1:
                out += c*integ_uell(m, j)
        E[w] = s.expand(out)
    return E


def split_term(t):
    c = t; m = 0; j = 0
    pd = t.as_powers_dict()
    m = pd.get(u_, 0); j = pd.get(ell, 0)
    c = s.cancel(t/(u_**m*ell**j))
    assert not c.has(u_) and not c.has(ell)
    return c, int(m), int(j)


def ser_hor(p, n, EH):
    """Even ring element -> expansion at x=1+u through u^n (Laurent allowed)."""
    tot = 0
    for (w, cm), c in linearize(p).items():
        e = s.cancel(DOMR.to_sympy(c).subs(r, s.sqrt(1+u_)))*cmon_expr(cm)
        ce = s.series(e, u_, 0, n+1).removeO()
        lo = min([split_term(t)[1] for t in (s.expand(ce).args if s.expand(ce).is_Add else [s.expand(ce)])] or [0])
        sw = 1 if not w else EH[w]
        tot += s.expand(ce*sw)
    return trunc_u(s.expand(tot), n)


def d_u(e):
    return s.expand(s.diff(e, u_)+s.diff(e, ell)/u_)


# ---------------- driver ----------------
def build(N):
    global NORD
    NORD = N+1
    t0 = time.perf_counter()
    d = json.loads((HERE/'master_k1.json').read_text())
    P = lambda e: s.sympify(e, locals={'r': r})
    eb, ef, eh = reduced_equations()
    U = s.factor(-ef.subs(uS, 0)/s.diff(ef, uS))
    rhs_coeff = [eb.subs({pS: 0, tS: 0}), s.diff(eb, pS), s.diff(eb, tS),
                 eh.subs({pS: 0, tS: 0, vS: 0}), s.diff(eh, pS), s.diff(eh, tS), s.diff(eh, vS)]
    eval_u = compile_exact((r, fS, hS, qS, C2, alpha), [U, s.diff(U, r), s.diff(U, fS), s.diff(U, hS), s.diff(U, qS)])
    eval_e = compile_exact((r, fS, hS, qS, uS, C2, alpha), rhs_coeff)
    def rhs(rv, fv, hv, qv, cv, av):
        uv, ur, uf, uh, uq = eval_u(rv, fv, hv, qv, cv, av)
        ec, ep, et, hc, hp, ht, hvcoef = eval_e(rv, fv, hv, qv, uv, cv, av)
        hc += hvcoef*(ur+uh*qv); hp += hvcoef*uf; ht += hvcoef*uq
        det = ep*ht-et*hp
        return (et*hc-ec*ht)/det, (ec*hp-ep*hc)/det, uv
    A = [P(e) for e in d['A']]; B = [P(e) for e in d['B']]; Uc = [P(e) for e in d['U_partials']]
    c_ = s.cancel(s.diff(B[0], r)/B[0]+A[0])
    xs = lambda e: s.cancel(s.cancel(e).subs(r, s.sqrt(x)))
    y1 = 1/x; y2 = (x*x-x-1)/2; z2 = (x-1)*(3*x+1)/(2*x); n3 = 3*(3*x**2+1)/((x-1)**5*(3*x+1))
    y3 = (15*x**2-18*x+7)/(20*x*(x-1)**3)
    src = json.loads((Path(__file__).resolve().parent/'extremal_perturbation.json').read_text())
    base = [P(e) for e in src['base']]; base[3] = s.cancel(base[3]/s.sqrt(2))
    first = [P(e) for e in src['delta']]; first[3] = s.cancel(s.expand(first[3]/s.sqrt(2)))
    Lr = s.log(1-1/r**2)
    def from_Lexpr(e):
        pol = s.Poly(s.expand(e.subs(Lr, s.Symbol('Lq'))), s.Symbol('Lq'))
        return from_F(HL.from_Lpoly([xs(pol.coeff_monomial(s.Symbol('Lq')**j)) for j in range(pol.degree()+1)]))
    fields = {k: [rc(b)] for k, b in zip('bfhv', base)}
    for k, e in zip('bfhv', first): fields[k].append(from_Lexpr(e))
    beta = [None, from_Lexpr(s.cancel(first[0]/base[0]))]
    Xs = [s.Integer(0), s.Rational(16, 3)]
    # ---- consistency of order 1 with the solver's own conventions ----
    WS = inf_word_series(40)
    EH = hor_word_series(N+12, N+1)
    say(f'word expansions ready ({time.perf_counter()-t0:.1f}s)')
    masses, omegas, consts = {}, {}, {}
    K1, K2, K3, Xn, cB, cV = s.symbols('K1 K2 K3 Xn cB cV')
    for n in range(2, N+1):
        tn = time.perf_counter()
        fv = Ser(*fields['f'][:n]); hv = Ser(*fields['h'][:n])
        qv = Ser(*[dr(e) for e in fields['h'][:n]])
        fp, hpp, uu = rhs(Ser(r), fv, hv, qv, Ser(32), Ser(0, 1))
        for k in range(n):
            assert iszero(fp.c[k]-dr(fields['f'][k])), ('f', k)
            assert iszero(hpp.c[k]-dr(dr(fields['h'][k]))), ('h', k)
        sf, sh, su = fp.c[n], hpp.c[n], uu.c[n]
        say(f'order {n}: lower orders satisfy the reduced system; sources built ({time.perf_counter()-tn:.1f}s)')
        Sm = -rc(c_)*sh+rc(B[0])*sf+dr(sh)
        # the source is linear in the zeta monomials: three quadratures per monomial
        hp_ring = RING.zero
        for cm, Sx in to_Fs(Sm*rc(1/(4*r**2)), odd=True).items():
            W1 = HL.integrate(Sx.scale(HL.R(1/n3)))
            Hp = HL.integrate(HL.integrate(W1.scale(HL.R(n3/z2))).scale(HL.R(z2/y1))).scale(HL.R(y1))
            say(f'order {n}: particular h_{n} [{cmon_expr(cm)}] has weight {Hp.weight()}, {len(Hp.t)} words')
            hp_ring += from_F(Hp, cm)
        Ksyms = {'K1': (K1, y3), 'K2': (K2, y2), 'K3': (K3, y1)}
        hn = {'p': hp_ring}
        for k, (sym, y) in Ksyms.items(): hn[k] = rc(y.subs(x, r**2))
        def lin(fn):
            return {k: fn(v, k) for k, v in hn.items()}
        hr = {k: dr(v) for k, v in hn.items()}
        hrr = {k: dr(v) for k, v in hr.items()}
        fn = {k: (hrr[k]-rc(B[1])*hn[k]-rc(B[2])*hr[k]-(sh if k == 'p' else 0))*rc(1/B[0]) for k in hn}
        for k in hn:
            res = dr(fn[k])-rc(A[0])*fn[k]-rc(A[1])*hn[k]-rc(A[2])*hr[k]-(sf if k == 'p' else 0)
            assert iszero(res), ('f-eq', k)
        un = {k: rc(Uc[0])*fn[k]+rc(Uc[1])*hn[k]+rc(Uc[2])*hr[k]+(su if k == 'p' else 0) for k in hn}
        bn = {k: integ_r(v) for k, v in un.items()}         # beta_n without constant
        say(f'order {n}: f_{n} verified against the f-equation; log b quadrature done')
        # assemble symbolic-in-K series for the angular velocity
        Kv = {'p': 1, 'K1': K1, 'K2': K2, 'K3': K3}
        out_n = {}
        for k in hn:
            fl = fields['f'][:n]+[fn[k]] if True else None
        # v_n: v_r = v0_r exp(beta/2) (f/f0)^-1/2 (h/h0)^-3/2 (K/r^2)^-1 ; linear in order-n pieces
        def vr_series(fn_, hn_, bn_):
            F_ = Ser(*(fields['f'][:n]+[fn_])); H_ = Ser(*(fields['h'][:n]+[hn_]))
            Bt = Ser(*([RING.zero]+beta[1:n]+[bn_]))
            Kr = Ser(1)+(Ser(4)-F_-H_*rc(3/r**2))*rc(4/r**2)*Ser(0, 1)
            return Ser(rc(s.diff(base[3], r)))*ser_exp(Bt*s.Rational(1, 2))*unit_pow(F_*rc(1/base[1])-1, s.Rational(-1, 2)) \
                * unit_pow(H_*rc(1/base[2])-1, s.Rational(-3, 2))*Kr.inverse()
        vp = vr_series(fn['p'], hn['p'], bn['p']).c
        for k in range(1, n):
            assert iszero(vp[k]-dr(fields['v'][k])), ('v', k)
        vn = {'p': integ_r(vp[n])}
        zer = RING.zero
        for k in ('K1', 'K2', 'K3'):
            # linear response: subtract the K-independent part
            full = vr_series(fn['p']+fn[k], hn['p']+hn[k], bn['p']+bn[k]).c[n]
            vn[k] = integ_r(full-vp[n])
        say(f'order {n}: angular velocity quadrature done ({time.perf_counter()-tn:.1f}s)')
        # ---------------- constants ----------------
        tot = lambda D: sum((Kv[k]*D[k] for k in D), 0)
        hinf = sum(Kv[k]*ser_inf(hn[k], 2, WS) for k in hn)
        eqs = [s.expand(hinf).coeff(z_, j) for j in (-2, -1, 0)]
        sol = s.solve([e for e in eqs if e != 0], [K2], dict=True)
        sol = sol[0] if sol else {}
        assert all(s.simplify(e.subs(sol)) == 0 for e in eqs), eqs
        binf = sum(Kv[k]*ser_inf(bn[k], 2, WS) for k in bn)
        sol[cB] = -s.expand(binf).coeff(z_, 0).subs(sol)
        vinf = sum(Kv[k]*ser_inf(vn[k], 2, WS) for k in vn)
        sol[cV] = -s.expand(vinf).coeff(z_, 0).subs(sol)
        # horizon-centred expansions
        Xall = Xs+[Xn]
        def centred(fieldlist, nthD, order_u):
            # [alpha^n] of sum_k alpha^k phi_k(1+u+delta), delta=sum X_j alpha^j
            ex = [ser_hor(fieldlist[k], order_u+n, EH) for k in range(n)]
            exn = sum(Kv[k]*ser_hor(nthD[k], order_u+n, EH) for k in nthD)
            alp = s.Symbol('alp')
            delta = sum(Xall[j]*alp**j for j in range(1, n+1))
            total = exn
            for k in range(n):
                e, m = ex[k], 0
                acc = 0
                dm = e
                for m in range(0, n-k+1):
                    coeff = s.expand(delta**m).coeff(alp, n-k)/s.factorial(m)
                    if coeff != 0: acc += coeff*dm
                    dm = d_u(dm)
                total += acc
            return trunc_u(s.expand(total), order_u)
        Fh = centred(fields['f'], fn, 1)
        Hh = centred(fields['h'], hn, 0)
        def coeffmap(e):
            m = {}
            for t in (e.args if e.is_Add else [e]):
                c, a, j = split_term(t) if t.has(u_) or t.has(ell) else (t, 0, 0)
                m[(a, j)] = m.get((a, j), 0)+c
            return m
        hm = coeffmap(Hh)
        hbad = [v for (a, j), v in hm.items() if a < 0 or (a == 0 and j > 0)]
        k1sol = None
        for v in hbad:
            if v.subs(sol).has(K1):
                k1sol = s.solve(v.subs(sol), K1)[0]; break
        sol[K1] = k1sol
        assert all(s.simplify(v.subs(sol)) == 0 for v in hbad), 'h not regular'
        fm = coeffmap(Fh)
        fbad = [v for (a, j), v in fm.items() if a < 0 or (a <= 1 and j > 0)]
        conds = [fm.get((0, 0), 0).subs(sol), fm.get((1, 0), 0).subs(sol)]
        s3 = s.solve(conds, [K3, Xn], dict=True)
        assert len(s3) == 1, s3
        sol.update(s3[0])
        sol = {k: s.simplify(v.subs(sol)) for k, v in sol.items()}
        sol = {k: s.simplify(v.subs(sol)) for k, v in sol.items()}
        assert all(s.simplify(v.subs(sol)) == 0 for v in fbad), 'f not a double zero'
        say(f'order {n}: K1={sol[K1]}, K2={sol.get(K2, 0)}, K3={sol[K3]}, X{n}={sol[Xn]}')
        # store fields of order n
        sub = lambda D: sum((rc(s.sympify(Kv[k]).subs(sol))*D[k] if k != 'p' else D[k] for k in D), RING.zero)
        fields['h'].append(sub(hn)); fields['f'].append(sub(fn))
        bfull = sub(bn)+rc(sol[cB])
        vfull = sub(vn)+rc(sol[cV])
        fields['v'].append(vfull)
        beta.append(bfull)
        bser = Ser(rc(base[0]))*ser_exp(Ser(*([RING.zero]+beta[1:n+1])))
        fields['b'] = [bser.c[k] for k in range(n+1)]
        Xs.append(sol[Xn])
        # mass from b tail and Omega_H from v at the horizon
        bt = s.expand(ser_inf(fields['b'][n], 2, WS))
        assert bt.coeff(z_, 0) == 0
        Mn = s.simplify(-3*s.pi*bt.coeff(z_, 1)/8)
        Vh = coeffmap(centred(fields['v'], {'p': vfull}, 0) if False else centred(fields['v'][:n], {'p': vfull}, 0))
        assert all(s.simplify(v) == 0 for (a, j), v in Vh.items() if a < 0 or (a == 0 and j > 0))
        Omn = s.sqrt(2)*s.sympify(Vh.get((0, 0), 0)).subs(Xn, Xs[-1])
        masses[n], omegas[n] = Mn, s.simplify(Omn)
        say(f'order {n}: M_{n} (zeta values) = {s.expand(Mn.subs(ZVAL))}; numeric {s.N(Mn.subs(ZVAL), 30)}')
        consts[n] = {str(k): str(v) for k, v in sol.items()}
        say(f'order {n}: M_{n} = {Mn}')
        say(f'order {n}: Omega_{n} = {omegas[n]}')
        # first-law control: Omega_n J0^((1+2n)/3) = (1-n)/3 c_n,  c_n = M_n J0^((2n-2)/3)
        J0 = s.pi*s.sqrt(2)/4
        lhs = omegas[n]*J0**s.Rational(1+2*n, 3); rhs_ = s.Rational(1-n, 3)*Mn*J0**s.Rational(2*n-2, 3)
        diff = s.simplify(s.expand(lhs-rhs_))
        say(f'order {n}: first-law control Omega vs M: residual {diff}')
        say(f'order {n}: done in {time.perf_counter()-tn:.1f}s')
    return masses, omegas, consts, fields, time.perf_counter()-t0


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    masses, omegas, consts, fields, sec = build(N)
    res = dict(date=datetime.datetime.now().astimezone().isoformat(), command=[sys.executable, *sys.argv],
               environment=dict(python=platform.python_version(), sympy=s.__version__),
               masses={k: str(v) for k, v in masses.items()}, omegas={k: str(v) for k, v in omegas.items()},
               constants=consts, output=OUT, seconds=sec,
               constants_note='C<word> = regularised value of the word at the horizon (constant term of its expansion in u=x-1, ell=log u)')
    (HERE/f'pert_order{N}.json').write_text(json.dumps(res, indent=1)+'\n')
    store = {k: [{','.join(map(str, w))+'|'+str(cmon_expr(cm)): str(s.factor(DOMR.to_sympy(c))) for (w, cm), c in linearize(e).items()} for e in v]
             for k, v in fields.items()}
    (HERE/f'pert_fields_order{N}.json').write_text(json.dumps(store, indent=1)+'\n')
