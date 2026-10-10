"""Exact hyperlogarithm algebra for the extremal alpha^2 quadratures.

Functions are finite sums  sum_w R_w(x) G_w(x)  with R_w in QQ(x).
Words are tuples of letters in {0, 1, -1/3}, outermost letter first.
  G_()        = 1
  G_(a,)      = log(1 - a/x)  for a != 0     (innermost, regularised at x=oo)
  G_(0,)      = log x                         (innermost, only if forced)
  G_(a,)+w    = Integral_oo^x G_w(t) dt/(t-a) (outer letters, plain forms)
Every word whose innermost letter is nonzero vanishes at x=oo, and the outer
integrals converge.  Words with innermost letter 0 are accepted but flagged.
"""
from collections import defaultdict
import sympy as s
from sympy.polys.rings import PolyElement as _PE
from sympy.polys.heuristicgcd import heugcd as _heugcd
from sympy.polys.polyerrors import HeuristicGCDFailed as _HGF
from sympy.polys.euclidtools import dmp_inner_gcd as _dgcd


def _gcd_ZZ_fallback(f, g):
    """Sparse-ring heugcd has no fallback in SymPy; use the dense exact GCD when it fails."""
    try:
        return _heugcd(f, g)
    except _HGF:
        R = f.ring
        h, cff, cfg = _dgcd(f.to_dense(), g.to_dense(), R.ngens-1, R.domain)
        return R.from_dense(h), R.from_dense(cff), R.from_dense(cfg)


_PE._gcd_ZZ = _gcd_ZZ_fallback
from sympy.integrals.rationaltools import ratint_ratpart

x = s.Symbol('x', positive=True)
LETTERS = (s.Integer(0), s.Integer(1), s.Rational(-1, 3))
DOM = s.QQ.frac_field(x)
ONE = DOM.one
ZERO = DOM.zero


def R(e):
    """Coerce a sympy expression to an element of QQ(x)."""
    return DOM.from_sympy(s.sympify(e))


def E(c):
    return DOM.to_sympy(c)


class F:
    def __init__(self, terms=None):
        self.t = {}
        for w, c in (terms or {}).items():
            c = c if isinstance(c, type(ONE)) else R(c)
            if c: self.t[tuple(w)] = c

    @staticmethod
    def const(c): return F({(): c})

    def __add__(self, o):
        o = o if isinstance(o, F) else F.const(o)
        out = dict(self.t)
        for w, c in o.t.items():
            out[w] = out.get(w, ZERO)+c
        return F(out)
    __radd__ = __add__
    def __neg__(self): return F({w: -c for w, c in self.t.items()})
    def __sub__(self, o): return self+(-(o if isinstance(o, F) else F.const(o)))
    def __rsub__(self, o): return F.const(o)-self if not isinstance(o, F) else o-self

    def scale(self, c):
        c = c if isinstance(c, type(ONE)) else R(c)
        return F({w: c*v for w, v in self.t.items()})

    def __mul__(self, c):
        if isinstance(c, F):
            assert set(c.t) <= {()}, 'products of transcendental functions: use shuffle-free product()'
            c = c.t.get((), ZERO)
        return self.scale(c)
    __rmul__ = __mul__

    def weight(self): return max((len(w) for w in self.t), default=0)

    def iszero(self): return not self.t

    def words(self): return sorted(self.t, key=lambda w: (len(w), [str(a) for a in w]))


def dword(w):
    """Derivative of G_w as an F."""
    if not w: return F()
    a = w[0]
    if len(w) == 1:
        return F({(): R(1/x) if a == 0 else R(a/(x*(x-a)))})
    return F({w[1:]: R(1/(x-a))})


def deriv(f):
    out = F()
    for w, c in f.t.items():
        cd = R(s.diff(E(c), x))
        out = out+F({w: cd})+dword(w).scale(c)
    return out


def hermite(c):
    """c = Q' + sum_a res_a/(x-a); returns Q (QQ(x)) and {a: res_a}. Asserts letters suffice."""
    e = s.cancel(E(c))
    num, den = s.fraction(e)
    P, Dp = s.Poly(num, x), s.Poly(den, x)
    quo, rem = s.div(P, Dp)
    A, B = ratint_ratpart(rem, Dp, x)
    Q = s.cancel(A+s.integrate(quo.as_expr(), x))
    B = s.cancel(B)
    res = {}
    if B != 0:
        nb, db = s.fraction(B)
        for a in LETTERS:
            r_a = s.cancel((B*(x-a))).subs(x, a)
            if r_a != 0: res[a] = r_a
        left = s.cancel(B-sum(v/(x-a) for a, v in res.items()))
        assert left == 0, f'square-free part has poles outside the alphabet: {s.factor(B)}'
    assert s.cancel(s.diff(Q, x)+sum(v/(x-a) for a, v in res.items())-e) == 0
    return R(Q), res


def integrate(f):
    """An antiderivative of f inside the class (no free constant added)."""
    work = defaultdict(lambda: ZERO)
    for w, c in f.t.items(): work[w] += c
    out = F()
    while work:
        n = max(len(w) for w in work)
        layer = [w for w in work if len(w) == n]
        for w in layer:
            c = work.pop(w)
            if not c: continue
            Q, res = hermite(c)
            if Q: out = out+F({w: Q})
            for a, v in res.items():
                if not w:
                    if a == 0: out = out+F({(s.Integer(0),): v})
                    else:
                        # log(x-a) = log(1-a/x) + log x
                        out = out+F({(a,): v})+F({(s.Integer(0),): v})
                else:
                    out = out+F({(a,)+w: v})
            if Q and w:
                for ww, cc in dword(w).t.items():
                    work[ww] += -Q*cc
    out = F(out.t)
    assert deriv(out).t == f.t or (deriv(out)-f).iszero(), 'integration identity failed'
    return out


_PROD = {}


def word_product(w, v):
    """G_w * G_v as a linear F (constant coefficients); both vanish at infinity."""
    if not w: return F({v: 1})
    if not v: return F({w: 1})
    key = (w, v) if (len(w), [str(a) for a in w]) <= (len(v), [str(a) for a in v]) else (v, w)
    if key in _PROD: return _PROD[key]
    out = integrate(mulF(dword(w), F({v: 1}))+mulF(F({w: 1}), dword(v)))
    for ww, c in out.t.items():
        assert ww and c.numer.is_ground and c.denom.is_ground, 'product left non-constant coefficient'
    _PROD[key] = out
    return out


def mulF(a, b):
    out = F()
    for w, c in a.t.items():
        for v, d in b.t.items():
            out = out+word_product(w, v).scale(c*d)
    return out


def Lpow(j):
    """L^j, L=log(1-1/x), as an F vanishing at infinity for j>=1."""
    if j == 0: return F.const(1)
    if j == 1: return F({(s.Integer(1),): 1})
    prev = Lpow(j-1)
    return integrate(prev.scale(R(j/(x*(x-1)))))


def from_Lpoly(coeffs):
    out = F()
    for j, c in enumerate(coeffs): out = out+Lpow(j).scale(R(c))
    return out
