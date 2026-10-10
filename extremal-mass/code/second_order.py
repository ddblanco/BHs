"""Extremal fixed-J second-order exterior (k=1) as exact hyperlog sums.

h2 = Hp + K1*y3 + K2*y2 + K3*y1 solves the factorised master equation.
f2, Z=b2/b0 and v2=w2/sqrt(2) follow from the reduced system by algebra and
quadratures.  Constants are fixed by:
  h2 -> 0 at infinity (areal asymptotics), h2 finite at the horizon,
  common double zero of f at x_H = 1 + X1 alpha + X2 alpha^2,
  b(oo)=1, w(oo)=0, constant C2 (fixed J).
Outputs: mass and Omega_H at alpha^2, as controls against c2 = -236/105 pi^(5/3).
"""
from pathlib import Path
import datetime, json, platform, sys, time
import sympy as s
import mpmath as mp
from hyperlog import x, F, R, E, deriv, integrate, from_Lpoly, Lpow

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
r = s.Symbol('r', positive=True)
Ls = s.Symbol('L')                  # L = log(1-1/x) = log(1-1/r^2)
LOGR = s.log(1-1/r**2)
OUT = []


def say(t):
    print(t, flush=True); OUT.append(t)


def ev(e):
    """Even function of r -> rational function of x (asserted even)."""
    e = s.cancel(s.sympify(e))
    assert s.cancel(e-e.subs(r, -r)) == 0, 'not even'
    return s.factor(s.cancel(e.subs(r, s.sqrt(x))))


def Lpoly_r(expr):
    """Expression in r with log(1-1/r^2) -> list of L-coefficients in r."""
    p = s.Poly(s.expand(s.sympify(expr).subs(LOGR, Ls)), Ls)
    return [s.cancel(p.coeff_monomial(Ls**j)) for j in range(p.degree()+1)]


def dr_Lpoly(coeffs):
    """d/dr of sum c_j(r) L^j, L'=2/(r(r^2-1))."""
    n = len(coeffs)
    out = [s.Integer(0)]*n
    for j, c in enumerate(coeffs):
        out[j] += s.diff(c, r)
        if j: out[j-1] += j*c*2/(r*(r*r-1))
    return [s.cancel(c) for c in out]


def mul_Lpoly(a, b):
    out = [s.Integer(0)]*(len(a)+len(b)-1)
    for i, p in enumerate(a):
        for j, q in enumerate(b): out[i+j] += p*q
    return [s.cancel(c) for c in out]


def F_of(coeffs_r, odd=False):
    """sum c_j(r) L^j -> F in x.  If odd, divide by 2r first (d/dr -> d/dx)."""
    cs = [ev(c/(2*r)) if odd else ev(c) for c in coeffs_r]
    return from_Lpoly(cs)


class Lin:
    """Affine combination p + sum_k K_k * comp_k of F objects."""
    KEYS = ('p', 'K1', 'K2', 'K3')
    def __init__(self, d): self.d = {k: d.get(k, F()) for k in self.KEYS}
    def __add__(self, o): return Lin({k: self.d[k]+o.d[k] for k in self.KEYS})
    def __sub__(self, o): return Lin({k: self.d[k]-o.d[k] for k in self.KEYS})
    def scale(self, c): return Lin({k: self.d[k].scale(R(c)) for k in self.KEYS})
    def add_p(self, f): return Lin({**self.d, 'p': self.d['p']+f})
    def deriv(self): return Lin({k: deriv(v) for k, v in self.d.items()})
    def integrate(self): return Lin({k: integrate(v) for k, v in self.d.items()})


# ---------- closed forms of the words that occur (weight <= 3, letters 0,1) ----------
LI2 = s.Function('Li2')
def word_expr(w, Lsym, Li2sym):
    one, zero = s.Integer(1), s.Integer(0)
    table = {(): 1, (one,): Lsym, (zero, one): Li2sym, (one, one): Lsym**2/2+Li2sym}
    if w not in table: raise KeyError(w)
    return table[w]


def F_expr(f, Lsym=Ls, Li2sym=s.Symbol('Li2x')):
    return sum(E(c)*word_expr(w, Lsym, Li2sym) for w, c in f.t.items())


def main():
    t0 = time.perf_counter()
    d = json.loads((HERE/'master_k1.json').read_text())
    P = lambda e: s.sympify(e, locals={'r': r})
    src = json.loads((Path(__file__).resolve().parent/'extremal_perturbation.json').read_text())
    base = [P(e) for e in src['base']]
    first = [P(e) for e in src['delta']]
    base[3] = s.cancel(base[3]/s.sqrt(2)); first[3] = s.cancel(s.expand(first[3]/s.sqrt(2)))
    b0, f0, h0, v0 = base
    b1, f1, h1, v1 = [Lpoly_r(e) for e in first]

    # ---- h2: master equation in x (factorised) ----
    S = [ev(P(e)/(8*r**3)) for e in d['source']]
    y1 = 1/x; y2 = (x*x-x-1)/2; z2 = (x-1)*(3*x+1)/(2*x); n3 = 3*(3*x**2+1)/((x-1)**5*(3*x+1))
    y3 = (15*x**2-18*x+7)/(20*x*(x-1)**3)
    W1 = integrate(from_Lpoly(S).scale(R(1/n3)))
    Hp = integrate(integrate(W1.scale(R(n3/z2))).scale(R(z2/y1))).scale(R(y1))
    # verify the original (unfactorised) master operator on Hp and the homogeneous solutions
    op = [P(e) for e in d['operator']]
    px = [ev(op[0]/(8*r**3)), ev(op[1]/(4*r*r)+op[2]/(4*r**3)), ev(s.Rational(3, 2)/r**2+op[2]/(2*r)), 1]
    Hd = [Hp, deriv(Hp)]; Hd += [deriv(Hd[-1]), deriv(deriv(Hd[-1]))]
    resid = from_Lpoly(S).scale(R(-1))
    for i, c in enumerate(px): resid = resid+Hd[i].scale(R(c))
    assert resid.iszero(), resid.t
    for y in (y1, y2, y3):
        assert s.cancel(sum(c*s.diff(y, x, i) for i, c in enumerate(px))) == 0
    say('PASS particular h2 (hyperlog, weight %d) and y1,y2,y3 solve the k=1 master equation exactly' % Hp.weight())
    h2 = Lin({'p': Hp, 'K1': F.const(y3), 'K2': F.const(y2), 'K3': F.const(y1)})
    hx = h2.deriv(); hxx = hx.deriv()

    # ---- f2 from the h2'' equation:  h_rr = B0 f2 + B1 h2 + B2 h_r + sh ----
    A = [P(e) for e in d['A']]; B = [P(e) for e in d['B']]; U = [P(e) for e in d['U_partials']]
    sh = F_of([P(e) for e in d['source_h']])
    sf = F_of([P(e) for e in d['source_f']], odd=True)        # sf/(2r)
    su = [P(e) for e in d['source_u']]
    inv = 1/B[0]
    f2 = (hxx.scale(4*x)+hx.scale(2)-h2.scale(ev(B[1]))-hx.scale(ev(2*r*B[2]))).add_p(-sh).scale(ev(inv))
    # check the f' equation: f2_x = A0/(2r) f2 + A1/(2r) h2 + A2 h_x + sf/(2r)
    chk = f2.deriv()-f2.scale(ev(A[0]/(2*r)))-h2.scale(ev(A[1]/(2*r)))-hx.scale(ev(A[2]))
    chk = chk.add_p(-sf)
    assert all(v.iszero() for v in chk.d.values())
    say('PASS f2 reconstructed; first-order f-equation satisfied identically (all components)')

    # ---- Z = b2/b0 :  Z_r = u2 + beta1*beta1_r ----
    beta1 = Lpoly_r(s.cancel(first[0]/b0))
    bb = mul_Lpoly(beta1, dr_Lpoly(beta1))
    u2 = f2.scale(ev(U[0]/(2*r)))+h2.scale(ev(U[1]/(2*r)))+hx.scale(ev(U[2]))
    Zx = u2.add_p(F_of(su, odd=True)+F_of(bb, odd=True))

    # ---- v2 = w2/sqrt2 : v_r = v0_r * [alpha^2 coefficient of the product] ----
    phi1 = Lpoly_r(s.cancel(first[1]/f0)); eta1 = Lpoly_r(s.cancel(first[2]/h0))
    q0 = s.cancel(4*(4-3*h0/r**2-f0)/r**2)
    q1 = [s.cancel(-4*(3*a/r**2+b)/r**2) for a, b in zip(
        Lpoly_r(first[2])+[0]*(3-len(Lpoly_r(first[2]))), Lpoly_r(first[1])+[0]*(3-len(Lpoly_r(first[1]))))]
    pad = lambda c: (c+[s.Integer(0)]*3)[:3]
    a1, bq, c1 = [s.cancel(v/2) for v in pad(beta1)], [s.cancel(-v/2) for v in pad(phi1)], [s.cancel(-3*v/2) for v in pad(eta1)]
    dq = [-q0, 0, 0]
    v0r = s.cancel(s.diff(v0, r))
    lin1 = [a1[j]+bq[j]+c1[j]+dq[j] for j in range(3)]
    assert all(s.cancel(v) == 0 for v in [a-b for a, b in zip(pad(dr_Lpoly(Lpoly_r(first[3]))), [s.cancel(v0r*c) for c in lin1])])
    say('PASS first-order angular velocity profile reproduced from the conserved-J relation')
    quad = [s.Integer(0)]*5
    for term in [mul_Lpoly(pad(beta1), pad(beta1)), mul_Lpoly(pad(phi1), pad(phi1)), mul_Lpoly(pad(eta1), pad(eta1))]:
        pass
    sq = lambda a, b: mul_Lpoly(a, b)
    pieces = [(s.Rational(-1, 8), sq(pad(beta1), pad(beta1))), (s.Rational(3, 8), sq(pad(phi1), pad(phi1))),
              (s.Rational(15, 8), sq(pad(eta1), pad(eta1)))]
    firsts = [a1, bq, c1, dq]
    for i in range(4):
        for j in range(i+1, 4): pieces.append((1, sq(firsts[i], firsts[j])))
    for c, poly in pieces:
        for j, v in enumerate(poly): quad[j] += c*v
    quad[0] += q0**2
    for j, v in enumerate(q1): quad[j] -= v
    quad = [s.cancel(v0r*v) for v in quad]
    while len(quad) > 1 and quad[-1] == 0: quad.pop()
    vx = (Zx.integrate().scale(s.Rational(1, 2)) if False else None)
    # linear part: v0r*(Z/2 - f2/(2 f0) - 3 h2/(2 h0)); Z itself needed -> integrate Zx first
    Z = Zx.integrate()
    vlin = Z.scale(s.Rational(1, 2))-f2.scale(ev(1/(2*f0)))-h2.scale(ev(3/(2*h0)))
    vx = vlin.scale(ev(v0r/(2*r))).add_p(F_of(quad, odd=True))
    v2 = vx.integrate()
    say('DERIVED Z=b2/b0 and v2=w2/sqrt2 as hyperlog sums; max weights %d, %d' % (
        max(f.weight() for f in Z.d.values()), max(f.weight() for f in v2.d.values())))
    words = sorted({w for L_ in (h2, f2, Z, v2) for f in L_.d.values() for w in f.t}, key=len)
    say('words used: '+str([tuple(map(str, w)) for w in words]))
    store = {}
    for name, L_ in [('h2', h2), ('f2', f2), ('Z', Z), ('v2', v2), ('Zx', Zx), ('vx', vx)]:
        store[name] = {k: {','.join(map(str, w)): str(s.factor(E(c))) for w, c in f.t.items()} for k, f in L_.d.items()}
    (HERE/'second_order_fields.json').write_text(json.dumps(dict(
        date=datetime.datetime.now().astimezone().isoformat(), command=[sys.executable, *sys.argv],
        environment=dict(python=platform.python_version(), sympy=s.__version__),
        convention='x=r^2; word (a1,...,an): G_(a,)=log(1-a/x) (a!=0), G_(0,)=log x, outer letters Integral_oo^x dt/(t-a); '
                   'free constants: K1,K2,K3 (h2 homogeneous), plus additive constants of Z and v2 (fixed at infinity elsewhere)',
        fields=store, output=OUT, seconds=time.perf_counter()-t0), indent=1)+'\n')
    say('saved second_order_fields.json; %.1fs' % (time.perf_counter()-t0))


if __name__ == '__main__':
    main()
