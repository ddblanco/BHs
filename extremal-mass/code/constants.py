"""Fix the alpha^2 integration constants and extract M2, Omega2 (controls).

Expansions are exact: at infinity in z=1/x; at the horizon in u=x-1 with
ell=log(u) kept symbolic.  Li2(1/x) = pi^2/6 + L log(1+u) - Li2(u/(1+u)).
"""
from pathlib import Path
import datetime, json, platform, sys, time
import sympy as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
x, z, u, ell = s.symbols('x z u ell', positive=True)
r = s.Symbol('r', positive=True)
Ls, Li2x = s.Symbol('L'), s.Symbol('Li2x')
K1, K2, K3, X2, cZ, cV = s.symbols('K1 K2 K3 X2 cZ cV')
KS = {'p': 1, 'K1': K1, 'K2': K2, 'K3': K3}
OUT = []
NINF, NHOR = 6, 4


def say(t): print(t, flush=True); OUT.append(t)


def words_expr(terms):
    tab = {'': 1, '1': Ls, '0,1': Li2x, '1,1': Ls**2/2+Li2x}
    return sum(s.sympify(c, locals={'x': x})*tab[w] for w, c in terms.items())


def field(name, data):
    return sum(KS[k]*words_expr(v) for k, v in data['fields'][name].items())


def at_inf(e, n=NINF):
    sub = {Ls: s.series(s.log(1-z), z, 0, n+2).removeO(),
           Li2x: sum(z**k/s.Integer(k)**2 for k in range(1, n+2)), x: 1/z}
    return s.expand(s.series(s.expand(e.subs(sub)), z, 0, n).removeO())


def at_hor(e, n=NHOR):
    lg = s.series(s.log(1+u), u, 0, n+6).removeO()
    Lh = ell-lg
    li = s.pi**2/6+Lh*lg-sum((u/(1+u))**k/s.Integer(k)**2 for k in range(1, n+6))
    ex = s.expand(e.subs({Ls: Lh, Li2x: li}).subs(x, 1+u))
    return s.expand(s.series(ex, u, 0, n).removeO())


def coeffs_u(ser, lo, hi):
    """dict (power of u, power of ell) -> coefficient."""
    out = {}
    p = s.Poly(s.expand(ser*u**(-lo)), u, ell)
    for (a, b), c in p.terms():
        out[(a+lo, b)] = c
    return out


def main():
    t0 = time.perf_counter()
    data = json.loads((HERE/'second_order_fields.json').read_text())
    h2, f2, Z, v2 = (field(n, data) for n in ('h2', 'f2', 'Z', 'v2'))
    Z = Z+cZ; v2 = v2+cV
    src = json.loads((Path(__file__).resolve().parent/'extremal_perturbation.json').read_text())
    P = lambda e: s.sympify(e, locals={'r': r})
    to_x = lambda e: s.expand(s.sympify(e).subs(s.log(1-1/r**2), Ls)).subs(r, s.sqrt(x))
    b0, f0, h0, v0 = [to_x(P(e)) for e in src['base']]; v0 = v0/s.sqrt(2)
    b1, f1, h1, v1 = [to_x(P(e)) for e in src['delta']]; v1 = s.expand(v1/s.sqrt(2))
    sol = {}
    # ---- infinity ----
    hi = at_inf(h2)
    eqs = [hi.coeff(z, k) for k in (-2, -1, 0)]
    sK2 = s.solve(eqs[0], K2)[0]; sol[K2] = sK2
    assert all(s.simplify(e.subs(K2, sK2)) == 0 for e in eqs)
    say('PASS h2 -> 0 at infinity with one constant: K2 = %s' % sK2)
    fi = at_inf(f2.subs(sol))
    assert fi.coeff(z, 0) == 0 and all(fi.coeff(z, k) == 0 for k in (-1, -2))
    sol[cZ] = s.solve(at_inf(Z.subs(sol)).coeff(z, 0), cZ)[0]
    sol[cV] = s.solve(at_inf(v2.subs(sol)).coeff(z, 0), cV)[0]
    say('PASS f2 -> 0 at infinity; b(oo)=1 and w(oo)=0 fix the additive constants')
    # ---- horizon: h2 finite ----
    hh = at_hor(h2.subs(sol), 2)
    sing = coeffs_u(hh, -3, 2)
    bad = [c for (a, b), c in sing.items() if a < 0 or (a == 0 and b > 0)]
    sK1 = s.solve(bad[0], K1)[0] if bad and bad[0].has(K1) else None
    for c in bad:
        if c.has(K1): sK1 = s.solve(c, K1)[0]; break
    sol[K1] = sK1
    assert all(s.simplify(c.subs(sol)) == 0 for c in bad)
    say('PASS h2 finite at the horizon (all u^-n and ell terms cancel) with K1 = %s' % s.nsimplify(sK1))
    # ---- double zero of f at x_H = 1 + X1 a + X2 a^2, X1 = 2*(8/3) ----
    X1 = s.Rational(16, 3)
    fh2 = at_hor(f2.subs(sol), 3)
    c2u = coeffs_u(fh2, -3, 3)
    assert all(s.simplify(c) == 0 for (a, b), c in c2u.items() if a < 1 and (a < 0 or b > 0)), 'f2 singular at horizon'
    f20 = c2u.get((0, 0), 0); f21 = c2u.get((1, 0), 0)
    assert all(s.simplify(c2u.get((1, b), 0)) == 0 for b in (1, 2, 3)), 'f2 has u log u at horizon'
    df = lambda e, k: s.diff(e, x, k)
    f1h = coeffs_u(at_hor(f1, 4), 0, 4)
    f1x1, f1x2 = f1h.get((1, 0), 0), 2*f1h.get((2, 0), 0)
    assert all(f1h.get((a, b), 0) == 0 for a in (0, 1, 2) for b in (1, 2, 3))
    f0xx, f0xxx = df(f0, 2).subs(x, 1), df(f0, 3).subs(x, 1)
    assert f0.subs(x, 1) == 0 and df(f0, 1).subs(x, 1) == 0
    assert s.simplify(f1h.get((0, 0), 0)+0) == 0
    # order alpha^1 control: f1_x(1) + X1 f0_xx(1) = 0
    assert s.simplify(f1x1+X1*f0xx) == 0
    say('PASS first-order double zero reproduced: x_H = 1 + (16/3) alpha')
    cond0 = s.expand(f20+X1*f1x1+f0xx*X1**2/2)
    cond1 = s.expand(f21+X1*f1x2+X2*f0xx+f0xxx*X1**2/2)
    s3 = s.solve([cond0, cond1], [K3, X2], dict=True)
    assert len(s3) == 1
    sol.update(s3[0])
    say('DERIVED K3 = %s ; X2 = %s' % (s.nsimplify(sol[K3]), s.nsimplify(sol[X2])))
    sol = {k: s.nsimplify(s.simplify(v.subs(sol))) for k, v in sol.items()}
    sol = {k: s.simplify(v.subs(sol)) for k, v in sol.items()}
    # ---- b double zero at the same point (independent control) ----
    b2 = b0*(Z.subs(sol))
    bh = coeffs_u(at_hor(b2, 3), -3, 3)
    b1h = coeffs_u(at_hor(b1, 4), 0, 4)
    b0xx, b0xxx = df(b0, 2).subs(x, 1), df(b0, 3).subs(x, 1)
    bc0 = s.simplify(bh.get((0, 0), 0)+X1*b1h.get((1, 0), 0)+b0xx*X1**2/2)
    bc1 = s.simplify(bh.get((1, 0), 0)+X1*2*b1h.get((2, 0), 0)+sol[X2]*b0xx+b0xxx*X1**2/2)
    say('CHECK b double zero at the same x_H: residuals %s, %s' % (bc0, bc1))
    assert bc0 == 0 and bc1 == 0
    # ---- mass ----
    zi = at_inf(b2.subs(sol))
    U2 = s.simplify(zi.coeff(z, 1))
    M2 = s.nsimplify(s.simplify(-3*s.pi*U2/8))
    say('DERIVED M2 = %s  (expected -472*pi/105 from c2=-236/105 pi^(5/3))' % M2)
    assert s.simplify(M2+472*s.pi/105) == 0
    # ---- Omega_H ----
    vh = coeffs_u(at_hor(v2.subs(sol), 2), -3, 2)
    assert all(s.simplify(c) == 0 for (a, b), c in vh.items() if a < 0 or (a == 0 and b > 0))
    v1h = coeffs_u(at_hor(v1, 3), 0, 3)
    Om2 = s.simplify(vh.get((0, 0), 0)+X1*v1h.get((1, 0), 0)+sol[X2]*df(v0, 1).subs(x, 1)+X1**2/2*df(v0, 2).subs(x, 1))
    say('DERIVED Omega2/sqrt2 = %s (expected 944/315 from 3 omega = mu - y mu\')' % Om2)
    assert s.simplify(Om2-s.Rational(944, 315)) == 0
    say('PASS second-order extremal exterior reproduces c2 via ADM mass AND via Omega_H')
    json.dump(dict(date=datetime.datetime.now().astimezone().isoformat(), command=[sys.executable, *sys.argv],
                   environment=dict(python=platform.python_version(), sympy=s.__version__),
                   constants={str(k): str(v) for k, v in sol.items()}, M2=str(M2), Omega2_over_sqrt2=str(Om2),
                   output=OUT, seconds=time.perf_counter()-t0), open(HERE/'constants.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
