"""Exact alpha^2 master equation for h2 on the EXTREMAL fixed-J branch (k=1).

Reuses the reduced system of work/analytic_metric/second_order (angular
momentum eliminated, b'/b solved from the constraint) but expands around the
extremal Myers-Perry exterior with the verified first-order fixed-J profiles of
work/exact_resume_20261008/extremal_perturbation.json.  Fixed J <=> fixed C2,
asserted below by reproducing the first-order profiles with constant C2.

Arithmetic: truncated alpha series over QQ(r)[ell], ell=log(1-1/r^2).
"""
from pathlib import Path
import datetime, json, platform, sys, time
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reduced_system import s, r, alpha, reduced_equations, f, h, p, q, t, u, v, C2
from radial_series import compile_exact
from sympy.polys.rings import ring

DOMAIN = s.QQ.frac_field(r)
RING, ELL = ring('ell', DOMAIN)
ell = s.Symbol('ell')
LOG = s.log(1-1/r**2)
LOGP = 2/(r*(r*r-1))
NORD = 3
C2VAL = 32


def pol(expr):
    expr = s.sympify(expr).subs(LOG, ell)
    assert not expr.atoms(s.Float)
    return RING.from_expr(expr)


class Series:
    def __init__(self, *c):
        self.c = tuple((c[i] if isinstance(c[i], type(ELL)) else pol(c[i])) if i < len(c) else RING.zero
                       for i in range(NORD))
    @staticmethod
    def coerce(x): return x if isinstance(x, Series) else Series(x)
    def __add__(self, x):
        x = self.coerce(x); return Series(*(a+b for a, b in zip(self.c, x.c)))
    __radd__ = __add__
    def __neg__(self): return Series(*(-a for a in self.c))
    def __sub__(self, x): return self+-self.coerce(x)
    def __rsub__(self, x): return self.coerce(x)+-self
    def __mul__(self, x):
        x = self.coerce(x)
        return Series(*(sum((self.c[i]*x.c[n-i] for i in range(n+1)), RING.zero) for n in range(NORD)))
    __rmul__ = __mul__
    def inverse(self):
        a = self.c[0]; assert a and a.degree() == 0
        ai = RING.one.exquo(a); out = [ai]
        for n in range(1, NORD):
            out.append(-ai*sum((self.c[i]*out[n-i] for i in range(1, n+1)), RING.zero))
        return Series(*out)
    def __truediv__(self, x): return self*self.coerce(x).inverse()
    def __rtruediv__(self, x): return self.coerce(x)*self.inverse()
    def __pow__(self, n):
        n = int(n)
        if n < 0: return self.inverse()**(-n)
        out = Series(1)
        for _ in range(n): out = out*self
        return out


def derivative(value):
    out = RING.zero
    for (degree,), coef in value.terms():
        out += pol(s.diff(DOMAIN.to_sympy(coef), r))*ELL**degree
    return out+value.diff(ELL)*pol(LOGP)


def make_rhs():
    eb, ef, eh = reduced_equations()
    U = s.factor(-ef.subs(u, 0)/s.diff(ef, u))
    rhs_coeff = [eb.subs({p: 0, t: 0}), s.diff(eb, p), s.diff(eb, t),
                 eh.subs({p: 0, t: 0, v: 0}), s.diff(eh, p), s.diff(eh, t), s.diff(eh, v)]
    assert all(not e.has(p, t, v) for e in rhs_coeff)
    eval_u = compile_exact((r, f, h, q, C2, alpha), [U, s.diff(U, r), s.diff(U, f), s.diff(U, h), s.diff(U, q)])
    eval_e = compile_exact((r, f, h, q, u, C2, alpha), rhs_coeff)
    def rhs(rv, fv, hv, qv, cv, av):
        uv, ur, uf, uh, uq = eval_u(rv, fv, hv, qv, cv, av)
        ec, ep, et, hc, hp, ht, hvcoef = eval_e(rv, fv, hv, qv, uv, cv, av)
        hc += hvcoef*(ur+uh*qv); hp += hvcoef*uf; ht += hvcoef*uq
        det = ep*ht-et*hp
        return (et*hc-ec*ht)/det, (ec*hp-ep*hc)/det, uv
    vacuum = tuple(s.factor(e) for e in rhs(r, f, h, q, C2, s.Integer(0))[:2])
    return rhs, vacuum, s.factor(U.subs(alpha, 0)), U


def profiles():
    src = json.loads((Path(__file__).resolve().parent/'extremal_perturbation.json').read_text())
    base = [s.sympify(e, locals={'r': r}) for e in src['base']]
    first = [s.sympify(e, locals={'r': r}) for e in src['delta']]
    return base, first


def main():
    start = time.perf_counter(); out = []
    def say(x): print(x, flush=True); out.append(x)
    rhs, vacuum, U0, U = make_rhs()
    base, first = profiles()
    fv = Series(base[1], first[1]); hv = Series(base[2], first[2])
    qv = Series(s.diff(base[2], r), s.diff(first[2], r))
    fpv, hppv, uv = rhs(Series(r), fv, hv, qv, Series(C2VAL), Series(0, 1))
    for order, prof in [(0, base), (1, first)]:
        assert fpv.c[order] == pol(s.diff(prof[1], r)), order
        assert hppv.c[order] == pol(s.diff(prof[2], r, 2)), order
    assert uv.c[0] == pol(s.diff(base[0], r)/base[0])
    assert uv.c[1] == pol(s.diff(first[0]/base[0], r))
    say('PASS reduced RHS with constant C2=32 reproduces extremal fixed-J profiles through alpha^1')
    bg = {f: base[1], h: base[2], q: s.diff(base[2], r), C2: C2VAL}
    A = [pol(s.cancel(s.diff(vacuum[0], var).subs(bg))) for var in [f, h, q]]
    B = [pol(s.cancel(s.diff(vacuum[1], var).subs(bg))) for var in [f, h, q]]
    Uc = [pol(s.cancel(s.diff(U0, var).subs(bg))) for var in [f, h, q]]
    # Linearised system at alpha^2:  f2' = A0 f2 + A1 h2 + A2 h2' + sf ;  h2'' = B0 f2 + B1 h2 + B2 h2' + sh
    binv = RING.one.exquo(B[0])
    c = derivative(B[0])*binv+A[0]
    coeff = [c*B[1]-B[0]*A[1]-derivative(B[1]), c*B[2]-B[0]*A[2]-B[1]-derivative(B[2]), -c-B[2], RING.one]
    source = -c*hppv.c[2]+B[0]*fpv.c[2]+derivative(hppv.c[2])
    say(f'master equation for h2: order 3, source log degree {source.degree()}')
    tos = lambda e: str(s.factor(e.as_expr()))
    logs = lambda P: [str(s.factor(DOMAIN.to_sympy(P[(i,)]))) for i in range(P.degree()+1)]
    data = dict(date=datetime.datetime.now().astimezone().isoformat(), command=[sys.executable, *sys.argv],
                environment=dict(python=platform.python_version(), sympy=s.__version__),
                k='1 (extremal)', C2=str(C2VAL), log=str(LOG), output=out,
                operator=[tos(e) for e in coeff], source=logs(source),
                A=[tos(e) for e in A], B=[tos(e) for e in B], U_partials=[tos(e) for e in Uc],
                source_f=logs(fpv.c[2]), source_h=logs(hppv.c[2]), source_u=logs(uv.c[2]),
                convention='h=h0+alpha h1+alpha^2 h2 at fixed areal r; f2\'=A0 f2+A1 h2+A2 h2\'+source_f; '
                           'h2\'\'=B0 f2+B1 h2+B2 h2\'+source_h; (b\'/b)_2=U_f f2+U_h h2+U_q h2\'+source_u')
    data['seconds'] = time.perf_counter()-start
    (HERE/'master_k1.json').write_text(json.dumps(data, indent=2)+'\n')
    say(f'saved master_k1.json; {data["seconds"]:.1f}s')


if __name__ == '__main__':
    main()
