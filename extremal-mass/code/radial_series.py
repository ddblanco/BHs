"""Exact alpha^2 source for a scalar third-order radial master equation.

Arithmetic: truncated coupling series over QQ(r)[ell], ell=log(1-k^2/r^2),
at k=3/4. No fitted numerical coefficients enter the source or master operator.
"""
from reduced_system import *
from sympy.polys.rings import ring
from sympy.printing.pycode import PythonCodePrinter
import json
import time

KVAL=s.Rational(3,4)
DOMAIN=s.QQ.frac_field(r)
RING,ELL=ring('ell',DOMAIN)
ell=s.Symbol('ell')
LOG=s.log(1-KVAL**2/r**2)


def pol(expr):
    expr=s.sympify(expr).subs(LOG,ell)
    assert not expr.atoms(s.Float)
    return RING.from_expr(expr)


class Series:
    def __init__(self,a=0,b=0,c=0):
        self.c=tuple(x if isinstance(x,type(ELL)) else pol(x) for x in (a,b,c))
    @staticmethod
    def coerce(x): return x if isinstance(x,Series) else Series(x)
    def __add__(self,x):
        x=self.coerce(x)
        return Series(*(a+b for a,b in zip(self.c,x.c)))
    __radd__=__add__
    def __neg__(self): return Series(*(-v for v in self.c))
    def __sub__(self,x): return self+-self.coerce(x)
    def __rsub__(self,x): return self.coerce(x)+-self
    def __mul__(self,x):
        x=self.coerce(x)
        return Series(*(sum((self.c[i]*x.c[n-i] for i in range(n+1)),RING.zero) for n in range(3)))
    __rmul__=__mul__
    def inverse(self):
        a,b,c=self.c
        assert a and a.degree()==0
        ai=RING.one.exquo(a)
        return Series(ai,-b*ai**2,b*b*ai**3-c*ai**2)
    def __truediv__(self,x): return self*self.coerce(x).inverse()
    def __rtruediv__(self,x): return self.coerce(x)*self.inverse()
    def __pow__(self,n):
        n=int(n)
        if n<0: return self.inverse()**(-n)
        out=Series(1)
        factor=self
        while n:
            if n%2: out=out*factor
            factor=factor*factor
            n//=2
        return out


class ExactPrinter(PythonCodePrinter):
    def _print_Rational(self,x): return f'Q({x.p},{x.q})'


def compile_exact(args,values):
    return s.lambdify(args,values,modules={'Q':s.Rational},printer=ExactPrinter(),cse=True,
                      docstring_limit=0)


def derivative(value):
    out=RING.zero
    for (degree,),coef in value.terms():
        expr=DOMAIN.to_sympy(coef)
        out+=pol(s.diff(expr,r))*ELL**degree
    return out+value.diff(ELL)*pol(2*KVAL**2/(r*(r*r-KVAL**2)))


def make_rhs():
    eb,ef,eh=reduced_equations()
    U=s.factor(-ef.subs(u,0)/s.diff(ef,u))
    # Coefficients before elimination of u'. This prevents expression blowup.
    rhs_coeff=[eb.subs({p:0,t:0}),s.diff(eb,p),s.diff(eb,t),
               eh.subs({p:0,t:0,v:0}),s.diff(eh,p),s.diff(eh,t),s.diff(eh,v)]
    assert all(not e.has(p,t,v) for e in rhs_coeff)
    eval_u=compile_exact((r,f,h,q,C2,alpha),[U,s.diff(U,r),s.diff(U,f),s.diff(U,h),s.diff(U,q)])
    eval_e=compile_exact((r,f,h,q,u,C2,alpha),rhs_coeff)
    def rhs(rv,fv,hv,qv,cv,av):
        uv,ur,uf,uh,uq=eval_u(rv,fv,hv,qv,cv,av)
        ec,ep,et,hc,hp,ht,hvcoef=eval_e(rv,fv,hv,qv,uv,cv,av)
        hc+=hvcoef*(ur+uh*qv)
        hp+=hvcoef*uf
        ht+=hvcoef*uq
        det=ep*ht-et*hp
        return (et*hc-ec*ht)/det,(ec*hp-ep*hc)/det,uv
    # Only simplify the vacuum RHS, whose coefficients define the master operator.
    vacuum=tuple(s.factor(e) for e in rhs(r,f,h,q,C2,s.Integer(0))[:2])
    return rhs,vacuum,s.factor(U.subs(alpha,0))


def main():
    start=time.perf_counter()
    trial=Series(1+r,ell,ell*ell/r)
    assert (trial/trial).c==(RING.one,RING.zero,RING.zero)
    print('PASS exact truncated-series inversion',flush=True)
    rhs,vacuum,U0=make_rhs()
    base,first=paper_profiles()
    base=[s.factor(e.subs(k,KVAL)) for e in base]
    first=[s.factor(e.subs(k,KVAL)) for e in first]
    fv=Series(base[1],first[1])
    hv=Series(base[2],first[2])
    qv=Series(s.diff(base[2],r),s.diff(first[2],r))
    fpv,hppv,uv=rhs(Series(r),fv,hv,qv,Series(16*KVAL**2*(1+KVAL**2)),Series(0,1))
    for order,profiles in [(0,base),(1,first)]:
        assert fpv.c[order]==pol(s.diff(profiles[1],r))
        assert hppv.c[order]==pol(s.diff(profiles[2],r,2))
    expected_u1=s.diff(first[0]/base[0],r)
    assert uv.c[0]==pol(s.diff(base[0],r)/base[0])
    assert uv.c[1]==pol(expected_u1)
    print('PASS exact reduced RHS through alpha^1 against verified profiles',flush=True)
    bg={f:base[1],h:base[2],q:s.diff(base[2],r),C2:16*KVAL**2*(1+KVAL**2)}
    A=[pol(s.cancel(s.diff(vacuum[0],var).subs(bg))) for var in [f,h,q]]
    B=[pol(s.cancel(s.diff(vacuum[1],var).subs(bg))) for var in [f,h,q]]
    bf_inverse=RING.one.exquo(B[0])
    c=derivative(B[0])*bf_inverse+A[0]
    coeff=[c*B[1]-B[0]*A[1]-derivative(B[1]),
           c*B[2]-B[0]*A[2]-B[1]-derivative(B[2]),-c-B[2],RING.one]
    source=-c*hppv.c[2]+B[0]*fpv.c[2]+derivative(hppv.c[2])
    assert source.degree()<=2
    print('master equation constructed: derivative order 3, source log degree',source.degree(),flush=True)
    data=dict(k=str(KVAL),variable='h2: h=h0+alpha*h1+alpha^2*h2',
              log=str(LOG),operator=[str(s.factor(e.as_expr())) for e in coeff],
              source=[str(s.factor(DOMAIN.to_sympy(source[(i,)]))) for i in range(3)],
              A=[str(s.factor(e.as_expr())) for e in A],B=[str(s.factor(e.as_expr())) for e in B],
              source_f=[str(s.factor(DOMAIN.to_sympy(fpv.c[2][(i,)]))) for i in range(3)],
              source_h=[str(s.factor(DOMAIN.to_sympy(hppv.c[2][(i,)]))) for i in range(3)])
    data['source_u']=[str(s.factor(DOMAIN.to_sympy(uv.c[2][(i,)]))) for i in range(3)]
    data['U_partials']=[str(s.factor(s.diff(U0,var).subs(bg))) for var in [f,h,q]]
    (HERE/'master_equation.json').write_text(json.dumps(data,indent=2)+'\n')
    print('PASS master source saved; elapsed seconds',round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__': main()
