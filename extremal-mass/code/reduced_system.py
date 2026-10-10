"""Eliminate the cyclic angular variable exactly before expanding in alpha.

All writes stay in this exercise. C is the reduced angular first integral,
not an assumed normalization of the physical ADM angular momentum.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[0]))
from probe import s,r,k,alpha,raw_equations,paper_profiles

HERE=Path(__file__).resolve().parent
f,h,p,q,t,u,v,C2=s.symbols('f h fp hp hpp u up C2')


def reduced_equations():
    jj,ee=raw_equations()
    K=r*r+4*alpha*(4-3*h/r**2-f)
    wp2=C2/(f*h**3*K**2)
    mapping=dict(zip(jj,[s.Integer(1),f,h,s.Integer(0),u,p,q,s.Symbol('wp'),
                        v+u*u,s.Integer(0),t,s.Integer(0)]))
    reduced=[]
    for index in [0,1,3]:
        # The b, f and h equations contain only even powers of w'.
        polynomial=s.Poly(ee[index].xreplace(mapping),mapping[jj[7]])
        assert all(power%2==0 for (power,),_ in polynomial.terms())
        value=sum(coef*wp2**(power//2) for (power,),coef in polynomial.terms())
        reduced.append(s.factor(s.cancel(value).as_numer_denom()[0]))
    return reduced


def main():
    eb,ef,eh=reduced_equations()
    assert s.degree(ef,u)==1
    U=s.factor(-ef.subs(u,0)/s.diff(ef,u))
    # u=b'/b, u'=partial_r U + U_f f' + U_h h' + U_{h'} h''.
    print('constraint solved for b_prime/b; angular momentum eliminated exactly',flush=True)
    for name,expr in [('u',U),('Eb',eb),('Eh',eh)]:
        (HERE/(name+'.txt')).write_text(str(expr)+'\n')
    U0=s.factor(U.subs(alpha,0))
    Up0=s.diff(U0,r)+s.diff(U0,f)*p+s.diff(U0,h)*q+s.diff(U0,q)*t
    eb0=s.factor(eb.subs(alpha,0))
    eh0=s.factor(eh.subs(alpha,0).subs({u:U0,v:Up0},simultaneous=True))
    print('vacuum degrees (fp,hpp):',[(s.degree(s.fraction(e)[0],p),s.degree(s.fraction(e)[0],t))
                                      for e in [eb0,eh0]],flush=True)
    (HERE/'vacuum_Eh_eliminated.txt').write_text(str(eh0)+'\n')
    # Vacuum background is checked without invoking the alpha expansion.
    base,_=paper_profiles()
    sub={f:base[1],h:base[2],p:s.diff(base[1],r),q:s.diff(base[2],r),
         t:s.diff(base[2],r,2),C2:16*k*k*(1+k*k),alpha:0}
    assert s.cancel(U.subs(sub)-s.diff(base[0],r)/base[0])==0
    assert s.cancel(eb0.subs(sub))==0
    assert s.cancel(eh0.subs(sub))==0
    print('PASS exact reduced equations and lapse reconstruction on Myers-Perry',flush=True)


if __name__=='__main__': main()
