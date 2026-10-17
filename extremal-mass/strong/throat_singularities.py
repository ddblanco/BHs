"""Branch points of the algebraic extremal-throat data g_H(alpha), at fixed J = J0.
alpha(a,q) = a g with g^3 Jbar2 = 1, Jbar2 = q^2(4-q)R/8, P(a,q)=0.
Singularities of g_H(alpha): (i) discriminant of P in q, (ii) d alpha/da = 0.
Compared with the radius of convergence of the exact Taylor series of g_H(s), s = 2 alpha."""
import cmath
import numpy as np
import sympy as s
a, q, al = s.symbols('a q alpha')
P = 16*a**2*q**2*(4-q)-8*a*(2*q**2-7*q+4)+q-2
R = 1+4*a*(4-3*q)
Jb2 = q**2*(4-q)*R/8
disc = s.factor(s.discriminant(P, q)); print('disc_q P =', disc)
dJ = s.diff(Jb2, a)+s.diff(Jb2, q)*(-s.diff(P, a)/s.diff(P, q))
crit = s.numer(s.together(3*Jb2-a*dJ))
res = s.factor(s.resultant(crit, P, q)); print('critical resultant =', res)
pts = []
for name, poly in [('disc', disc), ('crit', res)]:
    for fac, k in s.factor_list(poly)[1]:
        if fac.free_symbols != {a}: continue
        for root in np.roots([float(c) for c in s.Poly(fac, a).all_coeffs()]):
            if abs(root) < 1e-12: continue
            qs = np.roots([complex(c) for c in s.Poly(P.subs(a, root), q).all_coeffs()])
            for qq in qs:
                J2 = complex(Jb2.subs({a: root, q: qq}))
                if abs(J2) < 1e-14: continue
                for kbr in range(3):
                    g = cmath.exp((-cmath.log(J2)+2j*cmath.pi*kbr)/3)
                    pts.append((name, str(fac), complex(root), complex(qq), 2*complex(root)*g))
pts.sort(key=lambda p: abs(p[4]))
for p in [p for p in pts if 0.03<abs(p[4])<0.2]:
    print(f"{p[0]:5s} [{p[1][:28]:28s}] a={p[2]:.5f} q={p[3]:.5f}  s={p[4]:.6f} |s|={abs(p[4]):.6f}")

# exact Taylor series of g_H in alpha via undetermined coefficients on the polynomial system
N = 40
qa = [s.Integer(2)]
# q(a) series
import sympy
x = s.Symbol('x')
def series_solve(F, var, x0, N):
    """power series root var(x) of F(x, var)=0 with var(0)=x0 (simple root), via Newton doubling"""
    v = s.Integer(x0)
    Fv = s.diff(F, var)
    k = 1
    while k < N+1:
        k = min(2*k, N+1)
        num = s.series(F.subs(var, v), x, 0, k).removeO()
        den = s.series(Fv.subs(var, v), x, 0, k).removeO()
        v = s.expand(v - s.series(num/den, x, 0, k).removeO())
    return v
