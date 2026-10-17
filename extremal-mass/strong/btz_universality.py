"""Strong-coupling extremal entropy from the exact throat, and the BTZ-universality mass.

Pure-Lovelock universality (arXiv:1201.4994; D=5 EGB is the N=2 derivative-degenerate case):
the r << sqrt(alpha) core of static EGB is BTZ with l^2 = 4 alpha, 1/(4 G3) = 3 pi alpha
(derived: mu = r_h^2 + 2 alpha, Delta M = 3 pi r_h^2/8, S = 6 pi^2 alpha r_h, T = r_h/(8 pi alpha)).
Extremal BTZ: Delta M = 4 G3 S^2/(4 pi^2 l^2) = S^2/(48 pi^3 alpha^2).
Here: exact series of S_ext in t = (2a)^(-1/2), a = alpha/g_H, at fixed J; prediction for m(s)."""
import sympy as s
t, q, a = s.symbols('t q a', positive=True)
alpha, J = s.symbols('alpha J', positive=True)
N = 6
# q(t) from P(a,q) = 0 with a = 1/(2 t^2), physical root q -> 0 as t -> 0
P = 16*a**2*q**2*(4-q)-8*a*(2*q**2-7*q+4)+q-2
Pt = s.expand(P.subs(a, 1/(2*t**2))*t**4)
qs = 0
cs = s.symbols('c1:%d' % (N+2))
qser = sum(cs[k]*t**(k+1) for k in range(N+1))
eqs = s.Poly(s.expand(Pt.subs(q, qser)), t).all_coeffs()[::-1]
sol = {}
for k in range(len(eqs)):
    e = s.expand(eqs[k].subs(sol))
    if e == 0: continue
    unk = [c for c in cs if e.has(c) and c not in sol]
    if not unk: 
        assert e == 0, e
        continue
    r_ = s.solve(e, unk[0])
    r_ = [v for v in r_ if v != 0 and (unk[0] != cs[0] or v > 0)] or r_
    sol[unk[0]] = r_[0]
    if len(sol) == N+1: break
qt = s.expand(qser.subs(sol))
print('q(t) =', qt)
R = 1+4*a*(4-3*q)
Jb2 = s.series((q**2*(4-q)*R/8).subs(a, 1/(2*t**2)).subs(q, qt), t, 0, N).removeO()
print('Jbar2(t) =', s.expand(Jb2))
# g_H from 8 J^2/(pi^2 g^3) = Jbar2 ;  alpha = a g = g/(2 t^2)
g = (8*J**2/(s.pi**2*Jb2))**s.Rational(1, 3)
h = q*g
Swald = s.pi**2/2*g*s.sqrt(h)*(1+2*(g/(2*t**2))*(8/g-2*h/g**2))
alph = g/(2*t**2)
ratio = s.simplify(Swald.subs(q, qt)**2/(48*s.pi**3*alph**s.Rational(3, 2)*J))
rs = s.series(ratio, t, 0, 3).removeO()
print('S^2/(48 pi^3 alpha^(3/2) J) =', s.simplify(rs))
assert s.simplify(rs.subs(t, 0)-s.Rational(4, 3)) == 0
print('PASS: S_ext^2 -> (4/3) 48 pi^3 alpha^(3/2) J, i.e. extremal-BTZ Delta M = 4 J/(3 sqrt(alpha))')
