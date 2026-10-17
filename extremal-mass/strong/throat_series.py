"""Near-horizon (throat) relations are algebraic: check that they reproduce the
exact perturbative extremal horizon g_H = r_H^2 at fixed J (code units J0 = pi*sqrt2/4).
Throat equations from reports/strong-coupling-endpoint.pdf eqs (1)-(3), a = alpha/g_H, q = h_H/g_H."""
import sympy as s
a, q, al, g = s.symbols('a q alpha g')
P = 16*a**2*q**2*(4-q) - 8*a*(2*q**2-7*q+4) + q - 2
R = 1 + 4*a*(4-3*q)
Jbar2 = q**2*(4-q)*R/8          # = 8 J^2/(pi^2 g_H^3)
N = 6
# q(a) series on the physical root q(0)=2
qs = 2
for k in range(1, N+1):
    c = s.Symbol('c')
    trial = qs + c*a**k
    eq = s.expand(P.subs(q, trial)).coeff(a, k)
    qs = qs + s.solve(eq, c)[0]*a**k
assert s.expand(s.series(P.subs(q, qs), a, 0, N+1).removeO()) == 0
Phi = s.series(Jbar2.subs(q, qs), a, 0, N+1).removeO()
# fixed J = J0: g^3 * Phi(alpha/g) = 1
gs = 1
for k in range(1, N):
    c = s.Symbol('c')
    trial = gs + c*al**k
    e = s.series(trial**3*Phi.subs(a, al/trial) - 1, al, 0, k+1).removeO()
    gs = gs + s.solve(s.expand(e).coeff(al, k), c)[0]*al**k
print('q(a) =', s.expand(qs))
print('g_H(alpha) =', s.expand(gs))
target = 1 + s.Rational(16,3)*al + s.Rational(64,9)*al**2 - s.Rational(83968,81)*al**3 - 8192*al**4
print('difference from perturbative r_H^2 through alpha^4:', s.expand(gs - target).as_poly(al).truncate if False else s.expand(s.series(gs-target, al, 0, 5).removeO()))
