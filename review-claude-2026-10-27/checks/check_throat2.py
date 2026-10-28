"""Same derivation as check_throat.py (Sen entropy function on the extremal throat), with the
curvature contractions done as 25x25 matrix products instead of index loops.  Independent of
the project's code.  Run with .venv/bin/python -B.
"""
import sympy as s, time, itertools
t0 = time.time()
PASS = []
def ok(name, cond):
    assert cond, name
    PASS.append(name); print('PASS', name, flush=True)

t, rho, th, p1, p2 = s.symbols('t rho theta phi1 phi2')
v1, v2, v3, al = s.symbols('v1 v2 v3 alpha', positive=True)
e = s.symbols('e', real=True)
X = [t, rho, th, p1, p2]; n = 5
sn, cs = s.sin(th), s.cos(th)
one_forms = [s.Matrix([1, 0, 0, 0, 0]), s.Matrix([0, 1, 0, 0, 0]), s.Matrix([0, 0, 1, 0, 0]),
             s.Matrix([0, 0, 0, 1, -1]), s.Matrix([-e*rho, 0, 0, sn**2, cs**2])]
coef = [-v1*rho**2, v1/rho**2, v2, v2*sn**2*cs**2, v3]
G = s.zeros(5, 5)
for c, w in zip(coef, one_forms):
    G += c*w*w.T
G = G.applyfunc(s.expand)
Gi = G.inv().applyfunc(s.cancel)
dG = [G.applyfunc(lambda z: s.diff(z, xk)) for xk in X]
Gam = [[[s.cancel(sum(Gi[i, l]*(dG[k][l, j] + dG[j][l, k] - dG[l][j, k]) for l in range(n))/2)
         for k in range(n)] for j in range(n)] for i in range(n)]
print('Christoffels done', round(time.time()-t0), 's', flush=True)
# R^i_{jkl}
Riem = {}
for i, j, k, l in itertools.product(range(n), repeat=4):
    if l <= k:
        continue
    val = s.diff(Gam[i][l][j], X[k]) - s.diff(Gam[i][k][j], X[l]) + sum(Gam[i][k][m]*Gam[m][l][j] - Gam[i][l][m]*Gam[m][k][j] for m in range(n))
    val = s.cancel(val)
    Riem[(i, j, k, l)] = val; Riem[(i, j, l, k)] = -val
for i, j, k in itertools.product(range(n), repeat=3):
    Riem[(i, j, k, k)] = 0
print('Riemann done', round(time.time()-t0), 's', flush=True)
Ric = s.Matrix(n, n, lambda j, l: s.cancel(sum(Riem[(i, j, i, l)] for i in range(n))))
Rs = s.cancel(sum(Gi[j, l]*Ric[j, l] for j in range(n) for l in range(n)))
Rs = s.simplify(Rs)
# all-down Riemann as a 25x25 matrix  Rd[(i,j),(k,l)]
Rd = s.zeros(n*n, n*n)
for i, j, k, l in itertools.product(range(n), repeat=4):
    Rd[i*n+j, k*n+l] = s.cancel(sum(G[i, m]*Riem[(m, j, k, l)] for m in range(n)))
print('Riemann lowered', round(time.time()-t0), 's', flush=True)
GG = s.kronecker_product(Gi, Gi)                     # g^{ii'} g^{jj'}
Rup = (GG*Rd*GG).applyfunc(s.cancel)                 # R^{ij kl}
Rsq = s.simplify(sum(Rd[a_, b_]*Rup[a_, b_] for a_ in range(n*n) for b_ in range(n*n)))
Ricup = (Gi*Ric*Gi).applyfunc(s.cancel)
Ricsq = s.simplify(sum(Ric[j, l]*Ricup[j, l] for j in range(n) for l in range(n)))
LGB = s.simplify(Rs**2 - 4*Ricsq + Rsq)
print('R =', Rs); print('L_GB =', s.factor(LGB)); print('curvature done', round(time.time()-t0), 's', flush=True)
ok('R and L_GB constant on the throat', not (Rs.free_symbols | LGB.free_symbols) & {rho, th})
detg = s.factor(-G.det())
ok('-det g = v1^2 v2^2 v3 sin^2 cos^2  (so sqrt(-g) = v1 v2 sqrt(v3) sin cos)', s.simplify(detg - v1**2*v2**2*v3*sn**2*cs**2) == 0)
# angular integral of sin cos over [0, pi/2] x [0,2pi]^2 is 2 pi^2:  f = (1/16pi) 2pi^2 v1 v2 sqrt(v3) (R + alpha L_GB)
f = s.pi/8*v1*v2*s.sqrt(v3)*(Rs + al*LGB)
E1, E2, E3 = [s.cancel(s.diff(f, v)) for v in (v1, v2, v3)]
Jtot = s.cancel(s.diff(f, e))
S = s.cancel(2*s.pi*(e*Jtot - f))
# alpha = 0: extremal Myers-Perry
sol0 = s.solve([E1.subs(al, 0), E2.subs(al, 0), E3.subs(al, 0)], [v2, v3, e], dict=True)
sol0 = [d for d in sol0 if s.simplify(d[v2]) != 0 and s.simplify(d[v3]) != 0]
print('alpha=0 solutions:', sol0)
d0 = [d for d in sol0 if s.simplify(d[v2] - 4*v1) == 0 and d[e].is_positive][0]
ok('alpha=0: v2 = 4 v1, v3 = 8 v1 (manuscript app. B)', s.simplify(d0[v3] - 8*v1) == 0)
print('alpha=0: e =', d0[e])
aa = s.symbols('a', positive=True)
J0a = s.simplify(Jtot.subs(al, 0).subs(d0).subs(v1, aa**2/2)); S0a = s.simplify(S.subs(al, 0).subs(d0).subs(v1, aa**2/2))
print('alpha=0 with v2 = r_H^2 = 2a^2: J_total =', J0a, ' S =', S0a)
ok('alpha=0: J_total = 2 pi a^3 (pi a^3 per plane), S = 2 pi^2 a^3 = 2 pi J_each', s.simplify(J0a - 2*s.pi*aa**3) == 0 and s.simplify(S0a - 2*s.pi**2*aa**3) == 0)
# finite alpha: v2 = 1, a = alpha, q = v3; unknowns v1, e^2
a, q, e2 = s.symbols('a q e2', positive=True)
sub = {v2: 1, al: a, v3: q}
F = [s.expand(s.numer(s.together(E.subs(sub).subs(e, s.sqrt(e2))))) for E in (E1, E2, E3)]
Pref = 16*a**2*q**2*(4 - q) - 8*a*(2*q**2 - 7*q + 4) + q - 2
Rref = 1 + 4*a*(4 - 3*q)
vref = Rref*(3 + 4*a*(4 - q))/(2*(4 - q)*(3 + 4*a*(8 - 6*q)))
k2ref = 4*(4 - q)*vref**2/(q*Rref)
print('degrees of F_i in (v1, e2):', [(s.degree(Fi, v1), s.degree(Fi, e2)) for Fi in F])
sols = s.solve(F[:2], [v1, e2], dict=True)
print(len(sols), 'solutions of E1 = E2 = 0')
matched = False
for d in sols:
    v1s, e2s = s.factor(d[v1]), s.factor(d[e2])
    rest = s.factor(s.numer(s.together(F[2].subs({v1: v1s, e2: e2s}))))
    print('  v1 =', v1s, '\n  e^2 =', e2s, '\n  third equation ->', rest)
    rem = s.rem(s.Poly(rest, q), s.Poly(Pref, q))
    if rem == 0 or s.simplify(v1s - vref) == 0:
        matched = True
        def on_P(expr):
            num = s.numer(s.together(expr))
            return s.simplify(expr) == 0 or s.rem(s.Poly(s.expand(num), q), s.Poly(Pref, q)) == 0
        ok('v1 = v(a,q) of strong-coupling-endpoint.pdf eq. (2)  [equal modulo P(a,q) = 0]', on_P(v1s - vref))
        # the report's k is the KKR k (w = -k r), which is this e;  the manuscript's k_B is 2e
        ok('e^2 = k^2 = 4(4-q)v^2/(qR) of eq. (2)  [modulo P = 0]', on_P(e2s - k2ref))
        ok('throat curvature: R = 8 - 2q - 2/v + q k^2/(2 v^2) (report, recalled identity) with k = e, g_H = 1',
           s.simplify(Rs.subs(sub) - (8 - 2*q - 2/v1 + q*e**2/(2*v1**2))) == 0)
        ok('third stationarity equation reduces to P(a,q) = 0 of eq. (1)', rem == 0)
        Jh = s.cancel(Jtot.subs(sub).subs({v1: v1s, e: s.sqrt(e2s)}))
        Jbar2 = s.cancel(8*(Jh/2)**2/s.pi**2)
        ok('8 J_each^2/(pi^2 g_H^3) = q^2(4-q)R/8  [eq. (3)]', s.simplify(Jbar2 - q**2*(4 - q)*Rref/8) == 0)
        ok('KKR 2023 eq. (3.18) equals J_total = J1 + J2', s.simplify(Jh**2 - (s.pi/4)**2*q**2*(4 - q)*Rref) == 0)
        Sh = s.cancel(S.subs(sub).subs({v1: v1s, e: s.sqrt(e2s)}))
        SK = s.pi**2/2*s.sqrt(q)*(1 + 4*a*(4 - q))
        diff = s.numer(s.together(Sh - SK))
        ok('S_ext = Wald = KKR (3.19) on P = 0', s.simplify(diff) == 0 or s.rem(s.Poly(s.expand(diff), q), s.Poly(Pref, q)) == 0)
        D = 5*q**4 - 34*q**3 + 73*q**2 - 56*q + 16
        aK = (q - 2)/(4*(2*q**2 - 7*q + 4 - s.sqrt(D)))
        ok('KKR (3.17) is the physical root of P(a,q) = 0', s.simplify(Pref.subs(a, aK)) == 0)
if not matched:
    raise SystemExit('no branch matched')
print(f'\n{len(PASS)} assertions passed in {round(time.time()-t0)} s')
