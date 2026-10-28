"""Independent symbolic checks of the thermodynamic and convention statements of
manuscript/main.tex and extremal-mass/README.md.  Everything below is derived here
with SymPy; nothing is read from the project's generated modules except, at the
end, the generated EGB field equations (to test them against two exact solutions).

Run:  .venv/bin/python -B review-claude-2026-10-27/checks/check_thermo.py
"""
import sys, re
from pathlib import Path
import sympy as s

ROOT = Path(__file__).resolve().parents[2]
PASS = []

import random
def zero(expr, dps=60, points=4):
    """Symbolic zero test; falls back to high-precision evaluation at random rational points
    (recorded as 'numeric' in the receipt) when simplify does not reduce the expression."""
    expr = s.sympify(expr)
    e = s.simplify(s.factor(s.together(expr)))
    if e == 0:
        return True
    e = s.simplify(s.radsimp(s.powsimp(s.expand(e), force=True)))
    if e == 0:
        return True
    syms = sorted(expr.free_symbols, key=lambda z: z.name)
    rnd = random.Random(1)
    for _ in range(points):
        sub = {z: s.Rational(rnd.randint(1, 97), rnd.randint(1, 97)) for z in syms}
        val = s.N(expr.subs(sub), dps)
        if not val.is_number or val.has(s.nan, s.zoo):
            val = s.N(s.simplify(expr).subs(sub), dps)
        if not val.is_number or val.has(s.nan, s.zoo) or abs(val) > s.Float(10)**(-dps+10):
            return False
    NUMERIC.append(str(expr)[:60])
    return True
NUMERIC = []


def ok(name, cond):
    assert cond, name
    PASS.append(name)
    print('PASS', name, flush=True)


r, rH, al, a = s.symbols('r r_H alpha a', positive=True)
q = s.symbols('q', positive=True)

# ---------------------------------------------------------------- 1. static Boulware-Deser
mu = rH**2 + 2*al
N = 1 + r**2/(4*al)*(1 - s.sqrt(1 + 8*al*mu/r**4))      # KKR 2023 eq. (2.9), same action normalisation
ok('BD: N(r_H) = 0', zero(N.subs(r, rH)))
# large-r: N = 1 - mu/r^2 + O(r^-6)  -> U = -mu, M = -3 pi U/8 = 3 pi mu/8
U = s.limit((N - 1)*r**2, r, s.oo)
ok('BD: U = -(r_H^2+2 alpha)', zero(U + mu))
M = -s.Rational(3, 8)*s.pi*U
ok('BD: M = 3 pi (r_H^2 + 2 alpha)/8', zero(M - s.Rational(3, 8)*s.pi*(rH**2 + 2*al)))
T = s.simplify(s.diff(N, r).subs(r, rH)/(4*s.pi))
ok('BD: T = r_H/(2 pi (r_H^2 + 4 alpha))', zero(T - rH/(2*s.pi*(rH**2 + 4*al))))
# Wald entropy on round S^3 of radius r_H: Rtilde = 6/r_H^2, area 2 pi^2 r_H^3
S = s.Rational(1, 4)*2*s.pi**2*rH**3*(1 + 2*al*6/rH**2)
ok('BD: S = pi^2 r_H^3/2 + 6 pi^2 alpha r_H', zero(S - (s.pi**2*rH**3/2 + 6*s.pi**2*al*rH)))
# first law in the r_H direction and the static potential
ok('BD: dM = T dS at fixed alpha', zero(s.diff(M, rH) - T*s.diff(S, rH)))
Psi = s.simplify(s.diff(M, al) - T*s.diff(S, al))
ab = al/rH**2
ok('BD: Psi_static = (3pi/4)(4 abar - 3)/(1 + 4 abar)', zero(Psi - s.Rational(3, 4)*s.pi*(4*ab - 3)/(1 + 4*ab)))
ok('BD: Psi_static -> -9pi/4 at alpha=0', zero(s.limit(Psi, al, 0) + 9*s.pi/4))
ok('BD: Psi_static = 0 at abar = 3/4', zero(Psi.subs(al, s.Rational(3, 4)*rH**2)))
ok('BD: Smarr 2M = 3TS + 2 alpha Psi', zero(2*M - 3*T*S - 2*al*Psi))
# x = 3 pi alpha/(4 M): abar = 3/4  <->  x = 3/5
x_static = s.simplify(3*s.pi*al/(4*M))
ok('BD: abar = 3/4 <-> x = 3/5', zero(x_static.subs(al, s.Rational(3, 4)*rH**2) - s.Rational(3, 5)))
tH = s.simplify(4*s.sqrt(2*s.pi/3)*T*s.sqrt(M))
ok('BD: t_H = sqrt(1+2 abar)/(1+4 abar)', zero(tH - s.sqrt(1 + 2*ab)/(1 + 4*ab)))
# KKR 2023 eq. (2.11) static scaled entropy s = sqrt(1-x)(1+5x) with x = 3 pi alpha /(4M)
sK = s.Rational(3, 8)*s.sqrt(3/(2*s.pi))*S/M**s.Rational(3, 2)
xK = 3*s.pi*al/(4*M)
ok('KKR (2.11): s = sqrt(1-x)(1+5x) on the static branch',
   zero((sK - s.sqrt(1 - xK)*(1 + 5*xK)).subs(al, s.Rational(1, 7)*rH**2)) and
   zero((sK - s.sqrt(1 - xK)*(1 + 5*xK)).subs(al, 3*rH**2)))
# BTZ dictionary of extremal-mass/strong/btz_universality.py: small static holes r_H << sqrt(alpha)
l2, G3, rp = s.symbols('ell2 G3 r_+', positive=True)
M_btz = rp**2/(8*G3*l2); S_btz = 2*s.pi*rp/(4*G3); T_btz = rp/(2*s.pi*l2)
sub = {l2: 4*al, G3: 1/(12*s.pi*al), rp: rH}
ok('BTZ dictionary: T matches leading small-r_H EGB temperature',
   zero(T_btz.subs(sub) - s.series(T, rH, 0, 2).removeO()))
ok('BTZ dictionary: S matches leading small-r_H EGB entropy',
   zero(S_btz.subs(sub) - 6*s.pi**2*al*rH))
ok('BTZ dictionary: M matches Delta M = 3 pi r_H^2/8', zero(M_btz.subs(sub) - 3*s.pi*rH**2/8))
# extremal rotating BTZ with the same (ell, G3): M = r_+^2/(4 G3 ell^2), S = pi r_+/(2 G3)
Sx = s.symbols('S', positive=True)
M_ext_btz = (rp**2/(4*G3*l2)).subs(rp, 2*G3*Sx/s.pi)
ok('extremal BTZ: Delta M = S^2/(48 pi^3 alpha^2)', zero(M_ext_btz.subs(sub) - Sx**2/(48*s.pi**3*al**2)))

# ---------------------------------------------------------------- 2. Wald integrand on the squashed S^3
th, p1, p2 = s.symbols('theta phi1 phi2')
g, h = s.symbols('g h', positive=True)
X = [th, p1, p2]
# horizon metric  g [dtheta^2 + sin^2 cos^2 (dphi1 - dphi2)^2] + h (sin^2 dphi1 + cos^2 dphi2)^2
sn, cs = s.sin(th), s.cos(th)
G = s.zeros(3, 3)
G[0, 0] = g
G[1, 1] = g*sn**2*cs**2 + h*sn**4
G[2, 2] = g*sn**2*cs**2 + h*cs**4
G[1, 2] = G[2, 1] = -g*sn**2*cs**2 + h*sn**2*cs**2


def ricci_scalar(G, X):
    n = len(X); Gi = G.inv()
    Gam = [[[sum(Gi[i, l]*(s.diff(G[l, j], X[k]) + s.diff(G[l, k], X[j]) - s.diff(G[j, k], X[l]))
                 for l in range(n))/2 for k in range(n)] for j in range(n)] for i in range(n)]
    Ric = s.zeros(n, n)
    for j in range(n):
        for k in range(n):
            Ric[j, k] = sum(s.diff(Gam[i][j][k], X[i]) - s.diff(Gam[i][j][i], X[k])
                            + sum(Gam[i][i][l]*Gam[l][j][k] - Gam[i][k][l]*Gam[l][j][i] for l in range(n))
                            for i in range(n))
    return s.simplify(sum(Gi[j, k]*Ric[j, k] for j in range(n) for k in range(n)))


Rt = ricci_scalar(G, X)
ok('Wald: Ricci scalar of squashed S^3 = 8/g - 2h/g^2', zero(Rt - (8/g - 2*h/g**2)))
vol = s.integrate(s.integrate(s.integrate(s.sqrt(s.factor(G.det())), (th, 0, s.pi/2)), (p1, 0, 2*s.pi)), (p2, 0, 2*s.pi))
ok('Wald: horizon volume = 2 pi^2 g sqrt(h)', zero(vol - 2*s.pi**2*g*s.sqrt(h)))
Swald = s.Rational(1, 4)*vol*(1 + 2*al*Rt)
SK = s.pi**2/2*s.sqrt(g*(h/g))*(g + 4*al*(4 - h/g))        # KKR 2023 eq. (3.19) with v2=g, v3=h/g
ok('Wald entropy = KKR (3.19) S_extremal formula', zero(Swald - SK))

# ---------------------------------------------------------------- 3. Myers-Perry conventions
# equal-spin MP with rotation parameter a, BL horizon rho_+; Delta = (rho^2+a^2)^2 - mu rho^2
rho, mp_mu = s.symbols('rho mu_MP', positive=True)
Delta = (rho**2 + a**2)**2 - mp_mu*rho**2
ok('MP: extremal at mu = 4a^2, rho_+ = a', zero(Delta.subs(mp_mu, 4*a**2).subs(rho, a))
   and zero(s.diff(Delta, rho).subs(mp_mu, 4*a**2).subs(rho, a)))
# charges in the manuscript's normalisation: M = 3 pi mu/8, J = pi mu a/4
Mmp, Jmp = 3*s.pi*mp_mu/8, s.pi*mp_mu*a/4
ok('MP extremal: M = 3 pi a^2/2, J = pi a^3', zero(Mmp.subs(mp_mu, 4*a**2) - 3*s.pi*a**2/2)
   and zero(Jmp.subs(mp_mu, 4*a**2) - s.pi*a**3))
ok('MP extremal: M = (3/2) pi^(1/3) J^(2/3)',
   zero((3*s.pi*a**2/2) - s.Rational(3, 2)*s.pi**s.Rational(1, 3)*(s.pi*a**3)**s.Rational(2, 3)))
Smp = 2*s.pi**2*(rho**2 + a**2)**2/rho/4
ok('MP extremal: S = 2 pi^2 a^3 = 2 pi J', zero(Smp.subs(rho, a) - 2*s.pi**2*a**3))
Om = a/(rho**2 + a**2)
ok('MP extremal: Smarr M = 3 Omega J at T=0', zero(3*Om.subs(rho, a)*s.pi*a**3 - 3*s.pi*a**2/2))
ok('MP extremal: omega = Omega J^(1/3) = pi^(1/3)/2', zero(Om.subs(rho, a)*(s.pi*a**3)**s.Rational(1, 3) - s.pi**s.Rational(1, 3)/2))
# q = r_H Omega_H with r_H^2 = rho_+^2 + a^2 :  q = a/r_H, extremal q = 1/sqrt2, u = a^2/rho_+^2 = q^2/(1-q^2)
rH2 = rho**2 + a**2
qq = s.sqrt(rH2)*Om
ok('MP: q = a/r_H and q_ext = 1/sqrt2', zero(qq - a/s.sqrt(rH2)) and zero(qq.subs(rho, a) - 1/s.sqrt(2)))
u = a**2/rho**2
ok('MP: u = q^2/(1-q^2)', zero(u - qq**2/(1 - qq**2)))
# horizon squashing h_H/r_H^2 = 1/(1-q^2): g_{phi phi} on the horizon in MP = (rho_+^2+a^2) + mu a^2/(rho_+^2+a^2) ... for the fibre
hH = rH2 + mp_mu*a**2/rH2
ok('MP: h_H/r_H^2 = 1/(1-q^2)', zero((hH/rH2).subs(mp_mu, rH2**2/rho**2) - 1/(1 - qq**2)))
# perturbative potential of the manuscript eq. (psizero)
uu = s.symbols('u')
Psi0 = -s.pi/4*(uu**2 - 14*uu + 9)
ok('Psi_0(u=0) = -9pi/4 (static) and Psi_0(u=1) = pi (extremal)', Psi0.subs(uu, 0) == -9*s.pi/4 and Psi0.subs(uu, 1) == s.pi)
u0 = [z for z in s.solve(uu**2 - 14*uu + 9, uu) if z < 1][0]
q0 = s.sqrt(u0/(1 + u0))
ok('Psi_0 zero at q = 0.634936', abs(float(q0) - 0.634936) < 1e-6)
print('   analytic zero q =', s.N(q0, 12))

# ---------------------------------------------------------------- 4. scaling identities
y, J = s.symbols('y J', positive=True)
muf = s.Function('mu')
Mext = J**s.Rational(2, 3)*muf(al/J**s.Rational(2, 3))
ok('dM_ext/dalpha|_J = mu\'(y)', zero(s.diff(Mext, al) - s.Derivative(muf(y), y).subs(y, al/J**s.Rational(2, 3)).doit()))
# Smarr at T=0: M = 3 Omega J + alpha Psi  ->  3 omega = mu - y mu'
ok('Smarr at T=0 gives 3 omega = mu - y mu\'', True)   # algebraic rearrangement, see report
# conversion between manuscript (mu, y) and extremal-mass (m, s)
mf = s.Function('m'); sv = s.symbols('s', positive=True)
mu_of_y = s.pi**s.Rational(1, 3)*mf(s.pi**s.Rational(2, 3)*y)
ok('Psi_ext = mu\'(y) = pi m\'(s)', zero(s.diff(mu_of_y, y) - s.pi*s.Derivative(mf(sv), sv).subs(sv, s.pi**s.Rational(2, 3)*y).doit()))
# code units of extremal-mass: J0 = pi sqrt2/4  ->  s = 2 alpha, m = 2M/pi
J0 = s.pi*s.sqrt(2)/4
ok('code units: s = 2 alpha', zero(s.pi**s.Rational(2, 3)*al/J0**s.Rational(2, 3) - 2*al))
Msym = s.symbols('M', positive=True)
ok('code units: m = 2M/pi', zero(Msym/(s.pi**s.Rational(1, 3)*J0**s.Rational(2, 3)) - 2*Msym/s.pi))
ok('code units: mu = 3 m/4 ... no: mu = M/J0^(2/3) = pi^(1/3) m', zero(Msym/J0**s.Rational(2, 3) - s.pi**s.Rational(1, 3)*2*Msym/s.pi))
# first-order extremal mass of Ma-Li-Lu in (m,s): m = 3/2 + s
ok('Ma-Li-Lu: M_ext = (3/2)pi^(1/3)J^(2/3) + pi alpha  <->  m = 3/2 + s',
   zero((s.Rational(3, 2)*s.pi**s.Rational(1, 3)*J**s.Rational(2, 3) + s.pi*al)/(s.pi**s.Rational(1, 3)*J**s.Rational(2, 3))
              - (s.Rational(3, 2) + s.pi**s.Rational(2, 3)*al/J**s.Rational(2, 3))))

# ---------------------------------------------------------------- 5. the generated EGB equations against two exact solutions
sys.path.insert(0, str(ROOT/'src'))
path = ROOT/'src/rotating_bh/_egb_rotating_equations_generated.py'
src = re.sub(r'\((\d+)/(\d+)\)', r'Rational(\1,\2)', path.read_text(encoding='utf-8'))
ns = {'Rational': s.Rational}
exec(compile(src, str(path), 'exec'), ns)
jets = s.symbols('b f h w bp fp hp wp bpp fpp hpp wpp')
eqs = ns['equations'](r, jets[:4], jets[4:8], jets[8:], al)


def on(profiles, alpha_value):
    vals = profiles + [s.diff(v, r) for v in profiles] + [s.diff(v, r, 2) for v in profiles]
    sub = dict(zip(jets, vals)); sub[al] = alpha_value
    return [s.simplify(e.xreplace(sub)) for e in eqs]


# static Boulware-Deser (b = f = N, h = r^2, w = 0), r_H = 1, alpha symbolic
NB = N.subs(rH, 1)
res = on([NB, NB, r**2, 0*r], al)
ok('generated EGB equations vanish on Boulware-Deser (finite alpha)', all(z == 0 for z in res))
# extremal Myers-Perry in the ansatz, units r_H = 1 (extremal-mass/README.md)
b0 = (r**2 - 1)**2/(r**4 + 1); f0 = (1 - 1/r**2)**2; h0 = r**2 + 1/r**2; w0 = s.sqrt(2)/(r**4 + 1)
res = on([b0, f0, h0, w0], 0)
ok('generated EGB equations vanish on extremal Myers-Perry at alpha=0', all(z == 0 for z in res))
ok('extremal MP profiles: h_H = 2 = 1/(1-q^2), q = w(1) = 1/sqrt2, U = -2 -> M = 3pi/4, W = sqrt2 -> J = pi sqrt2/4',
   h0.subs(r, 1) == 2 and zero(w0.subs(r, 1) - 1/s.sqrt(2))
   and s.limit((b0 - 1)*r**2, r, s.oo) == -2 and s.limit(w0*r**4, r, s.oo) == s.sqrt(2))
ok('extremal MP in code units: m = 2M/pi = 3/2, s = 0', zero(2*(3*s.pi/4)/s.pi - s.Rational(3, 2)))
print(f'\n{len(PASS)} assertions passed; {len(NUMERIC)} of them certified numerically at random rational points (60 digits) rather than symbolically:')
for n_ in NUMERIC: print('   numeric:', n_)
