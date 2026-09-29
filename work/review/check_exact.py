"""Check egb_tensor.py on closed-form solutions, then derive Psi_0(q) independently.

1. Myers-Perry (equal spins) in the manuscript's radial gauge must have E_ab=0 at alpha=0.
2. Boulware-Deser must have E_ab=0 at finite alpha.
3. Psi_0 = dG/dalpha|_{T,Omega} = -(1/16pi) * int_{r>r_H} d^4x sqrt(-g) L_GB[MP]   (Reall-Santos),
   compared with eq. (16): -(pi/4)(u^2-14u+9), u=q^2/(1-q^2).
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from egb_tensor import field_equations

r = sp.Symbol('r', positive=True)


def vals_from(exprs, rv):
    out = {}
    for n, e in zip('bfhw', exprs):
        for k in range(3):
            out[f'{n}{k}'] = float(sp.diff(e, r, k).subs(r, rv))
    return out


def mp_exprs(q, rH=1):
    a = q*rH
    mu = rH**4/(rH**2 - a**2)
    f = 1 - mu/r**2 + mu*a**2/r**4
    h = r**2 + mu*a**2/r**2
    w = mu*a/(r**2*h)
    b = r**2*f/h
    return [sp.nsimplify(0) + e for e in (b, f, h, w)]


worst = 0
for q in [0.2, 0.5, 0.7]:
    ex = mp_exprs(sp.Rational(q).limit_denominator(1000))
    for rv in [1.3, 2.0, 5.0]:
        for thv in [0.4, 1.1]:
            E, G, aH, K, L = field_equations(rv, thv, vals_from(ex, rv), 0.0)
            worst = max(worst, np.max(np.abs(E))/max(1, np.max(np.abs(L))))
print(f'MP: max |G_ab| over samples = {worst:.2e}')
assert worst < 1e-10

# Boulware-Deser, alpha in the action R + alpha L_GB
worst = 0
for alpha in [0.1, 0.5]:
    rH = 1
    mu = rH**2 + 2*alpha
    f = 1 + r**2/(4*alpha)*(1 - sp.sqrt(1 + 8*alpha*mu/r**4))
    ex = [f, f, r**2, sp.Integer(0)]
    for rv in [1.2, 3.0]:
        E, G, aH, K, L = field_equations(rv, 0.7, vals_from(ex, rv), alpha)
        worst = max(worst, np.max(np.abs(E))/np.max(np.abs(G)))
    # and with the wrong coupling it must fail
    E, G, aH, K, L = field_equations(1.2, 0.7, vals_from(ex, 1.2), 1.1*alpha)
    assert np.max(np.abs(E))/np.max(np.abs(G)) > 1e-3
print(f'BD: max relative |G+aH| = {worst:.2e}')
assert worst < 1e-10

# Psi_0 by the Reall-Santos action integral
def psi0_action(q):
    ex = mp_exprs(sp.Rational(q).limit_denominator(10**6))
    b, f, h, w = ex
    # sqrt(-g) = sqrt(b/f) r^2 sqrt(h) sin(th)cos(th); angular integral of sin*cos over
    # th in [0,pi/2], phi1, phi2 in [0,2pi): (1/2)(2pi)^2 = 2 pi^2
    sqrtg = sp.lambdify(r, sp.sqrt(b/f)*r**2*sp.sqrt(h))
    def integrand(rv):
        _, _, _, K, L = field_equations(rv, 0.7, vals_from(ex, rv), 0.0)
        return 2*np.pi**2*sqrtg(rv)*L
    # The lambdified Riemann tensor contains 1/f and loses all digits within
    # ~1e-4 of the horizon (f=0), so [1, 1+d] is covered by a Chebyshev fit of
    # the integrand on [1+d, 1+10d], extrapolated inward (the integrand is a
    # rational function whose poles are at |r| <= 1 off the real axis).
    d = 2e-3
    far, _ = quad(integrand, 1+d, np.inf, limit=400, epsabs=1e-13, epsrel=1e-12)
    xs = 1+d+9*d*(1-np.cos(np.linspace(0, np.pi, 25)))/2
    fit = np.polynomial.chebyshev.Chebyshev.fit(xs, [integrand(x) for x in xs], 16)
    near = fit.integ()(1+d)-fit.integ()(1.0)
    val = far+near
    return -val/(16*np.pi)


print(' q        action-integral        eq.(16)            diff')
for q in [0.0, 0.3, 0.5, 0.634936, 0.7]:
    u = q*q/(1-q*q)
    ref = -np.pi/4*(u*u - 14*u + 9)
    got = psi0_action(q)
    print(f'{q:.6f}  {got: .12f}  {ref: .12f}  {got-ref: .2e}')
    assert abs(got-ref) < 1e-7*max(1, abs(ref))
print('Psi_0(q) of eq.(16) reproduced from the MP action integral')
