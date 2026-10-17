"""Extremal throat at fixed J = J0 = pi sqrt2/4 (code units, C2 = 32).
Formulas (1)-(3) of reports/strong-coupling-endpoint.pdf with g_H scaled back in:
  a = alpha/g_H,  q = h_H/g_H,
  P(a,q) = 16 a^2 q^2 (4-q) - 8 a (2q^2 - 7q + 4) + q - 2 = 0   (root in (0,2]),
  g_H^3 q^2 (4-q) R / 8 = 1,   R = 1 + 4a(4-3q),
  AdS2 radius^2  v_phys = g_H R[3+4a(4-q)] / (2(4-q)[3+4a(8-6q)]).
Also the throat exponent gamma from the indicial equation (7) of the same report."""
import numpy as np
from scipy.optimize import brentq


def q_of_a(a):
    if a == 0:
        return 2.0
    P = lambda q: 16*a**2*q**2*(4-q)-8*a*(2*q**2-7*q+4)+q-2
    return brentq(P, 1e-14, 2.0, xtol=1e-16, rtol=1e-15)


def Jbar2(a):
    q = q_of_a(a)
    R = 1+4*a*(4-3*q)
    return q**2*(4-q)*R/8


def throat(alpha):
    """returns dict g_H, h_H, q, a, v (physical AdS2 radius^2), gamma"""
    if alpha == 0:
        g = 1.0
    else:
        # g^3 Jbar2(alpha/g) = 1 ; solve in log g
        F = lambda lg: 3*lg+np.log(Jbar2(alpha/np.exp(lg)))
        g = np.exp(brentq(F, -10, 10, xtol=1e-15))
    a = alpha/g
    q = q_of_a(a)
    R = 1+4*a*(4-3*q)
    vhat = R*(3+4*a*(4-q))/(2*(4-q)*(3+4*a*(8-6*q)))
    return dict(g_H=g, h_H=q*g, q=q, a=a, R=R, v=g*vhat, gamma=gamma(a, q))


def Qfun(a, q):
    N5 = 2048*q**2*(6*q**4-61*q**3+212*q**2-304*q+192)
    N4 = 256*q*(-32*q**4+159*q**3-132*q**2-48*q+320)
    N3 = 128*(-44*q**4+213*q**3-274*q**2+208*q+32)
    N2 = 64*(4*q**3+6*q**2-27*q+44)
    N1 = -16*(q**2+12*q-19)
    N0 = 9
    num = 4*sum(an*Nn for an, Nn in zip([1, a, a**2, a**3, a**4, a**5], [N0, N1, N2, N3, N4, N5]))
    den = (q-4)*(1+8*a*q)*(12*a*q-16*a-1)**2*(24*a*q-32*a-3)
    return num/den


def gamma(a, q):
    Q = Qfun(a, q)
    return (-1+np.sqrt(1+4*Q))/2


if __name__ == '__main__':
    # benchmarks quoted in the report: Q(0,2)=6, Q(1/12,1)=288/55, gamma=2 at a=0.7319292...
    assert abs(Qfun(0, 2)-6) < 1e-12 and abs(Qfun(1/12, 1)-288/55) < 1e-12
    a0 = 0.73192922516721036349
    assert abs(gamma(a0, q_of_a(a0))-2) < 1e-10
    for al in [0, 0.01, 0.1, 0.5, 1, 2, 3.1176, 5, 10, 25, 50]:
        T = throat(al)
        s_ = 2*al
        print(f"alpha={al:7.3f} s={s_:7.3f} g_H={T['g_H']:.10f} q={T['q']:.8f} a={T['a']:.5f} v={T['v']:.8f} gamma={T['gamma']:.6f}")
