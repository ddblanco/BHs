"""Rebuild b,f,h,w(r) and r-derivatives from stored Chebyshev coefficients.

Compact variables of the manuscript's appendix A: z=1/r, x=1-z (x=0 horizon, x=1 infinity),
b=(1-z^2)B, f=(1-z^2)F, h=(1+z^4 H)/z^2, w=z^4 W. No project code is used.
"""
import numpy as np
from numpy.polynomial import Chebyshev


class Profile:
    def __init__(self, coefficients):
        self.p = [Chebyshev(c, domain=[0, 1]) for c in np.asarray(coefficients)]
        self.d = [[p.deriv(k) for p in self.p] for k in range(3)]

    def compact(self, x, k=0):
        return np.array([p(x) for p in self.d[k]])

    def radial(self, r):
        """dict b0,b1,b2,...,w2 at radius r (r-derivatives)."""
        r = np.asarray(r, float)
        z = 1/r
        x = 1-z
        V, Vx, Vxx = self.compact(x, 0), self.compact(x, 1), self.compact(x, 2)
        d1 = Vx*z**2
        d2 = Vxx*z**4-2*Vx*z**3
        (B, F, H, W), (B1, F1, H1, W1), (B2, F2, H2, W2) = V, d1, d2
        out = {}
        for n, (X, X1, X2) in (('b', (B, B1, B2)), ('f', (F, F1, F2))):
            out[n+'0'] = (1-z**2)*X
            out[n+'1'] = 2*z**3*X+(1-z**2)*X1
            out[n+'2'] = -6*z**4*X+4*z**3*X1+(1-z**2)*X2
        out['h0'] = 1/z**2+z**2*H
        out['h1'] = 2/z-2*z**3*H+z**2*H1
        out['h2'] = 2+6*z**4*H-4*z**3*H1+z**2*H2
        out['w0'] = z**4*W
        out['w1'] = -4*z**5*W+z**4*W1
        out['w2'] = 20*z**6*W-8*z**5*W1+z**4*W2
        return out

    def observables(self, alpha, omega):
        """E, J, T, S from the profile alone, without tail fits.

        b=1+U/r^2+...: with B(z)=1+beta z^2+..., U=beta-1 and beta=B_xx(1)/2 (B_x(1)=0 by the BC).
        w=W_inf/r^4: J=pi W(1)/4.  T=sqrt(b'f')/(4 pi) at r=1: b'(1)=2B(0), f'(1)=2F(0).
        S=(A/4)(1+2 alpha R~), A=2 pi^2 sqrt(h_H), R~=8-2h_H, h_H=1+H(0).
        """
        B, F, H, W = self.compact(0.)
        Binf_xx = self.compact(1., 2)[0]
        Winf = self.compact(1.)[3]
        U = Binf_xx/2-1
        hH = 1+H
        area = 2*np.pi**2*np.sqrt(hH)
        return dict(E=-3*np.pi*U/8, J=np.pi*Winf/4,
                    T_H=np.sqrt(4*B*F)/(4*np.pi),
                    S=area/4*(1+2*alpha*(8-2*hH)),
                    Omega_H=W, bx_inf=self.compact(1., 1)[0])
