"""High-precision (mpmath) horizon data for the quad-precision outward leg.

Same construction as horizon_series.py / gamma_mode.py, in mpmath (dps = DPS):
  * throat (g_H, q) from  Pt = 16 al^2 q^2 (4-q) - 8 al g (2q^2-7q+4) + g^2 (q-2) = 0,
                          Jt = g^2 q^2 (4-q) (g + 4 al (4-3q)) - 8 = 0          (J = J0, code units);
    r_H = sqrt(g), H0 = q g, F0 = 1/v, v = g R (3+4a(4-q)) / (2 (4-q)(3+4a(8-6q))), a = al/g;
  * H1 from Nt_2 (root nearest the double-precision physical root), kappa_m at that H1;
  * levels m = 2..L: (F_{m-1}, H_m) from [Nb[m], Nh[m+2] - kappa_{m+1} Nb[m+1]];
  * gamma: root of det M(lambda) of the level matrices (entries are polynomials of degree 1 and 2
    in the level, interpolated exactly from m = 2,3,4), checked against the throat formula;
  * gamma-mode series (phi_k, psi_k), psi_0 = 1.
Derivatives by complex step with mpc (exact to working precision)."""
import numpy as np
import mpmath as mp
from tps import TPS
from ode_poly import coeffs_tps
from throat import throat, Qfun

DPS = 60
mp.mp.dps = DPS
EPS = mp.mpf(10)**(-DPS+10)


def _arr(v):
    return np.array([mp.mpc(x) for x in v], dtype=object)


def residual(rH, F, H, alpha, K):
    TPS.K = K
    x = TPS(np.array([mp.mpc(0), mp.mpc(1)], dtype=object))
    R = x+rH
    f = TPS(np.asarray(F, dtype=object)).shift(2)
    h = TPS(np.asarray(H, dtype=object))
    hp = h.deriv()
    Ab, Bb, Cb, Ah, Bh, Ch = coeffs_tps(R, f, h, hp, alpha)
    fp, hpp = f.deriv(), hp.deriv()
    return (Ab*fp+Bb*hpp+Cb).c, (Ah*fp+Bh*hpp+Ch).c


def _D(P, e0):
    c = P.c
    q = np.array([mp.mpc(0)]*len(c), dtype=object)
    q[:-1] = c[1:]
    return TPS(np.array([e0*v for v in q], dtype=object))+P.deriv()


class MPSeries:
    def __init__(self, alpha, L=14, H1_guess=None, ng=10):
        self.alpha_f = float(alpha)
        al = self.alpha = mp.mpf(alpha)
        self.L = L
        T = throat(float(alpha))
        Pt = lambda g, q: 16*al**2*q**2*(4-q)-8*al*g*(2*q**2-7*q+4)+g**2*(q-2)
        Jt = lambda g, q: g**2*q**2*(4-q)*(g+4*al*(4-3*q))-8
        g, q = mp.findroot([Pt, Jt], (mp.mpf(T['g_H']), mp.mpf(T['q'])))
        a = al/g
        R = 1+4*a*(4-3*q)
        v = g*R*(3+4*a*(4-q))/(2*(4-q)*(3+4*a*(8-6*q)))
        self.g, self.q, self.a, self.v = g, q, a, v
        self.rH = mp.sqrt(g)
        self.F0, self.H0 = 1/v, q*g
        Q = Qfun(a, q)
        self.gamma_throat = (-1+mp.sqrt(1+4*Q))/2
        # H1
        if H1_guess is None:
            from extremal_shoot import build
            S, I, gerr = build(float(alpha), 12)
            H1_guess = S.H[1]
        self.H1 = mp.findroot(self.Nt2, mp.mpf(H1_guess))
        self.kappa = self._kappa(self.H1)
        self._levels()
        self._indicial()
        self._gamma_series(ng)

    # ---------------------------------------------------------------- background
    def _base(self, n):
        F = _arr([0]*(n+1)); H = _arr([0]*(n+2))
        F[0], H[0] = self.F0, self.H0
        return F, H

    def Nt2(self, H1):
        F, H = self._base(4)
        H[1] = mp.mpc(H1)
        b, h = residual(self.rH, F, H, self.alpha, 5)
        Fc = F.copy(); Fc[1] += mp.mpc(0, EPS)
        bb, hh = residual(self.rH, Fc, H, self.alpha, 5)
        kap = mp.im(hh[3])/mp.im(bb[2])
        return mp.re(h[3]-kap*b[2])

    def _kappa(self, H1):
        L = self.L
        F, H = self._base(L+1)
        H[1] = mp.mpc(H1)
        kap = {}
        for m in range(2, L+3):
            Fc = F.copy(); Fc[m-1] += mp.mpc(0, EPS)
            b, h = residual(self.rH, Fc, H, self.alpha, m+2)
            kap[m] = mp.im(h[m+1])/mp.im(b[m])
        return kap

    def _levels(self, newton=3):
        L = self.L
        F, H = self._base(L+1)
        H[1] = mp.mpc(self.H1)
        self.mats = {}
        for m in range(2, L+2):
            for it in range(newton):
                def eq(Fm, Hm):
                    Fc = F.copy(); Hc = H.copy(); Fc[m-1] = Fm; Hc[m] = Hm
                    b, h = residual(self.rH, Fc, Hc, self.alpha, m+3)
                    return [b[m], h[m+2]-self.kappa[m+1]*b[m+1]]
                e = [mp.re(z) for z in eq(F[m-1], H[m])]
                c1 = eq(F[m-1]+mp.mpc(0, EPS), H[m]); c2 = eq(F[m-1], H[m]+mp.mpc(0, EPS))
                J = mp.matrix([[mp.im(c1[0])/EPS, mp.im(c2[0])/EPS], [mp.im(c1[1])/EPS, mp.im(c2[1])/EPS]])
                d = mp.lu_solve(J, mp.matrix([-e[0], -e[1]]))
                F[m-1] += d[0]; H[m] += d[1]
            self.mats[m] = J
        self.F = [mp.re(v) for v in F[:L+1]]
        self.H = [mp.re(v) for v in H[:L+2]]
        # residual check (relative to the size of the linear terms)
        b, h = residual(self.rH, _arr(self.F), _arr(self.H), self.alpha, L+3)
        self.check = (max(abs(b[k]) for k in range(L+1)), max(abs(h[k]) for k in range(L+2)))

    def _indicial(self):
        ms = [2, 3, 4]
        def M(lam):
            out = mp.matrix(2, 2)
            for i in range(2):
                for j in range(2):
                    out[i, j] = mp.interpolate if False else sum(
                        self.mats[mk][i, j]*mp.fprod([(lam-ml)/(mk-ml) for ml in ms if ml != mk]) for mk in ms)
            return out
        self.M = M
        det = lambda lam: mp.det(M(lam))
        self.gamma = mp.findroot(det, self.gamma_throat)
        Mg = M(self.gamma)
        self.nv = (-Mg[0, 1]/Mg[0, 0], mp.mpf(1))
        self.gamma_check = abs(self.gamma-self.gamma_throat)

    # ---------------------------------------------------------------- gamma mode
    def _linear(self, phi, psi, K):
        gam = self.gamma
        e0 = gam-3
        TPS.K = K
        x = TPS(np.array([mp.mpc(0), mp.mpc(1)], dtype=object))
        R = x+self.rH
        f = TPS(_arr(self.F)).shift(2); h = TPS(_arr(self.H))
        hp = h.deriv(); hpp = hp.deriv(); fp = f.deriv()
        Fh = TPS(_arr(phi)).shift(4); Hh = TPS(_arr(psi)).shift(3)
        dHp = _D(Hh, e0); dHpp = _D(dHp, e0); dFp = _D(Fh, e0)
        ie = mp.mpc(0, EPS)
        fc = f+Fh*ie; hc = h+Hh*ie; hpc = hp+dHp*ie; hppc = hpp+dHpp*ie; fpc = fp+dFp*ie
        Ab, Bb, Cb, Ah, Bh, Ch = coeffs_tps(R, fc, hc, hpc, self.alpha)
        Nb = [mp.im(v)/EPS for v in (Ab*fpc+Bb*hppc+Cb).c]
        Nh = [mp.im(v)/EPS for v in (Ah*fpc+Bh*hppc+Ch).c]
        return Nb, Nh

    def _gamma_series(self, n):
        kap = self.kappa[2]
        phi = [mp.mpf(0)]*(n+1); psi = [mp.mpf(0)]*(n+1)
        phi[0], psi[0] = self.nv[0], mp.mpf(1)
        K = n+8
        Nb, Nh = self._linear(phi, psi, K)
        self.gamma_lead = (abs(Nb[3]), abs(Nh[5]-kap*Nb[4]))
        for k in range(1, n+1):
            def ev(a_, b_):
                ph = list(phi); ps = list(psi); ph[k], ps[k] = a_, b_
                Nb, Nh = self._linear(ph, ps, K)
                return [Nb[3+k], Nh[5+k]-kap*Nb[4+k]]
            e = ev(0, 0); e1 = ev(1, 0); e2 = ev(0, 1)
            J = mp.matrix([[e1[0]-e[0], e2[0]-e[0]], [e1[1]-e[1], e2[1]-e[1]]])
            d = mp.lu_solve(J, mp.matrix([-e[0], -e[1]]))
            phi[k], psi[k] = d[0], d[1]
        self.phi, self.psi = phi, psi

    # ---------------------------------------------------------------- start data
    def start(self, B, x0):
        B = mp.mpf(B); x0 = mp.mpf(x0); gam = self.gamma
        f = x0**2*sum(c*x0**k for k, c in enumerate(self.F)) + B*sum(c*x0**(gam+1+k) for k, c in enumerate(self.phi))
        h = sum(c*x0**k for k, c in enumerate(self.H)) + B*sum(c*x0**(gam+k) for k, c in enumerate(self.psi))
        hp = sum(k*c*x0**(k-1) for k, c in enumerate(self.H) if k) + B*sum(c*(gam+k)*x0**(gam+k-1) for k, c in enumerate(self.psi))
        return self.rH+x0, [f, h, hp]

    def radius(self):
        """root-test estimate of the convergence radius of the background and gamma-mode series"""
        out = []
        for c in (self.H, self.F, self.psi):
            ks = [k for k in range(3, len(c)) if c[k] != 0]
            out.append(min(abs(c[k])**(-mp.mpf(1)/k) for k in ks))
        return min(out)


if __name__ == '__main__':
    import sys, time
    for al in [float(v) for v in sys.argv[1:]] or [1.491010442529934, 10.0]:
        t = time.time()
        S = MPSeries(al)
        print(f'alpha={al}: rH={mp.nstr(S.rH, 20)} H1={mp.nstr(S.H1, 20)} gamma={mp.nstr(S.gamma, 20)} '
              f'|gamma-throat|={mp.nstr(S.gamma_check, 3)} nv={mp.nstr(S.nv[0], 12)} lead={[mp.nstr(v, 3) for v in S.gamma_lead]} '
              f'check={[mp.nstr(v, 3) for v in S.check]} radius~{mp.nstr(S.radius(), 4)}  [{time.time()-t:.0f}s]', flush=True)
        print('   psi:', [mp.nstr(v, 8) for v in S.psi[:5]])
