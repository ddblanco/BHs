"""Formal power-series solution of the reduced fixed-J system at the extremal horizon,
    f = x^2 sum_k F_k x^k,  h = sum_k H_k x^k,  x = r - r_H .
* r_H = sqrt(g_H), H_0 = h_H, F_0 = 1/v from the exact throat (throat.py); they satisfy
  Nb[0], Nb[1], Nh[1], Nh[2] identically (checked in probe_struct2.py).
* The level-m block d(Nb[m], Nh[m+1])/d(F_{m-1}, H_m) has rank one:
  Nh[m+1] - kappa_m Nb[m] =: Nt_m does not contain the level-m unknowns.
* Square system: unknowns H_1, (F_{m-1}, H_m) for m = 2..L;
  equations Nt_2, and Nb[m], Nt_{m+1} for m = 2..L.  Solved by Newton (complex step).
  H_1 enters Nt_2 quadratically: two roots; the physical one is selected by continuity.
The lapse u = b'/b = U(r,f,h,h') is then 2/x + O(1) automatically (checked)."""
import numpy as np
from tps import TPS
from ode_poly import coeffs_tps
from throat import throat
import _U_generated as UG


def _fields(rH, F, H, K):
    TPS.K = K
    dt = np.result_type(np.asarray(F).dtype, np.asarray(H).dtype, type(rH))
    x = TPS(np.array([0., 1.], dtype=dt))
    R = x+rH
    f = TPS(np.asarray(F, dtype=dt)).shift(2)
    h = TPS(np.asarray(H, dtype=dt))
    return R, f, h, h.deriv()


def residual_series(rH, F, H, alpha, K):
    R, f, h, hp = _fields(rH, F, H, K)
    Ab, Bb, Cb, Ah, Bh, Ch = coeffs_tps(R, f, h, hp, alpha)
    fp, hpp = f.deriv(), hp.deriv()
    return (Ab*fp+Bb*hpp+Cb).c, (Ah*fp+Bh*hpp+Ch).c


def lapse_series(rH, F, H, alpha, K):
    R, f, h, hp = _fields(rH, F, H, K)
    return UG.Unum(R, f, h, hp, alpha).c, UG.Uden(R, f, h, hp, alpha).c


class HorizonSeries:
    def __init__(self, alpha, L):
        self.alpha, self.L = alpha, L
        T = self.T = throat(alpha)
        self.rH = np.sqrt(T['g_H'])
        self.F0, self.H0 = 1/T['v'], T['h_H']
        self.K = L+3
        self.set_kappa(0.)

    def set_kappa(self, H1):
        """kappa_m from the rank-one blocks; they depend on the x^0 data r_H, H_0 and H_1 = h'(r_H)"""
        L = self.L
        F = np.zeros(L+1, complex); H = np.zeros(L+2, complex)
        F[0], H[0], H[1] = self.F0, self.H0, H1
        self.kappa = {}
        for m in range(2, L+2):
            Fc = F.copy(); Fc[m-1] += 1e-20j
            b, h_ = residual_series(self.rH, Fc, H, self.alpha, self.K)
            self.kappa[m] = np.imag(h_[m+1])/np.imag(b[m])

    def unpack(self, u):
        L = self.L
        F = np.zeros(L+1, dtype=u.dtype); H = np.zeros(L+2, dtype=u.dtype)
        F[0], H[0] = self.F0, self.H0
        H[1] = u[0]
        for m in range(2, L+1):
            F[m-1] = u[1+2*(m-2)]; H[m] = u[2+2*(m-2)]
        return F, H

    def equations(self, u):
        F, H = self.unpack(u)
        b, h_ = residual_series(self.rH, F, H, self.alpha, self.K)
        out = [h_[3]-self.kappa[2]*b[2]]
        for m in range(2, self.L+1):
            out += [b[m], h_[m+2]-self.kappa[m+1]*b[m+1]]
        return np.array(out)

    def solve(self, u0, tol=1e-14, maxit=40, verbose=False):
        u = np.array(u0, float)
        for it in range(maxit):
            e = np.real(self.equations(u.astype(complex)))
            J = np.empty((len(e), len(u)))
            for i in range(len(u)):
                uc = u.astype(complex); uc[i] += 1e-20j
                J[:, i] = np.imag(self.equations(uc))/1e-20
            sc = np.max(np.abs(J), axis=1); sc[sc == 0] = 1
            step = np.linalg.solve(J/sc[:, None], e/sc)
            u = u-step
            if verbose:
                print(it, np.max(np.abs(step)), np.max(np.abs(e/sc)))
            if np.max(np.abs(step)) < tol*max(1, np.max(np.abs(u))):
                break
        self.u = u
        self.F, self.H = self.unpack(u)
        self.cond = float(np.linalg.cond(J/sc[:, None]))
        e = np.real(self.equations(u.astype(complex)))
        self.res = float(np.max(np.abs(e/sc)))
        return u

    def check(self):
        b, h_ = residual_series(self.rH, self.F, self.H, self.alpha, self.K)
        un, ud = lapse_series(self.rH, self.F, self.H, self.alpha, self.K)
        return dict(Nb=np.abs(b[:self.L+1]).max(), Nh=np.abs(h_[:self.L+2]).max(),
                    Nb_next=abs(b[self.L+2]), Nh_next=abs(h_[self.L+3]),
                    xU=float(np.real(un[1]/ud[2])), U0=float(abs(un[0])))


    # ---- sequential solve: H1 from Nt_2 (1-D, root chosen by the caller), then levels 2..L
    def Nt2(self, H1):
        F = np.zeros(4, complex); H = np.zeros(5, complex)
        F[0], H[0] = self.F0, self.H0
        H[1] = H1
        b, h_ = residual_series(self.rH, F, H, self.alpha, 5)
        Fc = F.copy(); Fc[1] += 1e-20j
        bb, hh = residual_series(self.rH, Fc, H, self.alpha, 5)
        kap = np.imag(hh[3])/np.imag(bb[2])
        return np.real(h_[3]-kap*b[2])

    def H1_roots(self, lo=-6., hi=6., n=241):
        from scipy.optimize import brentq
        xs = np.linspace(lo, hi, n)
        v = np.array([self.Nt2(x) for x in xs])
        out = []
        for i in range(n-1):
            if np.sign(v[i]) != np.sign(v[i+1]):
                out.append(brentq(self.Nt2, xs[i], xs[i+1], xtol=1e-15, rtol=1e-15))
        return out

    def solve_sequential(self, H1, newton=4):
        L = self.L
        F = np.zeros(L+1, complex); H = np.zeros(L+2, complex)
        F[0], H[0], H[1] = self.F0, self.H0, H1
        # refine H1 by Newton on Nt_2
        for it in range(5):
            e = self.Nt2(H[1].real)
            d = (self.Nt2(H[1].real+1e-7)-self.Nt2(H[1].real-1e-7))/2e-7
            H[1] = H[1].real-e/d
        self.set_kappa(H[1].real)
        dets = {}
        for m in range(2, L+1):
            for it in range(newton):
                def eq(Fm, Hm):
                    Fc = F.copy(); Hc = H.copy(); Fc[m-1] = Fm; Hc[m] = Hm
                    b, h_ = residual_series(self.rH, Fc, Hc, self.alpha, m+3)
                    return np.array([b[m], h_[m+2]-self.kappa[m+1]*b[m+1]])
                e = np.real(eq(F[m-1], H[m]))
                J = np.array([np.imag(eq(F[m-1]+1e-20j, H[m]))/1e-20, np.imag(eq(F[m-1], H[m]+1e-20j))/1e-20]).T
                d = np.linalg.solve(J, -e)
                F[m-1] += d[0]; H[m] += d[1]
            dets[m] = np.linalg.det(J)/np.prod(np.abs(J).max(axis=1))
        self.F, self.H = F.real, H.real
        self.dets = dets
        u = np.zeros(1+2*(L-1)); u[0] = self.H[1]
        for m in range(2, L+1):
            u[1+2*(m-2)] = self.F[m-1]; u[2+2*(m-2)] = self.H[m]
        self.u = u
        e = np.real(self.equations(u.astype(complex)))
        self.res = float(np.max(np.abs(e)))
        return self


if __name__ == '__main__':
    for alpha in [0.01, 0.1, 0.3, 1.0, 5.0]:
        r = H1_roots(alpha)
        print(f'alpha={alpha}: H1 roots {r}')
        for root in r:
            S = HorizonSeries(alpha, 10)
            u0 = np.zeros(1+2*(S.L-1)); u0[0] = np.real(root)
            S.solve(u0)
            c = S.check()
            print(f'   H1={S.H[1]:.10f}: res {S.res:.1e} cond {S.cond:.1e}  check {c}')
            print('     F:', np.array2string(S.F[:6], precision=6), '\n     H:', np.array2string(S.H[:6], precision=6))
