"""Two-sided solver with a high-precision outward leg (for large s).

outward: mp_series.MPSeries (60 digits) start data at x0 = xf * radius, integrated in __float128
         (build_qrhs.qintegrate, Bulirsch-Stoer, tol 1e-30) to r_m;
inward : large-r series + double DOP853 (twoside.TwoSide.inward), from R to r_m;
match  : (f, h, h') at r_m, Newton in (B, mu, b4).   m(s) = 3 mu / 4,  s = 2 alpha (code units).
Usage: python twoside_q.py alpha [xf L]"""
import sys, time, json
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from mp_series import MPSeries
from build_qrhs import qintegrate
from asym_numeric import solve_seq
from build_crhs import crhs


class TwoSideQ:
    def __init__(self, alpha, xf=0.03, L=18, ng=10, rm=None, R=None, M=16, S=None):
        self.alpha = float(alpha)
        self.S = S or MPSeries(alpha, L=L, ng=ng)
        self.rho = self.S.radius()
        self.x0 = xf*self.rho
        rH = float(self.S.rH)
        self.rm = rm or 1.5*rH+0.5*np.sqrt(alpha)
        self.R = R or max(40., 25*np.sqrt(alpha))
        self.M = M

    def outward(self, B, r1=None, tol=1e-30):
        """quad precision from r_H + x0 to r_H + xs (xs = 0.5 * series radius, where the gamma-mode is
        already O(0.1) of the departure from the throat), then double DOP853 to r1"""
        from extremal_shoot import ev_h, ev_big, ev_det
        r1 = r1 or self.rm
        r0, y0 = self.S.start(B, self.x0)
        rs = min(float(self.S.rH)+0.5*float(self.rho), r1)
        code, r, y, n = qintegrate(r0, rs, self.S.alpha, y0, tol=tol)
        y = np.array([float(v) for v in y])
        if code != 0 or rs >= r1:
            return code, float(r), y
        sol = solve_ivp(crhs, (float(r), r1), y, args=(self.alpha,), method='DOP853', rtol=1e-13, atol=1e-15,
                        events=[ev_h, ev_big, ev_det])
        code = 0 if sol.status == 0 else (1 if sol.t_events[0].size else 2 if sol.t_events[1].size else 3)
        return code, float(sol.t[-1]), sol.y[:, -1]

    def inward(self, mu, b4):
        a, b, chk = solve_seq(self.alpha, mu, b4, M=self.M)
        z = 1/self.R; k = np.arange(len(a))
        Ht = np.sum(b*z**k); Htp = np.sum(k[1:]*b[1:]*z**(k[1:]-1))
        y0 = np.array([np.sum(a*z**k), Ht/z**2, (2*Ht-z*Htp)/z])
        sol = solve_ivp(crhs, (self.R, self.rm), y0, args=(self.alpha,), method='DOP853', rtol=1e-13, atol=1e-15)
        return sol.y[:, -1]

    def mismatch(self, p):
        code, r, yo = self.outward(p[0])
        if code != 0:
            return np.full(3, np.nan)
        yi = self.inward(p[1], p[2])
        return (yo-yi)/np.array([1., self.rm**2, self.rm])

    def T(self, B, rend):
        code, r, y = self.outward(B, rend)
        f, h, hp = y
        if h <= 0:
            return -1.0, r, y
        eta = np.log(h/r**2)
        return r*(hp/h-2/r)+4*eta, r, y

    def guess(self, n=13, verbose=False, Bhat=None):
        sc = float(self.rho)**(-float(self.S.gamma))
        if Bhat is not None:     # continuation: bracket around the previous normalised amplitude
            Bs = np.sort(sc*Bhat*np.array([0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 2.0]))
        else:
            Bs = np.r_[-sc*np.logspace(4, -4, n), sc*np.logspace(-4, 4, n)]
        rend = 6*self.rm
        sg = [np.sign(self.T(B, rend)[0]) for B in Bs]
        if verbose:
            print('scan signs', ''.join('+' if v > 0 else '-' for v in sg), flush=True)
        for i in range(len(Bs)-1):
            if sg[i] != sg[i+1]:
                lo, hi, slo = Bs[i], Bs[i+1], sg[i]
                best = None
                for it in range(45):
                    mid = 0.5*(lo+hi)
                    t, r, y = self.T(mid, rend)
                    if best is None or r >= best[1]:
                        best = (mid, r, y)
                    if np.sign(t) == slo: lo = mid
                    else: hi = mid
                B, r, y = best
                rr = min(r, 3*self.rm)
                code, r2, y2 = self.outward(B, rr)
                return [B, rr**2*(1-y2[0]), 0.0]
        raise RuntimeError('no B bracket')

    def solve(self, p0, maxit=25, verbose=False):
        p = np.array(p0, float); hist = []
        for it in range(maxit):
            e = self.mismatch(p)
            if not np.all(np.isfinite(e)):
                raise RuntimeError('non-finite mismatch')
            hist.append(np.max(np.abs(e)))
            if hist[-1] < 1e-12 or (len(hist) > 3 and hist[-1] > 0.3*min(hist[:-1]) and min(hist) < 1e-7):
                break
            Jm = np.empty((3, 3))
            for j in range(3):
                dp = np.zeros(3); dp[j] = 1e-6*max(1, abs(p[j]))
                Jm[:, j] = (self.mismatch(p+dp)-self.mismatch(p-dp))/(2*dp[j])
            step = np.linalg.solve(Jm, -e)
            lam = 1.0
            for bt in range(30):
                en = self.mismatch(p+lam*step)
                if np.all(np.isfinite(en)) and np.max(np.abs(en)) < np.max(np.abs(e)):
                    break
                lam *= 0.5
            p = p+lam*step
            if verbose:
                print(f'  it {it}: |e|={hist[-1]:.2e} p={p}', flush=True)
        self.p, self.err = p, float(min(hist))
        return p


if __name__ == '__main__':
    alpha = float(sys.argv[1])
    xf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    t = time.time()
    T = TwoSideQ(alpha, xf=xf, L=L)
    print(f'alpha={alpha} gamma={mp.nstr(T.S.gamma, 12)} (throat {mp.nstr(T.S.gamma_check, 2)}) radius={mp.nstr(T.rho, 4)} '
          f'x0={mp.nstr(T.x0, 4)} rm={T.rm:.3f} R={T.R:.1f}  series {time.time()-t:.0f}s', flush=True)
    g = T.guess(verbose=True)
    print('guess', g, f'{time.time()-t:.0f}s', flush=True)
    p = T.solve(g, verbose=True)
    print(f'RESULT alpha={alpha} s={2*alpha} B={p[0]:.12g} mu={p[1]:.13f} b4={p[2]:.10g} m={0.75*p[1]:.13f} err={T.err:.1e} [{time.time()-t:.0f}s]', flush=True)
