"""Extremal fixed-J EGB black hole at coupling alpha (code units J = J0 = pi sqrt2/4, s = 2 alpha) by shooting.

1. exact throat (throat.py) -> r_H, H_0, F_0, gamma;
2. horizon series (horizon_series.py); the physical H_1 root is the one whose indicial roots are
   {-1-gamma, -1, gamma} (checked against the independent throat formula);
3. gamma-mode amplitude B shot so that the squashing growing mode is absent at R:
   T(B) = R eta'(R) + 4 eta(R) = 6 A R^2 + O(R^-6),  eta = log(h/r^2);
4. mu = lim r^2 (1-f) by polynomial extrapolation in 1/r^2;  m(s) = 3 mu / 4.
Usage: python extremal_shoot.py alpha [L x0 R]
"""
import sys, json, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from horizon_series import HorizonSeries
from shoot import Indicial, initial_data
from build_crhs import crhs as rhs


def build(alpha, L=12):
    S = HorizonSeries(alpha, L)
    best = None
    for r in S.H1_roots(-12, 12, 481):
        S.solve_sequential(r)
        I = Indicial(S, mmax=min(L, 10))
        err = np.min(np.abs(I.det_roots()-S.T['gamma']))
        if best is None or err < best[0]:
            best = (err, r)
    S.solve_sequential(best[1])
    I = Indicial(S, mmax=min(L, 10))
    return S, I, best[0]


def ev_h(r, y, a): return y[1]-0.05*r**2
ev_h.terminal = True
def ev_big(r, y, a): return y[1]-20*r**2
ev_big.terminal = True
def ev_det(r, y, a):
    # blow-up of the derivatives (approach to the det = 0 surface off the true solution)
    d = rhs(r, y, a)
    return 1e3-max(abs(d[0])*r, abs(d[2])*r*r/max(abs(y[1]), 1e-300))
ev_det.terminal = True


class Shooter:
    def __init__(self, alpha, L=12, x0=None, R=None, rtol=1e-12):
        self.alpha = alpha
        self.S, self.I, self.gerr = build(alpha, L)
        self.gam = self.S.T['gamma']
        self.nv = self.I.null(self.gam)
        rH = self.S.rH
        self.x0 = x0 or 0.02*rH
        self.R = R or max(12., 6*np.sqrt(alpha))
        self.rtol = rtol

    def run(self, B, R=None, dense=False):
        y0 = initial_data(self.S, self.gam, self.nv, B, self.x0)
        return solve_ivp(rhs, (self.S.rH+self.x0, R or self.R), y0, args=(self.alpha,), method='DOP853',
                         rtol=self.rtol, atol=1e-14, events=[ev_h, ev_big, ev_det], dense_output=dense)

    def T(self, B):
        """sign-carrying shooting function, read at the end of the trajectory (R or escape)"""
        sol = self.run(B)
        r = sol.t[-1]; f, h, hp = sol.y[:, -1]
        if h <= 0:
            return -1.0, sol
        eta = np.log(h/r**2); etap = hp/h-2/r
        return (r*etap+4*eta), sol

    def solve(self, B_lo, B_hi, nbis=200, verbose=False):
        tlo, slo = self.T(B_lo); thi, shi = self.T(B_hi)
        assert np.sign(tlo) != np.sign(thi), (tlo, thi)
        best = None
        for it in range(nbis):
            Bm = 0.5*(B_lo+B_hi)
            if Bm in (B_lo, B_hi):
                break
            tm, sm = self.T(Bm)
            if best is None or sm.t[-1] >= best[1].t[-1]:
                best = (Bm, sm)
            if np.sign(tm) == np.sign(tlo):
                B_lo, tlo = Bm, tm
            else:
                B_hi, thi = Bm, tm
        self.B, self.best = best[0], best[1]
        if verbose:
            print(f'B* in [{B_lo:.17g}, {B_hi:.17g}], longest trajectory reaches r = {best[1].t[-1]:.3f} (status {best[1].status})', flush=True)
        return self.B

    def scan(self, Bs):
        return [(B, np.sign(self.T(B)[0])) for B in Bs]

    def mass(self, rmin_factor=0.25):
        """mu from r^2(1-f) along the longest trajectory, on [rmin, r_ok] where the growing mode is
        still negligible (|R eta' + 4 eta| small); fit mu + c1/r^2 + c2/r^4 + c3/r^6."""
        sol = self.run(self.B, R=self.best.t[-1], dense=True)
        r = np.linspace(sol.t[0], sol.t[-1], 20000)
        f, h, hp = sol.sol(r)
        eta = np.log(h/r**2); T = r*(hp/h-2/r)+4*eta
        ok = np.abs(T) < 1e-6
        # last r before the growing mode becomes visible
        idx = np.where(~ok & (r > 2*self.S.rH))[0]
        rok = r[idx[0]] if idx.size else r[-1]
        out = {}
        for frac in (0.25, 0.35, 0.5):
            sel = (r > frac*rok) & (r < rok)
            y = r[sel]**2*(1-f[sel])
            fits = []
            for K in (3, 4, 5):
                A = np.vstack([r[sel]**(-2*k) for k in range(K)]).T
                fits.append(np.linalg.lstsq(A, y, rcond=None)[0][0])
            out[frac] = fits
        allv = np.array([v for fs in out.values() for v in fs])
        return float(np.median(allv)), float(np.ptp(allv)), rok, out


if __name__ == '__main__':
    alpha = float(sys.argv[1])
    B_lo, B_hi = float(sys.argv[2]), float(sys.argv[3])
    x0f = float(sys.argv[4]) if len(sys.argv) > 4 else 0.003
    R = float(sys.argv[5]) if len(sys.argv) > 5 else 400.
    t = time.time()
    sh = Shooter(alpha, 12, R=R)
    sh.x0 = x0f*sh.S.rH
    print(f'alpha={alpha} s={2*alpha} gamma={sh.gam:.6f} (indicial err {sh.gerr:.1e}) H1={sh.S.H[1]:.8f} x0={sh.x0:.3e}', flush=True)
    sh.solve(B_lo, B_hi, verbose=True)
    mu, spread, rok, fits = sh.mass()
    print(f'B*={sh.B:.15g} mu={mu:.12f} (fit spread {spread:.1e}, r_ok={rok:.1f}) m(s)=3mu/4={0.75*mu:.12f}  [{time.time()-t:.1f}s]', flush=True)
