"""Two-sided shooting for the extremal fixed-J EGB black hole (code units J = J0, s = 2 alpha).

outward: horizon series + gamma-mode B (extremal_shoot.Shooter) from r_H + x0 to r_m;
inward : large-r series (asym_numeric.solve_seq) with free (mu, b4) from R to r_m;
match  : (f, h, h') at r_m  ->  3 equations for (B, mu, b4); Newton with finite differences.
m(s) = 3 mu / 4.   Usage: python twoside.py alpha B_guess mu_guess b4_guess [rm R]"""
import sys, time, json
import numpy as np
from scipy.integrate import solve_ivp
from extremal_shoot import Shooter
from asym_numeric import solve_seq
from build_crhs import crhs
from shoot import initial_data


class TwoSide:
    def __init__(self, alpha, rm=None, R=None, M=16, x0f=0.003, L=12, rtol=1e-13):
        self.alpha = alpha
        self.sh = Shooter(alpha, L)
        self.sh.x0 = x0f*self.sh.S.rH
        self.rm = rm or 1.5*self.sh.S.rH+0.5*np.sqrt(alpha)
        self.R = R or max(40., 25*np.sqrt(alpha))
        self.M, self.rtol = M, rtol

    def outward(self, B):
        S = self.sh.S
        y0 = initial_data(S, self.sh.gam, self.sh.nv, B, self.sh.x0)
        sol = solve_ivp(crhs, (S.rH+self.sh.x0, self.rm), y0, args=(self.alpha,), method='DOP853',
                        rtol=self.rtol, atol=1e-15)
        return sol.y[:, -1], sol.status

    def asym_data(self, mu, b4, R):
        a, b, chk = solve_seq(self.alpha, mu, b4, M=self.M)
        z = 1/R
        k = np.arange(len(a))
        Ft = np.sum(a*z**k); Ftp = np.sum(k[1:]*a[1:]*z**(k[1:]-1))
        Ht = np.sum(b*z**k); Htp = np.sum(k[1:]*b[1:]*z**(k[1:]-1))
        f = Ft
        h = Ht/z**2
        hp = (2*Ht-z*Htp)/z      # dh/dr
        return np.array([f, h, hp]), chk

    def inward(self, mu, b4):
        y0, chk = self.asym_data(mu, b4, self.R)
        sol = solve_ivp(crhs, (self.R, self.rm), y0, args=(self.alpha,), method='DOP853',
                        rtol=self.rtol, atol=1e-15)
        return sol.y[:, -1], sol.status

    def mismatch(self, p):
        B, mu, b4 = p
        yo, so = self.outward(B)
        yi, si = self.inward(mu, b4)
        sc = np.array([1., self.rm**2, self.rm])
        return (yo-yi)/sc

    def solve(self, p0, tol=1e-13, maxit=30, verbose=False):
        p = np.array(p0, float)
        hist = []
        for it in range(maxit):
            e = self.mismatch(p)
            hist.append(np.max(np.abs(e)))
            if not np.all(np.isfinite(e)):
                raise RuntimeError('non-finite mismatch')
            if hist[-1] < 1e-11 or (len(hist) > 3 and hist[-1] > 0.3*min(hist[:-1]) and min(hist) < 1e-6):
                break
            Jm = np.empty((3, 3))
            for j in range(3):
                dp = np.zeros(3); dp[j] = 1e-6*max(1, abs(p[j]))
                Jm[:, j] = (self.mismatch(p+dp)-self.mismatch(p-dp))/(2*dp[j])
            step = np.linalg.solve(Jm, -e)
            p = p+step
            if verbose:
                print(f'  it {it}: |e|={np.max(np.abs(e)):.2e} p={p}', flush=True)
            if np.max(np.abs(step)/np.maximum(1, np.abs(p))) < tol:
                break
        self.p = p
        self.err = float(min(hist))
        self.cond = float(np.linalg.cond(Jm))
        return p


if __name__ == '__main__':
    alpha = float(sys.argv[1]); p0 = [float(v) for v in sys.argv[2:5]]
    rm = float(sys.argv[5]) if len(sys.argv) > 5 else None
    R = float(sys.argv[6]) if len(sys.argv) > 6 else None
    t = time.time()
    T = TwoSide(alpha, rm=rm, R=R)
    print(f'alpha={alpha} rm={T.rm:.4f} R={T.R} gamma={T.sh.gam:.6f}', flush=True)
    p = T.solve(p0, verbose=True)
    print(f'B={p[0]:.14g} mu={p[1]:.14f} b4={p[2]:.12g}  m(s={2*alpha})={0.75*p[1]:.14f}  cond {T.cond:.1e} [{time.time()-t:.1f}s]')
