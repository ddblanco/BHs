"""Multiple shooting for the extremal fixed-J EGB black hole at large s.

Why: in the core (r_H << r << sqrt(alpha)) the squashing mode behaves like exp(+-4 sqrt6 alpha/r), so a single
integration in either direction loses all digits.  Short segments bound the growth per segment.

t = log(x), x = r - r_H;  Z = (g, h, h'),  g = f/x^2.   dZ/dt = (f'/x - 2g, x h', x h'').
Nodes t_0 < ... < t_K:  Z_0 = U + beta V (regular horizon family, 60-digit data, bvp.BVP.left_lin);
Z_K = large-r series with (mu, b4) (bvp.BVP.right).  Unknowns: beta, Z_1..Z_{K-1}, mu, b4 (3K).
Equations: Phi_i(Z_i) - Z_{i+1} = 0, i = 0..K-1, each row scaled by the size of Z_{i+1}.
Damped Newton; segment Jacobians by forward differences (3 extra integrations per segment).
m(s) = 3 mu/4, s = 2 alpha (code units)."""
import sys, time, json, pickle
import numpy as np
from scipy.integrate import solve_ivp
from build_crhs import crhs
from bvp import BVP


_W = {}


def _winit(alpha, dphi, nh, nfar, rtol, xf, R, qfar):
    _W['ms'] = MS(alpha, dphi, nh, nfar, rtol, pool=False, xf=xf, R=R, qfar=qfar)


def _wflow(args):
    i, Z = args
    return _W['ms'].flow(i, np.asarray(Z))


class MS:
    def __init__(self, alpha, dphi=6.0, nh=12, nfar=25, rtol=1e-12, pool=True, nproc=14, xf=0.003, R=None, Rfac=None,
                 qfar=False):
        # outer radius: R = max(30, Rfac sqrt(alpha)) if Rfac is given.  The default 25 sqrt(alpha) is
        # noise-limited at large alpha (m off by ~2e-7 at s = 49); Rfac = 8 and 12 agree to 5e-10 (ms_check.json)
        if Rfac is not None:
            R = max(30., Rfac*np.sqrt(alpha))
        self.b = BVP(alpha, xf=xf, R=R)
        b = self.b
        self.alpha = float(alpha)
        # left family at x0 = xf*rho:  Z0 = U + Bhat V,  B = Bhat rho^-gamma  (60-digit series, linear in B;
        # at x0 = 0.003 rho the O(B^2) term is negligible, see the quad check in RESUME)
        _, Yb = b.S.start(0, b.x0); _, Yg = b.S.start(1, b.x0)
        sc = b.rho**(-b.gam)
        U = [float(v) for v in Yb]; V = [float((g-bb)*sc) for g, bb in zip(Yg, Yb)]
        self.U = np.array([U[0]/b.x0**2, U[1], U[2]]); self.V = np.array([V[0]/b.x0**2, V[1], V[2]])
        self.rtol = rtol
        self.stag = 0.5        # Newton stops when |res| > stag * best after it >= 2
        # nodes: log x near the horizon, WKB phase 4 sqrt6 alpha / r in the core, log r outside
        c = 4*np.sqrt(6)*self.alpha
        r_core_in = b.rH+10*b.rho
        r_core_out = 3*np.sqrt(self.alpha)+b.rH
        # qfar: far segments (r >= r_core_out) in quad.  In double, h ~ r^2 swamps the b4/r^4 data at R
        # (relative 1e-11 at s = 84), giving a Newton floor ~2e-10 there
        self.x_far = r_core_out-b.rH if qfar else np.inf
        xs = list(np.geomspace(b.x0, 0.05*b.rho, 5))+list(np.geomspace(0.05*b.rho, 10*b.rho, nh+1)[1:])
        if r_core_out > r_core_in:
            ph_in, ph_out = c/r_core_in, c/r_core_out
            nph = max(2, int(np.ceil((ph_in-ph_out)/dphi)))
            rr = c/np.linspace(ph_in, ph_out, nph+1)
            xs += list(rr[1:]-b.rH)
        xs += list(np.geomspace(xs[-1], b.R-b.rH, nfar+1)[1:])
        self.t = np.log(np.array(xs))
        self.K = len(self.t)-1
        self.pool = None
        if pool:
            from multiprocessing import Pool
            self.pool = Pool(nproc, initializer=_winit, initargs=(alpha, dphi, nh, nfar, rtol, xf, R, qfar))

    def flows(self, tasks):
        if self.pool is None:
            return [self.flow(i, Z) for i, Z in tasks]
        return self.pool.map(_wflow, tasks, chunksize=max(1, len(tasks)//(4*14)))

    class _Budget(Exception):
        pass

    def rhsZ(self, t, Z):
        self._nev = getattr(self, '_nev', 0)+1
        if self._nev > getattr(self, '_maxev', 10**9):
            raise MS._Budget()
        x = np.exp(t)
        y = np.array([Z[0]*x*x, Z[1], Z[2]])
        d = crhs(self.b.rH+x, y, self.alpha)
        return np.array([d[0]/x-2*Z[0], x*d[1], x*d[2]])

    def flow(self, i, Z):
        if not np.all(np.isfinite(Z)):
            return np.full(3, np.nan)
        if np.exp(self.t[i+1]) <= 0.5*self.b.rho*1.000001 or np.exp(self.t[i]) >= self.x_far*0.999999:
            return self.qflow(i, Z)
        self._nev = 0; self._maxev = 300000
        try:
            Zs = np.abs(Z)+1e-3
            def blow(t, Zt):
                d = self.rhsZ(t, Zt)
                return 1e6-max(np.max(np.abs(d)/Zs), np.max(np.abs(Zt)/Zs)/1e2)
            blow.terminal = True
            g0 = abs(Z[0])
            def gzero(t, Zt):
                return Zt[0]-1e-3*g0
            gzero.terminal = True
            sol = solve_ivp(self.rhsZ, (self.t[i], self.t[i+1]), Z, method='DOP853', rtol=self.rtol, atol=1e-14,
                            events=[blow, gzero])
        except Exception:
            self._maxev = 10**9
            return np.full(3, np.nan)
        self._maxev = 10**9
        if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
            return np.full(3, np.nan)
        return sol.y[:, -1]

    def qflow(self, i, Z):
        """quad-precision segment (near the horizon double precision is noise-limited)"""
        from build_qrhs import qintegrate
        import mpmath as mp
        x0, x1 = np.exp(self.t[i]), np.exp(self.t[i+1])
        y0 = [repr(float(Z[0]*x0*x0)), repr(float(Z[1])), repr(float(Z[2]))]
        try:
            code, r, y, n = qintegrate(repr(float(self.b.rH+x0)), repr(float(self.b.rH+x1)), repr(self.alpha),
                                       y0, tol=1e-24)
        except Exception:
            return np.full(3, np.nan)
        if code != 0:
            return np.full(3, np.nan)
        return np.array([float(y[0])/x1**2, float(y[1]), float(y[2])])

    def right(self, mu, b4):
        y = self.b.right_lin(mu, b4)
        xb = self.b.R-self.b.rH
        return np.array([y[0]/xb**2, y[1], y[2]])

    def unpack(self, u):
        beta = u[0]; mu, b4 = u[-2], u[-1]
        Z = [self.U+beta*self.V]+[u[1+3*k:4+3*k] for k in range(self.K-1)]+[self.right(mu, b4)]
        return beta, mu, b4, Z

    def residual(self, u, flows=None):
        beta, mu, b4, Z = self.unpack(u)
        F = flows if flows is not None else self.flows([(i, Z[i]) for i in range(self.K)])
        res = np.concatenate([(F[i]-Z[i+1])/self.scale[i+1] for i in range(self.K)])
        return res, F

    def set_scale(self, u):
        beta, mu, b4, Z = self.unpack(u)
        self.scale = [np.maximum(np.abs(z), 1e-8) for z in Z]

    def jacobian(self, u, F):
        """block-bidiagonal Jacobian (dense assembly); forward differences of each segment flow"""
        beta, mu, b4, Z = self.unpack(u)
        n = len(u)
        J = np.zeros((n, n))
        # column map: unknown index of Z_i components
        def col(i):
            return None if i == 0 or i == self.K else 1+3*(i-1)
        tasks, hs = [], []
        for i in range(self.K):
            for j in range(3):
                h = 1e-7*max(abs(Z[i][j]), 1e-6)
                Zp = Z[i].copy(); Zp[j] += h
                tasks.append((i, Zp)); hs.append(h)
        Fp = self.flows(tasks)
        for i in range(self.K):
            rows = slice(3*i, 3*i+3)
            sc = self.scale[i+1]
            DZ = np.zeros((3, 3))
            for j in range(3):
                DZ[:, j] = (Fp[3*i+j]-F[i])/hs[3*i+j]
            if i == 0:
                J[rows, 0] = DZ@self.V/sc
            else:
                J[rows, col(i):col(i)+3] = DZ/sc[:, None]
            # -d(Z_{i+1})/d(unknowns)
            if i+1 < self.K:
                J[rows, col(i+1):col(i+1)+3] -= np.diag(1/sc)
            else:
                for k, (dm, db) in enumerate(((1e-6, 0), (0, 1e-6))):
                    dR = (self.right(mu+dm, b4+db)-self.right(mu-dm, b4-db))/(2e-6)
                    J[rows, n-2+k] = -dR/sc
        return J

    def solve(self, u0, maxit=30, tol=1e-11, verbose=True):
        u = np.array(u0, float)
        self.set_scale(u)
        res, F = self.residual(u)
        best = np.inf
        for it in range(maxit):
            nr = np.max(np.abs(res))
            if verbose:
                beta, mu, b4, Z = self.unpack(u)
                print(f'  it {it}: |res|={nr:.2e} beta={beta:.12g} mu={mu:.13f} b4={b4:.8g} m={0.75*mu:.13f}', flush=True)
            if nr < tol or (it >= 2 and nr > self.stag*best):
                break
            best = min(best, nr) if it else nr
            J = self.jacobian(u, F)
            if not np.all(np.isfinite(J)):
                bad = ~np.isfinite(J); J[bad] = 0.0
            try:
                du = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(J, -res, rcond=None)[0]
            lam = 1.0
            for bt in range(25):
                un = u+lam*du
                rn, Fn = self.residual(un)
                if np.all(np.isfinite(rn)) and np.max(np.abs(rn)) < (1-0.25*lam)*nr:
                    break
                lam *= 0.5
            if verbose:
                print(f'     step lam={lam:.3g} |du|={np.max(np.abs(du)):.2e} nan in trial={np.sum(~np.isfinite(rn))}', flush=True)
            u, res, F = un, rn, Fn
        self.u, self.res = u, float(np.max(np.abs(res)))
        return u

    # ------------------------------------------------------------------ guesses
    def guess_from_profile(self, t, Y, mu, b4, alpha_prev=None):
        """Y = (f, h, h') on a t-grid (any alpha); nodes interpolated in x; beta = h at x1"""
        b = self.b
        x = np.exp(self.t)
        Zs = []
        for k in range(1, self.K):
            f = np.interp(self.t[k], t, Y[0]); h = np.interp(self.t[k], t, Y[1]); hp = np.interp(self.t[k], t, Y[2])
            Zs.append([f/x[k]**2, h, hp])
        # correct the near-horizon part by the change of the exact throat data (H0, H1, F0) between alphas,
        # faded out over ~10 rho
        if alpha_prev is not None and abs(alpha_prev-self.alpha) > 1e-12:
            from mp_series import cached
            So = cached(alpha_prev); Sn = b.S
            dH0 = float(Sn.H[0]-So.H[0]); dH1 = float(Sn.H[1]-So.H[1]); rF = float(Sn.F[0]/So.F[0])
            for k in range(1, self.K):
                w = np.exp(-x[k]/(10*b.rho))
                Zs[k-1][0] *= (1+w*(rF-1))
                Zs[k-1][1] += w*(dH0+dH1*x[k])
                Zs[k-1][2] += w*dH1
            beta = 0.0
        else:
            beta = 0.0
        Zs = [np.array(z, float) for z in Zs]
        beta = self.refine_beta(beta, Zs)
        return np.r_[beta, np.ravel(Zs), mu, b4]

    def chain(self, B, kstar):
        Z = self.U+B*self.V
        out = []
        for k in range(kstar):
            Z = self.flow(k, Z)
            out.append(Z)
            if not np.all(np.isfinite(Z)):
                break
        return out

    def refine_beta(self, beta, Zs):
        """Bhat by Gauss-Newton on |chain(Bhat)[k*] - Z_target(k*)|, k* = first node with x >= 0.05 rho;
        nodes below k* are then replaced by the chain (the old profiles do not reach that close to the horizon).
        Zs: list of node states Z_1..Z_{K-1} (modified in place)."""
        x = np.exp(self.t)
        kstar = int(np.searchsorted(x, 0.5*self.b.rho*0.999999))
        target = np.array(Zs[kstar-1]); sc = np.abs(target)+1e-3
        B = 0.0 if not np.isfinite(beta) else beta
        for it in range(8):
            c0 = self.chain(B, kstar); d = 1e-3*max(1, abs(B)); c1 = self.chain(B+d, kstar)
            if len(c0) < kstar or len(c1) < kstar or not (np.all(np.isfinite(c0[-1])) and np.all(np.isfinite(c1[-1]))):
                B = 0.5*B if B else 1e-3; continue
            g = (c1[-1]-c0[-1])/d
            dB = -np.dot(g/sc, (c0[-1]-target)/sc)/np.dot(g/sc, g/sc)
            B += dB
            if abs(dB) < 1e-12*max(1, abs(B)):
                break
        c = self.chain(B, kstar)
        if len(c) == kstar and np.all(np.isfinite(c[-1])):
            for k in range(kstar):
                Zs[k] = np.array(c[k])
            self.guess_err = float(np.max(np.abs(c[-1]-target)/sc))
        else:
            self.guess_err = np.inf
        return float(B)

    def guess_secant(self, P0, P1, a0, a1):
        """linear extrapolation in alpha of two converged solutions P = (t, Y, beta, mu, b4) onto this node set"""
        lam = (self.alpha-a1)/(a1-a0)
        x = np.exp(self.t)
        def nodes(P):
            t, Y = P[0], P[1]
            return np.array([[np.interp(self.t[k], t, Y[0])/x[k]**2, np.interp(self.t[k], t, Y[1]),
                              np.interp(self.t[k], t, Y[2])] for k in range(1, self.K)])
        Z0, Z1 = nodes(P0), nodes(P1)
        Z = Z1+lam*(Z1-Z0)
        beta = P1[2]+lam*(P1[2]-P0[2]); mu = P1[3]+lam*(P1[3]-P0[3]); b4 = P1[4]+lam*(P1[4]-P0[4])
        Zl = [np.array(z, float) for z in Z]
        beta = self.refine_beta(0.0 if not np.isfinite(beta) else beta, Zl)
        return np.r_[beta, np.ravel(Zl), mu, b4]

    def profile(self, npts=40):
        """dense profile: each segment re-integrated from its converged node state"""
        beta, mu, b4, Z = self.unpack(self.u)
        ts, Ys = [], []
        for i in range(self.K):
            te = np.linspace(self.t[i], self.t[i+1], npts)
            self._nev = 0; self._maxev = 20000      # the profile is only a continuation guess: cap each segment
            try:
                sol = solve_ivp(self.rhsZ, (self.t[i], self.t[i+1]), Z[i], method='DOP853', rtol=1e-10,
                                atol=1e-14, t_eval=te)
                ok = sol.status == 0 and sol.y.shape[1] == npts
            except MS._Budget:
                ok = False
            self._maxev = 10**9
            if not ok:                               # fall back to the two node states

                class _S:
                    pass
                sol = _S()
                Zi1 = Z[i+1] if i < self.K-1 else self.right(mu, b4)
                sol.t = np.array([self.t[i], self.t[i+1]]); sol.y = np.array([Z[i], Zi1]).T
            x = np.exp(sol.t)
            ts.append(sol.t[:-1] if i < self.K-1 else sol.t)
            Yi = np.vstack([sol.y[0]*x**2, sol.y[1], sol.y[2]])
            Ys.append(Yi[:, :-1] if i < self.K-1 else Yi)
        return np.concatenate(ts), np.hstack(Ys), beta, mu, b4
