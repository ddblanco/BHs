"""Global boundary-value solver for the extremal fixed-J EGB black hole (needed at large s: in the core the
squashing mode behaves like exp(+-4 sqrt6 alpha / r), so any initial-value shooting fails).

Variable t = log(x), x = r - r_H, on [log x0, log(R - r_H)];  Y = (f, h, h'),  dY/dt = x * F(r, Y).
Parameters p = (Bhat, mu, b4):
  left  (x0 = xf * radius):  Y = 60-digit horizon series + B gamma-mode,  B = Bhat * radius^-gamma  (3 BCs)
  right (R):                 Y = large-r series with mass mu and squashing b4                        (3 BCs)
scipy solve_bvp (4th-order collocation, adaptive mesh).  m(s) = 3 mu / 4, s = 2 alpha (code units)."""
import ctypes, json, sys, time
import numpy as np
import mpmath as mp
from scipy.integrate import solve_bvp
from build_crhs import load as load_c
from mp_series import cached
from asym_numeric import solve_seq

_lib = load_c()
_lib.rhs_many.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                          ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
_P = ctypes.POINTER(ctypes.c_double)


def rhs_many(r, Y, alpha):
    r = np.ascontiguousarray(r, float); y = np.ascontiguousarray(Y.T, float)
    out = np.empty_like(y)
    _lib.rhs_many(len(r), r.ctypes.data_as(_P), y.ctypes.data_as(_P), float(alpha), out.ctypes.data_as(_P))
    return out.T


class BVP:
    def __init__(self, alpha, xf=0.03, R=None, M=16):
        self.alpha = float(alpha)
        self.S = cached(alpha)
        self.rH = float(self.S.rH)
        self.rho = float(self.S.radius())
        self.gam = float(self.S.gamma)
        self.x0 = xf*self.rho
        self.R = R or max(40., 25*np.sqrt(alpha))
        self.ta, self.tb = np.log(self.x0), np.log(self.R-self.rH)
        self.M = M
        self._left = {}

    def fun(self, t, Z, p):
        """unknowns Z = (g, h, h'), g = f/x^2 (O(1) at the horizon; f itself is ~1e-7 there)"""
        x = np.exp(t)
        Y = np.vstack([Z[0]*x**2, Z[1], Z[2]])
        d = rhs_many(self.rH+x, Y, self.alpha)
        return np.vstack([d[0]/x-2*Z[0], x*d[1], x*d[2]])

    def left(self, Bhat):
        B = Bhat*self.rho**(-self.gam)
        r0, y0 = self.S.start(B, self.x0)          # linear in B: cache the two pieces
        return np.array([float(v) for v in y0])

    def left_lin(self):
        """Y(x0) = U + beta V on the regular family, beta = h(x1), x1 = 3 x0; formed at 60 digits so the
        divergent background/gamma-mode pieces cancel exactly near a resonance gamma = integer"""
        if not self._left:
            import mpmath as mp
            x1 = self.x0
            _, Yb0 = self.S.start(0, self.x0); _, Yg0 = self.S.start(1, self.x0)
            _, Yb1 = self.S.start(0, x1); _, Yg1 = self.S.start(1, x1)
            Yg0 = [g-b for g, b in zip(Yg0, Yb0)]; hg1 = Yg1[1]-Yb1[1]
            U = [Yb0[i]-Yb1[1]*Yg0[i]/hg1 for i in range(3)]
            V = [Yg0[i]/hg1 for i in range(3)]
            self._left = (np.array([float(v) for v in U]), np.array([float(v) for v in V]))
            self.x1 = x1
        return self._left

    def family(self, beta, x):
        """member of the regular horizon family with h(x1) = beta, evaluated at x (60-digit arithmetic)"""
        self.left_lin()
        _, Yb1 = self.S.start(0, self.x1); _, Yg1 = self.S.start(1, self.x1)
        B = (beta-Yb1[1])/(Yg1[1]-Yb1[1])
        return np.array([float(v) for v in self.S.start(B, x)[1]])

    def right(self, mu, b4):
        a, b, chk = solve_seq(self.alpha, mu, b4, M=self.M)
        z = 1/self.R; k = np.arange(len(a))
        Ht = np.sum(b*z**k); Htp = np.sum(k[1:]*b[1:]*z**(k[1:]-1))
        return np.array([np.sum(a*z**k), Ht/z**2, (2*Ht-z*Htp)/z])

    def right_lin(self, mu, b4, d=1e-6):
        """the large-r data are smooth in (mu, b4): local linear model refreshed when far from the base point"""
        key = getattr(self, '_rkey', None)
        if key is None or abs(mu-key[0]) > 1e-3*max(1, abs(key[0])) or abs(b4-key[1]) > 1e-3*max(1, abs(key[1])):
            y = self.right(mu, b4)
            dmu = (self.right(mu+d, b4)-self.right(mu-d, b4))/(2*d)
            db = (self.right(mu, b4+d)-self.right(mu, b4-d))/(2*d)
            self._rkey = (mu, b4, y, dmu, db)
        m0, b0, y, dmu, db = self._rkey
        return y+dmu*(mu-m0)+db*(b4-b0)

    def bc(self, Ya, Yb, p):
        Bh, mu, b4 = p
        y00, dy = self.left_lin()
        L = y00+Bh*dy
        L = np.array([L[0]/self.x0**2, L[1], L[2]])
        Rr = self.right_lin(mu, b4)
        xb = self.R-self.rH
        Yb = np.array([Yb[0]*xb**2, Yb[1], Yb[2]])
        # rows scaled so that mu and b4 enter at O(1): f ~ 1 - mu/R^2, h ~ R^2 + ... + b4/R^2, h' ~ ... - 2 b4/R^3
        sc = np.array([self.R**2, self.R**2, self.R**3])
        return np.r_[(Ya-L), (Yb-Rr)*sc]

    def solve(self, t, Y, p, tol=1e-8, max_nodes=200000, verbose=0):
        """Y = (f, h, h') in and out; internally g = f/x^2"""
        Z = np.vstack([Y[0]/np.exp(2*t), Y[1], Y[2]])
        sol = solve_bvp(self.fun, self.bc, t, Z, p=p, tol=tol, max_nodes=max_nodes, verbose=verbose, bc_tol=1e-10)
        self._rkey = None          # refresh the linear model of the large-r data at the solution and polish
        sol = solve_bvp(self.fun, self.bc, sol.x, sol.y, p=sol.p, tol=tol, max_nodes=max_nodes, verbose=verbose, bc_tol=1e-10)
        sol.y = np.vstack([sol.y[0]*np.exp(2*sol.x), sol.y[1], sol.y[2]])
        self.sol = sol
        return sol

    def m(self):
        return 0.75*self.sol.p[1]


def guess_from_shooting(alpha, B, mu, b4, n=4000):
    """initial profile from the double two-sided solver (twoside.TwoSide) re-solved at this alpha:
    its own outward leg (its start data) and inward leg; the series fills x < its x0."""
    from twoside import TwoSide
    from scipy.integrate import solve_ivp
    from build_crhs import crhs
    from shoot import initial_data
    T = TwoSide(alpha)
    try:
        pT = T.solve([B, mu, b4])
        assert T.err < 1e-6
    except Exception:
        pT = T.solve(T.guess())
    print("TwoSide re-solved: m =", 0.75*pT[1], "err", T.err, flush=True)
    b = BVP(alpha)
    S = T.sh.S
    y0 = initial_data(S, T.sh.gam, T.sh.nv, pT[0], T.sh.x0, T.gser)
    so = solve_ivp(crhs, (S.rH+T.sh.x0, T.rm), y0, args=(alpha,), method='DOP853', rtol=1e-12, atol=1e-15, dense_output=True)
    si = solve_ivp(crhs, (b.R, T.rm), b.right(pT[1], pT[2]), args=(alpha,), method='DOP853', rtol=1e-12, atol=1e-15, dense_output=True)
    t = np.linspace(b.ta, b.tb, n); x = np.exp(t); r = b.rH+x
    Bh = pT[0]*b.rho**b.gam
    Y = np.empty((3, n))
    for i, (xi, ri) in enumerate(zip(x, r)):
        if ri < S.rH+T.sh.x0:
            Y[:, i] = [float(v) for v in b.S.start(pT[0], xi)[1]]
        elif ri < T.rm:
            Y[:, i] = so.sol(ri)
        else:
            Y[:, i] = si.sol(ri)
    return b, t, Y, np.array([Bh, pT[1], pT[2]])


if __name__ == '__main__':
    alpha = float(sys.argv[1])
    d = {round(r['alpha'], 6): r for r in json.load(open('sweep.json'))}
    r = d[round(alpha, 6)]
    t0 = time.time()
    b, t, Y, p = guess_from_shooting(alpha, r['B'], r['mu'], r['b4'])
    print('guess built', time.time()-t0, 'p0', p, flush=True)
    for tol in (1e-6, 1e-8, 1e-10):
        sol = b.solve(t, Y, p, tol=tol)
        print(f'tol={tol}: status {sol.status} ({sol.message}) nodes {len(sol.x)} p={sol.p} m={0.75*sol.p[1]:.12f} '
              f'(shooting m = {r["m"]:.12f}) [{time.time()-t0:.0f}s]', flush=True)
        t, Y, p = sol.x, sol.y, sol.p
