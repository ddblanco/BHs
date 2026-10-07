"""Spectral solver for EXTREMAL equal-spin EGB black holes in the gauge of
arXiv:2303.12471, in their radial coordinate r compactified as s = r/(1+r).

Ansatz (a = 1, P_i = exp(2 F_i) of the paper):
    b = P1 r^4/(u^2+1),  g^{rr} = P1^-1 r^2/u,  g = P2 u,
    h = P2 P3 u (1+1/u^2),  w = sqrt(2)/(u^2+1) + W,  u = r^2+1;
residuals from kkr_equations.py (power 0), J from pw_generator.py.

Why this coordinate and these rows (each found by diagnosing the Jacobian):
  * the gauge b g^{rr} = fixed excludes nearby non-extremal solutions, which
    in the g = r^2 gauge made the collocation matrix near singular;
  * at fixed (a, alpha) the gauge still leaves two exact zero modes: the
    physical size of the black hole relative to a, and a residual radial
    gauge r -> r + c sqrt(b g^{rr}) (~ r^3 at the horizon, a constant shift
    at infinity). They are fixed by
        g_H = P2(0) = 1                (horizon radius r_H = 1, as in the
                                         project's g = r^2 solver), and
        dP2/ds(1) = 0                   (no 1/r term in P2 at infinity),
    replacing the E_W and E_P2 rows at the first interior node, where the
    left null vectors of the unfixed system sit;
  * in r the residual-gauge term r^3 and the 1/r^3 tails are smooth; the
    only non-smooth term left is r^(2 gamma) at the horizon (nearhorizon.py);
  * at s = 0 the E_P1, E_P2, E_P3 rows vanish identically and are replaced by
    dP_i/ds = 0 (no odd r^1 term: the residual gauge starts at r^3); the E_W
    row is kept there;
  * at s = 1: P1 = P2 = P3 = 1, W = 0.

Observables:
    M = 3 pi (1 - c_t)/4 with P1 = 1 + 2 c_t/r^2 + ...  ->  c_t = P1''(s=1)/4,
    J from the conserved w-momentum (no derivative at infinity); the tail
      value J_inf = pi (c_w + sqrt 2)/4 from W = c_w/r^4 needs W''''(1), whose
      round-off grows like N^8: kept only as a low-resolution diagnostic,
    Omega_H = sqrt(2)/2 + W(0), S Wald, A_H, constraint E_f (diagnostic).
"""
import numpy as np

from _kkr_generated_p0 import residual as _equations, constraint as _constraint
from _pw_generated_p0 import J_of

H_STEP = 1e-30
JET_MAP = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2),
           (3, 0), (3, 1), (3, 2)]


def cheb(n):
    t = np.cos(np.pi*np.arange(n)/(n-1))          # descending
    x = (1-t)/2                                   # ascending: 0 horizon ... 1 infinity
    zeta = (1+t)/2                                # 1 - x without cancellation
    c = np.ones(n); c[[0, -1]] = 2; c *= (-1.)**np.arange(n)
    dt = t[:, None]-t[None, :]
    D = (c[:, None]/c[None, :])/(dt+np.eye(n))
    D -= np.diag(D.sum(axis=1))
    D = -2*D
    return x, zeta, D, D@D


class ExtremalR:
    def __init__(self, n, g_horizon=1.):
        self.n = n
        self.x, self.zeta, self.D, self.D2 = cheb(n)
        self.mats = [np.eye(n), self.D, self.D2]
        self.g_horizon = g_horizon
        self.weights = None

    # ---------------------------------------------------------------- rows
    def jets(self, u):
        out = []
        for i in range(4):
            out += [u[i], u[i]@self.D.T, u[i]@self.D2.T]
        return out

    def raw(self, u, alpha):
        return np.array(_equations(self.x, self.zeta, *self.jets(u), alpha))

    def rows(self, u, alpha):
        E = self.raw(u, alpha)
        D = self.D
        for i in range(3):                         # horizon: no r^1 term
            E[i, 0] = (D@u[i])[0]
        E[:, -1] = [u[0, -1]-1, u[1, -1]-1, u[2, -1]-1, u[3, -1]]
        E[3, 1] = u[1, 0]-self.g_horizon           # size: g_H
        E[1, 1] = (D@u[1])[-1]                     # residual gauge: no 1/r in P2
        return E

    def residual(self, flat, alpha):
        E = self.rows(flat.reshape(4, self.n), alpha)
        if self.weights is not None:
            E = E*self.weights
        return E.ravel()

    def jet_derivatives(self, u, alpha):
        J = self.jets(u)
        out = []
        for k in range(len(J)):
            Jc = [j.astype(complex) for j in J]
            Jc[k] = Jc[k]+1j*H_STEP
            out.append(np.array(_equations(self.x, self.zeta, *Jc, alpha)).imag/H_STEP)
        return out

    def jacobian(self, flat, alpha):
        n = self.n
        u = flat.reshape(4, n)
        Jac = np.zeros((4, n, 4, n))
        for (field, order), dE in zip(JET_MAP, self.jet_derivatives(u, alpha)):
            for eqn in range(4):
                Jac[eqn, :, field, :] += dE[eqn][:, None]*self.mats[order]
        for i in range(3):
            Jac[i, 0] = 0
            Jac[i, 0, i, :] = self.D[0]
        Jac[:, -1] = 0
        for i in range(4):
            Jac[i, -1, i, -1] = 1
        Jac[3, 1] = 0
        Jac[3, 1, 1, 0] = 1
        Jac[1, 1] = 0
        Jac[1, 1, 1, :] = self.D[-1]
        if self.weights is not None:
            Jac = Jac*self.weights[:, :, None, None]
        return Jac.reshape(4*n, 4*n)

    def dense_jacobian(self, flat, alpha):
        n = self.n
        out = np.zeros((4*n, 4*n))
        for col in range(4*n):
            v = flat.astype(complex)
            v[col] += 1j*H_STEP
            E = self.rows(v.reshape(4, n), alpha)
            if self.weights is not None:
                E = E*self.weights
            out[:, col] = E.ravel().imag/H_STEP
        return out

    def set_weights(self, flat, alpha):
        """Row equilibration (positive weights, frozen during a solve)."""
        self.weights = None
        A = self.jacobian(flat, alpha).reshape(4, self.n, 4*self.n)
        scale = np.abs(A).max(axis=2)
        self.weights = 1/np.where(scale > 0, scale, 1.)

    # -------------------------------------------------------------- Newton
    def solve(self, alpha, guess, tol=1e-12, max_iter=40, verbose=False):
        flat = np.asarray(guess, dtype=float).ravel().copy()
        self.set_weights(flat, alpha)
        norm = np.max(np.abs(self.residual(flat, alpha)))
        it = 0
        for it in range(max_iter):
            if norm < tol:
                break
            step = np.linalg.solve(self.jacobian(flat, alpha), -self.residual(flat, alpha))
            lam, accepted = 1., False
            while lam > 1e-6:
                trial = flat+lam*step
                new = np.max(np.abs(self.residual(trial, alpha)))
                if np.isfinite(new) and new < (1-1e-4*lam)*norm:
                    accepted = True
                    break
                lam /= 2
            if not accepted:
                break
            # at the round-off floor further steps only cost line searches
            stalled = norm < 1e-8 and new > 0.9*norm
            flat, norm = trial, new
            if stalled:
                break
            if verbose:
                print(f'   it {it}: |R| = {norm:.3e}  step {np.max(np.abs(step)):.2e}  lam {lam:g}')
        return flat, dict(residual=float(norm), iterations=it)

    def condition(self, flat, alpha):
        s = np.linalg.svd(self.jacobian(flat, alpha), compute_uv=False)
        return float(s[0]/s[-1])

    # ------------------------------------------------------- representation
    def coefficients(self, flat):
        u = flat.reshape(4, self.n)
        t = 1-2*self.x
        return [np.polynomial.chebyshev.chebfit(t, ui, self.n-1) for ui in u]

    def interpolate(self, flat, n_new):
        x_new = cheb(n_new)[0]
        return np.array([np.polynomial.chebyshev.chebval(1-2*x_new, c)
                         for c in self.coefficients(flat)]).ravel()

    def evaluate(self, flat, s, derivative=0):
        out = []
        for c in self.coefficients(flat):
            c = np.polynomial.chebyshev.chebder(c, derivative)*(-2)**derivative if derivative else c
            out.append(np.polynomial.chebyshev.chebval(1-2*np.asarray(s), c))
        return np.array(out)

    # ----------------------------------------------------------- observables
    def observables(self, flat, alpha):
        u = flat.reshape(4, self.n)
        P1, P2, P3, W = u
        D, D2 = self.D, self.D2
        c_t = (D2@P1)[-1]/4                        # P1 = 1 + 2 c_t (1-s)^2 + ...
        P1_slope_inf = (D@P1)[-1]                  # must vanish (no 1/r in P1)
        D4 = D2@D2
        c_w = (D4@W)[-1]/24                        # W = c_w (1-s)^4 + ...
        M = 3*np.pi*(1-c_t)/4
        # p_w is constant on solutions; next to s = 1 its expression loses
        # digits to cancellation, so only nodes with 0.05 < s < 0.9 are used
        k = (self.x > 0.05) & (self.x < 0.9)
        with np.errstate(all='ignore'):
            J_nodes = J_of(self.x[k], P1[k], (D@P1)[k], P2[k], (D@P2)[k],
                           P3[k], (D@P3)[k], W[k], (D@W)[k], alpha)
        Jm = float(np.median(J_nodes))
        J_inf = np.pi*(c_w+np.sqrt(2))/4
        g, h = P2[0], P2[0]*P3[0]*2.
        Omega = np.sqrt(2)/2+W[0]
        A = 2*np.pi**2*g*np.sqrt(h)
        S = np.pi**2/2*g*np.sqrt(h)*(1+2*alpha*(8/g-2*h/g**2))
        cons = np.array(_constraint(self.x, self.zeta, *self.jets(u), alpha))
        replaced = self.raw(u, alpha)
        return dict(alpha=alpha, M=M, J=Jm, J_inf=float(J_inf),
                    J_spread=float(np.max(J_nodes)-np.min(J_nodes)),
                    Omega_H=float(Omega), A_H=float(A), S=float(S), g_H=float(g), h_H=float(h),
                    P1_H=float(P1[0]), c_t=float(c_t), c_w=float(c_w),
                    P1_slope_inf=float(P1_slope_inf),
                    constraint=float(np.max(np.abs(cons[1:-1]))),
                    replaced_rows=[float(replaced[3, 1]), float(replaced[1, 1])])


def invariants(o):
    M, J, a = o['M'], o['J'], o['alpha']
    return dict(y=a/J**(2/3), mu=M/J**(2/3), sigma=o['S']/J, omega=o['Omega_H']*J**(1/3),
                x=3*np.pi*a/(4*M), j=(1.5)**1.5*np.sqrt(np.pi)*J/M**1.5,
                aH=3/32*np.sqrt(3/(2*np.pi))*o['A_H']/M**1.5,
                s=3/8*np.sqrt(3/(2*np.pi))*o['S']/M**1.5, tH=0.)


def myers_perry(n):
    return np.concatenate([np.ones(3*n), np.zeros(n)])

