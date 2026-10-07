"""Spectral solver for extremal equal-spin EGB black holes in the gauge of
arXiv:2303.12471 (kkr_equations.py), a = 1.

Unknowns P1, P2, P3, W on Chebyshev-Lobatto nodes in xi = r^2/(1+r^2):
xi = 0 is the degenerate horizon, xi = 1 infinity. Rows:
  * every node except infinity: E_P1, E_P2, E_P3, E_W (at xi = 0 they reduce
    to the near-horizon relations and the regular w-equation);
  * infinity: P1 = P2 = P3 = 1, W = 0 (asymptotic flatness, their sec. 3.1).
Rows are equilibrated per node (positive weights, frozen at the start of each
solve); the Jacobian is assembled by the chain rule from complex-step jet
derivatives and checked against a dense complex-step Jacobian in the tests.

Observables (their eq. 3.3, a = 1, our per-plane J):
    M = 3 pi (1 - c_t)/4,  c_t from P1 = 1 + 2 c_t (1-xi) + ...  at xi -> 1,
    J = pi (c_w + sqrt 2)/4, c_w from W = c_w (1-xi)^2 + ...,
and, independently, J from the conserved w-momentum at the horizon.
"""
import importlib

import numpy as np

H_STEP = 1e-30
JET_MAP = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2),
           (3, 0), (3, 1), (3, 2)]


def cheb(n):
    t = np.cos(np.pi*np.arange(n)/(n-1))          # descending
    x = (1-t)/2                                   # ascending 0 ... 1
    zeta = (1+t)/2                                # 1 - x, without cancellation
    c = np.ones(n); c[[0, -1]] = 2; c *= (-1.)**np.arange(n)
    dt = t[:, None]-t[None, :]
    D = (c[:, None]/c[None, :])/(dt+np.eye(n))
    D -= np.diag(D.sum(axis=1))
    D = -2*D
    return x, D, D@D, zeta


class KKR:
    def __init__(self, n, power=2):
        """power p: radial coordinate s with xi = s^p (p = 2 is the r of the
        paper); the horizon term xi^gamma becomes s^(p gamma)."""
        module = importlib.import_module(f'_kkr_generated_p{power}')
        self._equations, self._constraint = module.residual, module.constraint
        self.power = power
        self.n = n
        self.x, self.D, self.D2, self.zeta = cheb(n)
        self.mats = [np.eye(n), self.D, self.D2]
        self.weights = None
        self.g_horizon = 1.

    def jets(self, u):
        out = []
        for i in range(4):
            out += [u[i], u[i]@self.D.T, u[i]@self.D2.T]
        return out

    def raw(self, u, alpha):
        return np.array(self._equations(self.x, self.zeta, *self.jets(u), alpha))

    def rows(self, u, alpha):
        E = self.raw(u, alpha)
        E[:, -1] = [u[0, -1]-1, u[1, -1]-1, u[2, -1]-1, u[3, -1]]
        # Horizon row of E_W replaced by a normalisation that removes the
        # zero mode of the gauge b f = fixed at fixed a and alpha: the
        # extremal solutions form a one-parameter family there (physical size
        # against a). mode 'gH': g_H = P2(0) = g_horizon (r_H = 1 when 1).
        E[3, 0] = u[1, 0]-self.g_horizon
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
            out.append(np.array(self._equations(self.x, self.zeta, *Jc, alpha)).imag/H_STEP)
        return out

    def jacobian(self, flat, alpha):
        n = self.n
        u = flat.reshape(4, n)
        Jac = np.zeros((4, n, 4, n))
        for (field, order), dE in zip(JET_MAP, self.jet_derivatives(u, alpha)):
            for eqn in range(4):
                Jac[eqn, :, field, :] += dE[eqn][:, None]*self.mats[order]
        Jac[:, -1] = 0
        for i in range(4):
            Jac[i, -1, i, -1] = 1
        Jac[3, 0] = 0
        Jac[3, 0, 1, 0] = 1
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
        u = flat.reshape(4, self.n)
        scale = np.zeros((4, self.n))
        for (field, order), dE in zip(JET_MAP, self.jet_derivatives(u, alpha)):
            scale = np.maximum(scale, np.abs(dE)*np.abs(self.mats[order]).max(axis=1))
        w = 1/np.where(scale > 0, scale, 1.)
        w[:, -1] = 1.
        w[3, 0] = 1.
        self.weights = w

    def solve(self, alpha, guess, tol=1e-11, max_iter=40, verbose=False):
        flat = np.asarray(guess, dtype=float).ravel().copy()
        self.set_weights(flat, alpha)
        norm = np.max(np.abs(self.residual(flat, alpha)))
        it = 0
        for it in range(max_iter):
            if norm < tol:
                break
            A = self.jacobian(flat, alpha)
            step = np.linalg.solve(A, -self.residual(flat, alpha))
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
            flat, norm = trial, new
            if verbose:
                print(f'   it {it}: |R| = {norm:.3e}  step {np.max(np.abs(step)):.2e}  lam {lam:g}')
        A = self.jacobian(flat, alpha)
        s = np.linalg.svd(A, compute_uv=False)
        return flat, dict(residual=float(norm), iterations=it, condition=float(s[0]/s[-1]))

    def interpolate(self, flat, n_new):
        """Chebyshev interpolation to another resolution (same power)."""
        u = flat.reshape(4, self.n)
        t = 1-2*self.x
        coeffs = [np.polynomial.chebyshev.chebfit(t, ui, self.n-1) for ui in u]
        x_new = cheb(n_new)[0]
        return np.array([np.polynomial.chebyshev.chebval(1-2*x_new, c) for c in coeffs]).ravel()

    def coefficients(self, flat):
        """Chebyshev coefficients of each field (for convergence diagnostics)."""
        u = flat.reshape(4, self.n)
        t = 1-2*self.x
        return [np.polynomial.chebyshev.chebfit(t, ui, self.n-1) for ui in u]

    def observables(self, flat, alpha):
        u = flat.reshape(4, self.n)
        P1, P2, P3, W = u
        D, D2 = self.D, self.D2
        p = self.power                          # 1 - xi = 1 - s^p ~ p (1 - s)
        c_t = -(D@P1)[-1]/(2*p)                 # P1 = 1 + 2 c_t (1-xi) + ...
        c_w = (D2@W)[-1]/(2*p*p)                # W = c_w (1-xi)^2 + ...
        M = 3*np.pi*(1-c_t)/4
        Jm = np.pi*(c_w+np.sqrt(2))/4
        g, h = P2[0], P2[0]*P3[0]*2.            # xi = 0: u = 1, 1 + 1/u^2 = 2
        Omega = np.sqrt(2)/2+W[0]
        A = 2*np.pi**2*g*np.sqrt(h)
        S = np.pi**2/2*g*np.sqrt(h)*(1+2*alpha*(8/g-2*h/g**2))
        cons = np.array(self._constraint(self.x, self.zeta, *self.jets(u), alpha))
        # the replaced horizon row, checked a posteriori
        ew_horizon = float(self.raw(u, alpha)[3, 0]*(self.weights[3, 1] if self.weights is not None else 1.))
        return dict(alpha=alpha, M=M, J=Jm, Omega_H=Omega, A_H=A, S=S, g_H=g, h_H=h,
                    P1_H=P1[0], W_x_inf=(D@W)[-1], c_t=c_t, c_w=c_w,
                    constraint=float(np.max(np.abs(cons[:-1]))), E_W_horizon=ew_horizon)


def invariants(o):
    M, J, a = o['M'], o['J'], o['alpha']
    return dict(y=a/J**(2/3), mu=M/J**(2/3), sigma=o['S']/J, omega=o['Omega_H']*J**(1/3),
                x=3*np.pi*a/(4*M), j=(1.5)**1.5*np.sqrt(np.pi)*J/M**1.5,
                aH=3/32*np.sqrt(3/(2*np.pi))*o['A_H']/M**1.5,
                s=3/8*np.sqrt(3/(2*np.pi))*o['S']/M**1.5)


def myers_perry(n):
    return np.concatenate([np.ones(3*n), np.zeros(n)])
