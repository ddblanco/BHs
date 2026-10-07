"""Spectral solver for EXTREMAL equal-spin EGB black holes, r_H = 1.

Unknowns B, F, H, W on Chebyshev-Lobatto nodes in x = 1 - 1/r in [0, 1]
(x = 0 the degenerate horizon, x = 1 infinity), with
    b = (1-z^2)^2 B,  f = (1-z^2)^2 F,  h = (1+z^4 H)/z^2,  w = z^4 W,  z = 1-x.
Rows:
  * interior nodes: E_b, E_g, E_h, E_w of _extremal_generated.py;
  * horizon node x = 0: the same four residuals, which there reduce to the
    three near-horizon relations (E_b, E_g, E_h) and the regular w-equation;
  * infinity x = 1: B = 1, F = 1, H_x = 0, W_x = 0 (as in the project's
    non-extremal solver).
The horizon angular velocity Omega_H = W(0) is an output: at fixed r_H and
alpha the extremal solution is isolated, there is no spin parameter.

The Jacobian is assembled by the chain rule from complex-step derivatives of
the residual with respect to each jet (value, first, second derivative), times
the identity / D / D^2 collocation matrices; it is checked against a dense
complex-step Jacobian in `check_jacobian`.

The horizon expansion contains rho^gamma with non-integer gamma(alpha) in
(1.77, 2) (nearhorizon.py), so Chebyshev convergence in x is algebraic, not
geometric; the resolution study in `convergence` measures it.
"""
import numpy as np

from _extremal_generated import residual as _equations, constraint as _constraint

H_STEP = 1e-30


def cheb(n):
    t = np.cos(np.pi*np.arange(n)/(n-1))          # descending: t=1 first
    x = (1-t)/2                                   # x ascending: 0 ... 1
    c = np.ones(n); c[[0, -1]] = 2; c *= (-1.)**np.arange(n)
    dt = t[:, None]-t[None, :]
    D = (c[:, None]/c[None, :])/(dt+np.eye(n))
    D -= np.diag(D.sum(axis=1))
    D = -2*D                                      # d/dx = -2 d/dt
    return x, D, D@D


def unpack(flat, n):
    return flat.reshape(4, n)


def jets(u, D, D2):
    B, F, H, W = u
    return [B, u[0]@D.T, u[0]@D2.T, F, u[1]@D.T, H, u[2]@D.T, u[2]@D2.T,
            W, u[3]@D.T, u[3]@D2.T]


# jet index -> (field, derivative order)
JET_MAP = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
           (3, 0), (3, 1), (3, 2)]


class Extremal:
    def __init__(self, n):
        self.n = n
        self.x, self.D, self.D2 = cheb(n)
        self.mats = [np.eye(n), self.D, self.D2]
        # row scaling: the equations are of very different size; normalise
        # each equation row by a fixed weight computed once at the guess
        self.weights = None
        self.P = None

    def precondition(self, E):
        """Per-node 4x4 row combination (frozen): at interior nodes the inverse
        of the coefficient matrix of the highest derivatives (F_x, B_xx, H_xx,
        W_xx), i.e. the numerical analogue of solving the system for them."""
        if self.P is None:
            return E
        return np.einsum('nij,jn->in', self.P, E)

    def equations(self, u, alpha):
        J = jets(u, self.D, self.D2)
        return np.array(_equations(self.x, *J, alpha))

    def residual(self, flat, alpha):
        n = self.n
        u = unpack(flat, n)
        E = self.equations(u, alpha)
        out = self.precondition(E)
        # infinity row (index n-1, x = 1)
        out[:, -1] = [u[0, -1]-1, u[1, -1]-1, (self.D@u[2])[-1], (self.D@u[3])[-1]]
        if self.weights is not None:
            out = out*self.weights
        return out.ravel()

    def jacobian(self, flat, alpha):
        n = self.n
        u = unpack(flat, n)
        J = jets(u, self.D, self.D2)
        Jac = np.zeros((4, n, 4, n))
        for k, (field, order) in enumerate(JET_MAP):
            Jc = [j.astype(complex) for j in J]
            Jc[k] = Jc[k]+1j*H_STEP
            dE = np.array(_equations(self.x, *Jc, alpha)).imag/H_STEP   # (4, n)
            M = self.mats[order]
            for eqn in range(4):
                Jac[eqn, :, field, :] += dE[eqn][:, None]*M
        if self.P is not None:
            Jac = np.einsum('nij,jnkm->inkm', self.P, Jac)
        # infinity rows
        Jac[:, -1, :, :] = 0
        Jac[0, -1, 0, -1] = 1
        Jac[1, -1, 1, -1] = 1
        Jac[2, -1, 2, :] = self.D[-1]
        Jac[3, -1, 3, :] = self.D[-1]
        if self.weights is not None:
            Jac = Jac*self.weights[:, :, None, None]
        return Jac.reshape(4*n, 4*n)

    def check_jacobian(self, flat, alpha):
        """Dense complex-step Jacobian of the full residual, column by column."""
        n = self.n
        dense = np.zeros((4*n, 4*n))
        for col in range(4*n):
            v = flat.astype(complex)
            v[col] += 1j*H_STEP
            u = unpack(v, n)
            E = np.array(_equations(self.x, *jets(u, self.D, self.D2), alpha))
            E = self.precondition(E)
            E[:, -1] = [u[0, -1]-1, u[1, -1]-1, (self.D@u[2])[-1], (self.D@u[3])[-1]]
            if self.weights is not None:
                E = E*self.weights
            dense[:, col] = E.ravel().imag/H_STEP
        A = self.jacobian(flat, alpha)
        return np.max(np.abs(A-dense))/np.max(np.abs(dense))

    def solve(self, alpha, guess, tol=1e-9, max_iter=60, verbose=False):
        """Damped Newton. Stops when the residual is below `tol` and the
        Newton step has stopped shrinking (round-off floor), or the step is
        below 1e-13."""
        flat = np.asarray(guess, dtype=float).ravel().copy()
        self.set_weights(flat, alpha)       # refreshed at every solve
        norm = np.max(np.abs(self.residual(flat, alpha)))
        previous = np.inf
        for it in range(max_iter):
            A = self.jacobian(flat, alpha)
            r = self.residual(flat, alpha)
            step = np.linalg.solve(A, -r)
            lam, accepted = 1., False
            while lam > 1e-4:
                trial = flat+lam*step
                new = np.max(np.abs(self.residual(trial, alpha)))
                if np.isfinite(new) and new < (1-1e-4*lam)*norm:
                    accepted = True
                    break
                lam /= 2
            if not accepted:          # no descent: stop, never accept uphill
                break
            flat, norm = trial, new
            if verbose:
                print(f'   it {it}: |R| = {norm:.3e}, step {np.max(np.abs(step)):.2e}, lam {lam:g}')
            size = np.max(np.abs(lam*step))
            if size < 1e-13 or (norm < tol and it > 0 and size > 0.3*previous):
                break
            previous = size
        cond = np.linalg.cond(self.jacobian(flat, alpha))
        return flat, dict(residual=float(norm), iterations=it+1, condition=float(cond))

    def set_preconditioner(self, flat, alpha):
        u = unpack(flat, self.n)
        J = jets(u, self.D, self.D2)
        top = [4, 2, 7, 10]             # jet indices of Fx, Bxx, Hxx, Wxx
        C = np.zeros((self.n, 4, 4))
        for col, k in enumerate(top):
            Jc = [j.astype(complex) for j in J]
            Jc[k] = Jc[k]+1j*H_STEP
            dE = np.array(_equations(self.x, *Jc, alpha)).imag/H_STEP   # (4, n)
            C[:, :, col] = dE.T
        P = np.tile(np.eye(4), (self.n, 1, 1))
        for node in range(1, self.n-1):           # horizon and infinity keep raw rows
            P[node] = np.linalg.inv(C[node])
        self.P = P

    def set_weights(self, flat, alpha):
        """Row equilibration, one positive weight per (equation, node): the
        inverse of the largest sensitivity of that row to any jet, at the
        guess, times the size of the collocation matrix that jet enters
        through. It changes the conditioning, never the solution set."""
        self.set_preconditioner(flat, alpha)
        u = unpack(flat, self.n)
        J = jets(u, self.D, self.D2)
        scale = np.zeros((4, self.n))
        for k, (field, order) in enumerate(JET_MAP):
            Jc = [j.astype(complex) for j in J]
            Jc[k] = Jc[k]+1j*H_STEP
            dE = np.abs(self.precondition(np.array(_equations(self.x, *Jc, alpha)).imag/H_STEP))
            # the jet's own magnitude per node row is what multiplies M_k
            scale = np.maximum(scale, dE*np.abs(self.mats[order]).max(axis=1))
        w = 1/np.where(scale > 0, scale, 1.)
        w[:, -1] = 1.
        self.weights = w

    # ------------------------------------------------------------------
    # observables
    def interpolate(self, flat, n_new):
        """Chebyshev interpolation to another resolution (for continuation)."""
        u = unpack(flat, self.n)
        t_old = 1-2*self.x
        coeffs = [np.polynomial.chebyshev.chebfit(t_old, ui, self.n-1) for ui in u]
        x_new = cheb(n_new)[0]
        return np.array([np.polynomial.chebyshev.chebval(1-2*x_new, c) for c in coeffs]).ravel()

    def observables(self, flat, alpha):
        u = unpack(flat, self.n)
        B, F, H, W = u
        D, D2 = self.D, self.D2
        # infinity: z = 1-x, d/dz = -d/dx; B = 1 + B1 z + B2 z^2 + ...
        Bz = -(D@B)[-1]
        Bzz = (D2@B)[-1]
        U = Bzz/2-2                     # b = (1-z^2)^2 B = 1 + (B2-2) z^2 + ...
        M = -3*np.pi*U/8
        Jm = np.pi*W[-1]/4
        Omega = W[0]
        h_H = 1+H[0]
        S = np.pi**2/2*np.sqrt(h_H)*(1+2*alpha*(8-2*h_H))
        A = 2*np.pi**2*np.sqrt(h_H)
        cons = _constraint(self.x, *jets(u, D, D2), alpha)
        return dict(alpha=alpha, M=M, J=Jm, Omega_H=Omega, S=S, A_H=A, h_H=h_H,
                    B_H=B[0], F_H=F[0], Bz_inf=Bz,
                    constraint=float(np.max(np.abs(cons))))


def invariants(o):
    M, J, a = o['M'], o['J'], o['alpha']
    return dict(y=a/J**(2/3), mu=M/J**(2/3), sigma=o['S']/J, omega=o['Omega_H']*J**(1/3),
                x=3*np.pi*a/(4*M), j=(1.5)**1.5*np.sqrt(np.pi)*J/M**1.5,
                aH=3/32*np.sqrt(3/(2*np.pi))*o['A_H']/M**1.5,
                s=3/8*np.sqrt(3/(2*np.pi))*o['S']/M**1.5)


def myers_perry(x):
    z = 1-x
    return np.array([1/(1+z**4), np.ones_like(x), np.ones_like(x),
                     np.sqrt(2)/(1+z**4)]).ravel()
