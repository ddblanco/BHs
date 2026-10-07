"""Extremal solver of extremal_r_solver.py with the asymptotic structure built
into the unknowns (the production solver).

Unknowns Q_i at the Chebyshev nodes in s = r/(1+r), with
    P_i = 1 + (1-s)^2 Q_i  (i = 1, 2, 3),   W = (1-s)^2 Q_4 .
P is a polynomial of degree N+1, so its jets come from the product rule with
the spectral derivatives of Q (differentiating its nodal values with the
degree-(N-1) matrix D would lose Q(1)). This makes asymptotic flatness and the
residual-gauge condition (no 1/r term in P2) structural, and the mass enters
as a VALUE at infinity, c_t = Q_1(1)/2, which converges like the function
values; in the P representation it was P1''(1), weighted by k^4 in the
Chebyshev tail and limited by the r^(2 gamma) term at the horizon.

Rows: s = 0: dP_i/ds = 0 (i = 1..3) and E_W; first interior node: E_W
replaced by g_H = P2(0) = 1; all four equations vanish identically at s = 1
in this representation and are collocated at s* midway between the last two
nodes instead.
"""
import numpy as np

from extremal_r_solver import ExtremalR, JET_MAP, H_STEP, cheb, invariants  # noqa: F401
from _kkr_generated_p0 import residual as _equations, constraint as _constraint
from _pw_generated_p0 import J_of

CH = np.polynomial.chebyshev


class ExtremalRQ(ExtremalR):
    def __init__(self, n, g_horizon=1.):
        super().__init__(n, g_horizon)
        x = self.x
        self.offset = np.array([1., 1., 1., 0.])
        w, w1, w2 = (1-x)**2, -2*(1-x), 2*np.ones_like(x)
        D, D2 = self.D, self.D2
        self.qmats = [np.diag(w), np.diag(w1)+np.diag(w)@D,
                      np.diag(w2)+2*np.diag(w1)@D+np.diag(w)@D2]
        self.s_star = (x[-2]+x[-1])/2
        t, ts = 1-2*x, 1-2*self.s_star
        I = np.eye(n)
        C = np.array([CH.chebfit(t, I[:, k], n-1) for k in range(n)]).T
        L0 = np.array([CH.chebval(ts, C[:, k]) for k in range(n)])
        L1 = np.array([-2*CH.chebval(ts, CH.chebder(C[:, k])) for k in range(n)])
        L2 = np.array([4*CH.chebval(ts, CH.chebder(C[:, k], 2)) for k in range(n)])
        ws, ws1, ws2 = (1-self.s_star)**2, -2*(1-self.s_star), 2.
        self.smats = [ws*L0, ws1*L0+ws*L1, ws2*L0+2*ws1*L1+ws*L2]

    # jets of P at the nodes and at s*
    def qjets(self, q):
        out = []
        for i in range(4):
            out += [self.offset[i]*(k == 0)+self.qmats[k]@q[i] for k in range(3)]
        return out

    def sjets(self, q):
        out = []
        for i in range(4):
            out += [np.atleast_1d(self.offset[i]*(k == 0)+self.smats[k]@q[i]) for k in range(3)]
        return out

    def to_P(self, q):
        return np.array([self.offset[i]+self.qmats[0]@q[i] for i in range(4)])

    def rows(self, q, alpha):
        J = self.qjets(q)
        E = np.array(_equations(self.x, self.zeta, *J, alpha))
        for i in range(3):
            E[i, 0] = J[3*i+1][0]
        E[3, 1] = J[3][0]-self.g_horizon
        s_ = np.array([self.s_star])
        E[:, -1] = np.array(_equations(s_, 1-s_, *self.sjets(q), alpha))[:, 0]
        return E

    def residual(self, flat, alpha):
        E = self.rows(flat.reshape(4, self.n), alpha)
        if self.weights is not None:
            E = E*self.weights
        return E.ravel()

    def jacobian(self, flat, alpha):
        n = self.n
        q = flat.reshape(4, n)
        J = self.qjets(q)
        Jac = np.zeros((4, n, 4, n))
        for k, (field, order) in enumerate(JET_MAP):
            Jc = [j.astype(complex) for j in J]
            Jc[k] = Jc[k]+1j*H_STEP
            dE = np.array(_equations(self.x, self.zeta, *Jc, alpha)).imag/H_STEP
            for eqn in range(4):
                Jac[eqn, :, field, :] += dE[eqn][:, None]*self.qmats[order]
        for i in range(3):
            Jac[i, 0] = 0
            Jac[i, 0, i, :] = self.qmats[1][0]
        Jac[3, 1] = 0
        Jac[3, 1, 1, :] = self.qmats[0][0]
        Js = self.sjets(q)
        s_ = np.array([self.s_star])
        Jac[:, -1] = 0
        for k, (field, order) in enumerate(JET_MAP):
            Jc = [j.astype(complex) for j in Js]
            Jc[k] = Jc[k]+1j*H_STEP
            dE = np.array(_equations(s_, 1-s_, *Jc, alpha))[:, 0].imag/H_STEP
            for eqn in range(4):
                Jac[eqn, -1, field, :] += dE[eqn]*self.smats[order]
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

    def observables(self, flat, alpha):
        q = flat.reshape(4, self.n)
        J = self.qjets(q)
        P1, P2, P3, W = J[0], J[3], J[6], J[9]
        c_t = q[0, -1]/2
        k = (self.x > 0.05) & (self.x < 0.9)
        with np.errstate(all='ignore'):
            J_nodes = J_of(self.x[k], P1[k], J[1][k], P2[k], J[4][k], P3[k], J[7][k],
                           W[k], J[10][k], alpha)
        g, h = P2[0], P2[0]*P3[0]*2.
        cons = np.array(_constraint(self.x, self.zeta, *J, alpha))
        E = np.array(_equations(self.x, self.zeta, *J, alpha))
        return dict(alpha=alpha, M=float(3*np.pi*(1-c_t)/4), c_t=float(c_t),
                    J=float(np.median(J_nodes)), J_spread=float(np.ptp(J_nodes)),
                    c_w=float(q[3, -1]), Omega_H=float(np.sqrt(2)/2+W[0]),
                    A_H=float(2*np.pi**2*g*np.sqrt(h)),
                    S=float(np.pi**2/2*g*np.sqrt(h)*(1+2*alpha*(8/g-2*h/g**2))),
                    g_H=float(g), h_H=float(h), P1_H=float(P1[0]), P3_H=float(P3[0]),
                    constraint=float(np.max(np.abs(cons[1:-1]))),
                    replaced_rows=[float(E[3, 1])])

    def interpolate(self, flat, n_new):
        x_new = cheb(n_new)[0]
        return np.array([CH.chebval(1-2*x_new, c) for c in self.coefficients(flat)]).ravel()

    def from_P(self, other, flat_P):
        """Q on this grid from a solution of the P-representation solver."""
        u = flat_P.reshape(4, other.n)
        cP = [CH.chebfit(1-2*other.x, ui, other.n-1) for ui in u]
        q = np.empty((4, self.n))
        inner = self.x < 1-1e-9
        for i in range(4):
            vals = (CH.chebval(1-2*self.x, cP[i])-self.offset[i])/(1-self.x)**2
            c = CH.chebfit(1-2*self.x[inner], vals[inner], self.n-2)
            q[i] = CH.chebval(1-2*self.x, c)
        return q.ravel()


def myers_perry_q(n):
    return np.zeros(4*n)
