"""Shooting solver for the extremal fixed-J EGB black hole (reduced system, areal gauge g = r^2).

Horizon:  formal series (horizon_series.HorizonSeries) + gamma-mode
          dF = B phi x^(gamma-1), dH = B psi x^gamma  (null vector of the indicial matrix);
outward:  f' = F1(r,f,h,h'), h'' = F2(r,f,h,h') with scipy DOP853;
target:   decay of the squashing at infinity, read at R from the large-r behaviour;
mass:     mu = lim r^2 (1 - f),  M = 3 pi mu / 8 ;  J = J0 = pi sqrt2/4 fixed (C2 = 32),
          so s = 2 alpha and m(s) = 2 M / pi.
"""
import numpy as np
from scipy.integrate import solve_ivp
from ode_poly import coeffs
from horizon_series import HorizonSeries, residual_series


def rhs(r, y, alpha):
    f, h, hp = y
    Ab, Bb, Cb, Ah, Bh, Ch = coeffs(np.asarray(r), np.asarray(f), np.asarray(h), np.asarray(hp), alpha)
    det = Ab*Bh-Ah*Bb
    fp = (-Cb*Bh+Ch*Bb)/det
    hpp = (-Ab*Ch+Ah*Cb)/det
    return np.array([fp, hp, hpp])


class Indicial:
    """level matrices [dNb[m]; dNt_{m+1}] wrt (F_{m-1}, H_m) at integer m, fitted as polynomials in m"""

    def __init__(self, S, mmax=12, deg=6):
        self.S = S
        ms = np.arange(2, mmax+1)
        L = max(mmax, len(S.F))
        F = np.zeros(L+2, complex); H = np.zeros(L+3, complex)
        F[0], H[0] = S.F0, S.H0
        F[1:len(S.F)] = S.F[1:]; H[1:len(S.H)] = S.H[1:]
        rows = []
        for m in ms:
            M = np.zeros((2, 2))
            for j, (w, k) in enumerate((('F', m-1), ('H', m))):
                Fc = F.copy(); Hc = H.copy()
                (Fc if w == 'F' else Hc)[k] += 1e-20j
                b, h_ = residual_series(S.rH, Fc, Hc, S.alpha, L+4)
                # kappa for level m+1 from the leading block
                M[0, j] = np.imag(b[m])/1e-20
                M[1, j] = np.imag(h_[m+2])/1e-20
            rows.append(M)
        # kappa_{m+1} column-independent: subtract it
        kap = []
        for m in ms:
            Fc = F.copy(); Fc[m] += 1e-20j
            b, h_ = residual_series(S.rH, Fc, H, S.alpha, L+4)
            kap.append(np.imag(h_[m+2])/np.imag(b[m+1]))
        mats = []
        for m, M, k in zip(ms, rows, kap):
            Mt = M.copy()
            # Nt_{m+1} = Nh[m+2] - kappa_{m+1} Nb[m+1]; dNb[m+1]/d(level m) needed:
            for j, (w, kk) in enumerate((('F', m-1), ('H', m))):
                Fc = F.copy(); Hc = H.copy()
                (Fc if w == 'F' else Hc)[kk] += 1e-20j
                b, h_ = residual_series(S.rH, Fc, Hc, S.alpha, L+4)
                Mt[1, j] = np.imag(h_[m+2]-k*b[m+1])/1e-20
            mats.append(Mt)
        self.ms, self.mats = ms, np.array(mats)
        self.deg = deg
        # normalise by leading growth
        # entries are exactly polynomial in the level: degree 1 (F column), 2 (H column)
        self.fits = [[np.polyfit(ms, self.mats[:, i, j], (1, 2)[j]) for j in range(2)] for i in range(2)]

    def M(self, lam):
        return np.array([[np.polyval(self.fits[i][j], lam) for j in range(2)] for i in range(2)])

    def det_roots(self):
        d = np.polysub(np.polymul(self.fits[0][0], self.fits[1][1]), np.polymul(self.fits[0][1], self.fits[1][0]))
        return np.roots(d)

    def null(self, lam):
        M = self.M(lam)
        # null vector from first row: phi*M00 + psi*M01 = 0 -> (phi, psi) = (-M01, M00)
        v = np.array([-M[0, 1], M[0, 0]])
        return v/v[1]


def initial_data(S, gam, nv, B, x0, gser=None):
    """background series + B * gamma-mode; gser = (phi, psi) full gamma-mode series
    (gamma_mode.gamma_series); without it only the leading (nv) term is used"""
    F, H = S.F, S.H
    k = np.arange(len(F)); kh = np.arange(len(H))
    Fx = np.sum(F*x0**k)
    phi, psi = gser if gser is not None else (np.array([nv[0]]), np.array([1.0]))
    j = np.arange(len(psi))
    f = x0**2*Fx + B*np.sum(phi*x0**(gam+1+j))
    h = np.sum(H*x0**kh) + B*np.sum(psi*x0**(gam+j))
    hp = np.sum(H[1:]*kh[1:]*x0**(kh[1:]-1)) + B*np.sum(psi*(gam+j)*x0**(gam+j-1))
    return np.array([f, h, hp])


def integrate(S, gam, nv, B, x0, R, dense=False, rtol=1e-12):
    y0 = initial_data(S, gam, nv, B, x0)
    sol = solve_ivp(rhs, (S.rH+x0, R), y0, args=(S.alpha,), method='DOP853', rtol=rtol, atol=1e-14,
                    dense_output=dense)
    return sol


if __name__ == '__main__':
    import sys
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    S = HorizonSeries(alpha, 10)
    u0 = np.zeros(1+2*(S.L-1)); u0[0] = 0.0122
    S.solve(u0)
    print('series res', S.res, 'H1', S.H[1])
    I = Indicial(S)
    print('det roots:', np.sort_complex(I.det_roots()), ' throat gamma:', S.T['gamma'])
    gam = S.T['gamma']
    nv = I.null(gam)
    print('null vector (phi, psi):', nv, ' det at gamma:', np.linalg.det(I.M(gam))/np.abs(I.M(gam)).max()**2)
    for B in [0, 1e-3, -1e-3]:
        sol = integrate(S, gam, nv, B, 1e-3, 60)
        r = sol.t; f, h, hp = sol.y
        idx = [np.argmin(abs(r-v)) for v in (2, 5, 10, 20, 40, 60)]
        print(f'B={B}: status {sol.status}', ' '.join(f'r={r[i]:.1f}: r^2(1-f)={r[i]**2*(1-f[i]):.8f} (h/r^2-1)r^4={(h[i]/r[i]**2-1)*r[i]**4:.5f}' for i in idx))
