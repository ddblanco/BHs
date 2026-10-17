"""Series of the gamma-mode about the background horizon series:
    dh = B x^gamma sum_k psi_k x^k,   df = B x^(gamma+1) sum_k phi_k x^k,   psi_0 = 1.
Linearized residual by complex step; the perturbation is carried as x^e0 * (power series), e0 = gamma-3,
with d/dx (x^e0 P) = x^e0 (e0 P/x + P').  Pairing as for the background: at order k the unknowns
(phi_k, psi_k) solve [Nb at TPS order 3+k, Nh - kappa Nb at TPS order 4+k] = 0."""
import numpy as np
from tps import TPS
from ode_poly import coeffs_tps


def _D(P, e0):
    """x^e0 P -> derivative / x^e0 ; requires P[0] = 0"""
    c = P.c
    assert abs(c[0]) == 0 or abs(c[0]) < 1e-300
    q = np.zeros_like(c)
    q[:-1] = c[1:]                      # P/x
    return TPS(e0*q)+P.deriv()


def linear_residual(S, gam, phi, psi, K):
    """returns (Nb_lin, Nh_lin) as coefficient arrays of x^(e0+n)"""
    e0 = gam-3
    TPS.K = K
    eps = 1e-20
    x = TPS(np.array([0., 1.], complex))
    R = x+S.rH
    f = TPS(np.asarray(S.F, complex)).shift(2)
    h = TPS(np.asarray(S.H, complex))
    hp = h.deriv(); hpp = hp.deriv(); fp = f.deriv()
    Fh = TPS(np.asarray(phi, complex)).shift(4)      # x^4 Phi
    Hh = TPS(np.asarray(psi, complex)).shift(3)      # x^3 Psi
    dHp = _D(Hh, e0); dHpp = _D(dHp, e0); dFp = _D(Fh, e0)
    fc = f+1j*eps*Fh; hc = h+1j*eps*Hh; hpc = hp+1j*eps*dHp; hppc = hpp+1j*eps*dHpp; fpc = fp+1j*eps*dFp
    Ab, Bb, Cb, Ah, Bh, Ch = coeffs_tps(R, fc, hc, hpc, S.alpha)
    Nb = (Ab*fpc+Bb*hppc+Cb).c.imag/eps
    Nh = (Ah*fpc+Bh*hppc+Ch).c.imag/eps
    return Nb, Nh


def gamma_series(S, gam, nv, n=8):
    kap = S.kappa[2]
    phi = np.zeros(n+1); psi = np.zeros(n+1)
    phi[0], psi[0] = nv[0], 1.0
    K = n+8
    Nb, Nh = linear_residual(S, gam, phi, psi, K)
    lead = (abs(Nb[3]), abs(Nh[5]-kap*Nb[4]))
    for k in range(1, n+1):
        def ev(v):
            ph = phi.copy(); ps = psi.copy(); ph[k], ps[k] = v
            Nb, Nh = linear_residual(S, gam, ph, ps, K)
            return np.array([Nb[3+k], Nh[5+k]-kap*Nb[4+k]])
        e = ev((0., 0.))
        J = np.array([ev((1., 0.))-e, ev((0., 1.))-e]).T     # linear in (phi_k, psi_k)
        phi[k], psi[k] = np.linalg.solve(J, -e)
    Nb, Nh = linear_residual(S, gam, phi, psi, K)
    scale = np.max(np.abs(Nb[:n+4]))+1e-300
    chk = dict(lead=lead, Nb=np.abs(Nb[:n+4]).max(), Nt=np.abs(Nh[1:n+6]-kap*Nb[:n+5]).max(),
               Nb_terms=float(scale))
    return phi, psi, chk


if __name__ == '__main__':
    import sys
    from extremal_shoot import build
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.491010442529934
    S, I, gerr = build(alpha, 12)
    gam = S.T['gamma']; nv = I.null(gam)
    phi, psi, chk = gamma_series(S, gam, nv, 8)
    print('gamma', gam, 'nv', nv)
    print('phi', phi); print('psi', psi); print(chk)
