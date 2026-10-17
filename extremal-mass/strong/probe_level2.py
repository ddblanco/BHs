import numpy as np
from horizon_series import residual_series, lapse_series, _solve_linear
from throat import throat
for alpha in [0.3, 1.0, 5.0]:
    T = throat(alpha); rH = np.sqrt(T['g_H'])
    F = np.zeros(6, complex); H = np.zeros(7, complex)
    F[0] = 1/T['v']; H[0] = T['h_H']
    def Ntil2(H1):
        Hc = H.copy(); Hc[1] = H1
        b, h_ = residual_series(rH, F, Hc, alpha, 6)
        Fc = F.copy(); Fc[1] += 1e-20j
        bb, hh = residual_series(rH, Fc, Hc, alpha, 6)
        kap = np.imag(hh[3])/np.imag(bb[2])
        Hc2 = Hc.copy(); Hc2[2] += 1e-20j
        bb2, hh2 = residual_series(rH, F, Hc2, alpha, 6)
        kap2 = np.imag(hh2[3])/np.imag(bb2[2])
        return np.real(h_[3]-kap*b[2]), kap, kap2, np.real(h_[3]), np.real(b[2])
    def lap(H1):
        Hc = H.copy(); Hc[1] = H1
        un, ud = lapse_series(rH, F, Hc, alpha, 4)
        return np.real(un[1]-2*ud[2])
    from scipy.optimize import brentq
    # scan
    xs = np.linspace(-5, 5, 2001)
    vals = [Ntil2(v)[0] for v in xs]
    roots = [brentq(lambda v: Ntil2(v)[0], xs[i], xs[i+1]) for i in range(len(xs)-1) if np.sign(vals[i]) != np.sign(vals[i+1])]
    lv = [lap(v) for v in xs]
    lroots = [brentq(lap, xs[i], xs[i+1]) for i in range(len(xs)-1) if np.sign(lv[i]) != np.sign(lv[i+1])]
    print(f'alpha={alpha}: roots of Ntilde2(H1): {np.round(roots, 10)}; lapse-regular H1: {np.round(lroots, 10)}; kappa (F-col, H-col) = {Ntil2(1.0)[1]:.6e}, {Ntil2(1.0)[2]:.6e}')
