import numpy as np
from horizon_series import HorizonSeries, residual_series
from throat import throat
alpha = 0.3
S = HorizonSeries(alpha, 10)
T = throat(alpha); rH = np.sqrt(T['g_H'])
def probe(H1, K):
    F = np.zeros(6, complex); H = np.zeros(7, complex)
    F[0] = 1/T['v']; H[0] = T['h_H']; H[1] = H1
    b, h_ = residual_series(rH, F, H, alpha, K)
    Fc = F.copy(); Fc[1] += 1e-20j
    bb, hh = residual_series(rH, Fc, H, alpha, K)
    kap = np.imag(hh[3])/np.imag(bb[2])
    return np.real(h_[3]-kap*b[2]), kap
print('kappa class', S.kappa[2])
for H1 in [-1.7401336, 2.48269087, 4.168969, 0.0]:
    print(H1, 'class Nt2', S.Nt2(H1), ' probe K=5', probe(H1, 5), ' probe K=6', probe(H1, 6))
