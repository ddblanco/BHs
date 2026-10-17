import numpy as np
from horizon_series import residual_series
from throat import throat
for alpha in [0.3, 1.0, 5.0]:
    T = throat(alpha); rH = np.sqrt(T['g_H'])
    n = 6
    F = np.r_[1/T['v'], np.zeros(n)]; H = np.r_[T['h_H'], np.zeros(n+1)]
    nb, nh = residual_series(rH, F.astype(complex), H.astype(complex), alpha, n+5)
    print(f'alpha={alpha}: Nb0={abs(nb[0]):.2e} Nb1={abs(nb[1]):.2e} Nh1={abs(nh[1]):.2e} Nh2={abs(nh[2]):.2e}  (Nb2 scale {abs(nb[2]):.2e}, Nh3 scale {abs(nh[3]):.2e})')
    # level-2 Jacobian wrt (F1,H1,H2) of (Nb2, Nh3)
    J = []
    for which, idx in [('F', 1), ('H', 1), ('H', 2)]:
        Fc = F.astype(complex); Hc = H.astype(complex)
        (Fc if which == 'F' else Hc)[idx] += 1e-20j
        b, h_ = residual_series(rH, Fc, Hc, alpha, n+5)
        J.append([np.imag(b[2])/1e-20, np.imag(h_[3])/1e-20])
    J = np.array(J).T
    print('   level-2 Jacobian rows (Nb2, Nh3) wrt (F1,H1,H2):\n', J, '\n   singular values', np.linalg.svd(J/np.abs(J).max(axis=1)[:, None])[1])
