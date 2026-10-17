import numpy as np
from horizon_series import residual_series, lapse_series, _solve_linear
from throat import throat
alpha = 1.0
T = throat(alpha); rH = np.sqrt(T['g_H'])
n = 8
F = np.zeros(n+1, complex); H = np.zeros(n+2, complex)
F[0] = 1/T['v']; H[0] = T['h_H']
def lap(v):
    Hc = H.copy(); Hc[1] = v[0]
    un, ud = lapse_series(rH, F, Hc, alpha, 4)
    return np.array([un[1]-2*ud[2]])
(H1,), _ = _solve_linear(lap, [0.], newton=3); H[1] = H1
print('H1 =', H1)
nb, nh = residual_series(rH, F, H, alpha, 8)
print('after H1: Nb', np.abs(nb[:5]), '\n Nh', np.abs(nh[:6]))
# Jacobian of (Nb[0..5], Nh[0..6]) wrt (F1..F4, H2..H5)
names = [('F', k) for k in range(1, 5)]+[('H', k) for k in range(2, 6)]
rows = [('b', k) for k in range(0, 6)]+[('h', k) for k in range(0, 7)]
J = np.zeros((len(rows), len(names)))
for j, (w, k) in enumerate(names):
    Fc = F.copy(); Hc = H.copy()
    (Fc if w == 'F' else Hc)[k] += 1e-20j
    b, h_ = residual_series(rH, Fc, Hc, alpha, 9)
    for i, (rw, kk) in enumerate(rows):
        J[i, j] = np.imag((b if rw == 'b' else h_)[kk])/1e-20
np.set_printoptions(linewidth=200, precision=3)
print('cols:', names)
for i, rw in enumerate(rows):
    print(rw, J[i]/max(1, np.abs(J[i]).max()), ' rowmax %.1e' % np.abs(J[i]).max())
