import numpy as np
from horizon_series import residual_series
from throat import throat
alpha = 1.0
T = throat(alpha); rH = np.sqrt(T['g_H'])
n = 7
rng = np.random.default_rng(1)
F = np.r_[1/T['v'], 0.3*rng.standard_normal(n)]; H = np.r_[T['h_H'], 0.3*rng.standard_normal(n+1)]
K = n+5
def res(Fc, Hc):
    nb, nh = residual_series(rH, Fc, Hc, alpha, K)
    return nb, nh
nb0, nh0 = res(F.astype(complex), H.astype(complex))
print('scale nb', np.abs(nb0[:6]), '\nscale nh', np.abs(nh0[:6]))
for name, arr, idx in [(f'F{k}', 'F', k) for k in range(5)]+[(f'H{k}', 'H', k) for k in range(6)]:
    Fc = F.astype(complex); Hc = H.astype(complex)
    (Fc if arr == 'F' else Hc)[idx] += 1e-20j
    nb, nh = res(Fc, Hc)
    db = np.imag(nb)/1e-20; dh = np.imag(nh)/1e-20
    print(name, 'Nb:', ' '.join(f'{v:9.2e}' for v in db[:8]), '| Nh:', ' '.join(f'{v:9.2e}' for v in dh[:8]))
