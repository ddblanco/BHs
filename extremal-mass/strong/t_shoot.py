import time, numpy as np, sys
from horizon_series import HorizonSeries
from shoot import Indicial, integrate
alpha = float(sys.argv[1])
t = time.time()
S = HorizonSeries(alpha, 10)
u0 = np.zeros(1+2*(S.L-1)); u0[0] = 0.0122
S.solve(u0)
print('series res', S.res, 'H1', S.H[1], f'{time.time()-t:.1f}s', flush=True)
t = time.time()
I = Indicial(S)
print('det roots:', np.sort_complex(I.det_roots()), ' throat gamma:', S.T['gamma'], f'{time.time()-t:.1f}s', flush=True)
gam = S.T['gamma']; nv = I.null(gam)
print('null vector', nv, flush=True)
for x0 in [1e-2]:
    for B in [0.]:
        t = time.time()
        sol = integrate(S, gam, nv, B, x0, 20.0)
        print('integrate', sol.status, sol.message, len(sol.t), f'{time.time()-t:.1f}s', flush=True)
        r = sol.t; f, h, hp = sol.y
        for v in (1.5, 2, 5, 10, 20):
            i = np.argmin(abs(r-v))
            print(f'  r={r[i]:.2f} r^2(1-f)={r[i]**2*(1-f[i]):.8f} (h/r^2-1) r^4={(h[i]/r[i]**2-1)*r[i]**4:.6f}', flush=True)
