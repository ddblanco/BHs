import time, numpy as np, sys
from scipy.integrate import solve_ivp
from horizon_series import HorizonSeries
from shoot import Indicial, rhs, initial_data
alpha = float(sys.argv[1]); R = float(sys.argv[2]) if len(sys.argv) > 2 else 8.
H1g = float(sys.argv[3]); x0 = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-2
t = time.time()
S = HorizonSeries(alpha, 10)
roots = S.H1_roots(-8, 8, 321)
S.solve_sequential(min(roots, key=lambda v: abs(v-H1g)))
print('H1', S.H[1], 'F', S.F[:5], 'H', S.H[:5], 'level dets', {k: round(v, 4) for k, v in S.dets.items()}, f'{time.time()-t:.1f}s', flush=True)
I = Indicial(S); gam = S.T['gamma']; nv = I.null(gam)
print('gamma', gam, 'det-root check', min(abs(I.det_roots()-gam)), 'nv', nv, flush=True)


def ev_h(r, y, a): return y[1]-0.02*r**2
ev_h.terminal = True
def ev_eta(r, y, a): return abs(np.log(abs(y[1])/r**2))-3.0
ev_eta.terminal = True


def T(B, method='LSODA'):
    y0 = initial_data(S, gam, nv, B, x0)
    sol = solve_ivp(rhs, (S.rH+x0, R), y0, args=(alpha,), method=method, rtol=1e-11, atol=1e-13, events=[ev_h, ev_eta])
    r = sol.t[-1]; f, h, hp = sol.y[:, -1]
    eta = np.log(h/r**2); etap = hp/h-2/r
    return r*etap+4*eta, r**2*(1-f), sol


for method in ['DOP853', 'LSODA', 'Radau']:
    t = time.time(); v, mu, sol = T(0., method=method)
    print(f'{method}: T(0)={v:.6e} reached r={sol.t[-1]:.3f} steps {len(sol.t)} nfev {sol.nfev} {time.time()-t:.1f}s', flush=True)
for B in [-10, -1, -0.1, -0.01, 0.01, 0.1, 1, 10]:
    t = time.time()
    v, mu, sol = T(B)
    print(f'B={B:6}: T={v:.6e} r_end={sol.t[-1]:.3f} mu(R)={mu:.6f} steps {len(sol.t)} {time.time()-t:.1f}s', flush=True)
