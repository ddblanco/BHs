import pickle, numpy as np, time, sys
from bvp import BVP
from scipy.integrate import solve_bvp
a0, a1, mode = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
t, Y, p, rH = pickle.load(open('bvp_sols_beta/%.8f.pkl' % a0, 'rb'))
b0 = BVP(a0); b = BVP(a1); b.left_lin(); b0.left_lin()
xp = np.exp(t)
n = 4000
tn = np.linspace(b.ta, b.tb, n); x = np.exp(tn)
if mode == 'x':
    xq = x
else:   # blend: near horizon x scales with rho, far away r is the natural variable
    w = np.clip(np.log(x/b.rho)/np.log(30), 0, 1)
    xq = (1-w)*x*b0.rho/b.rho + w*(x+b.rH-b0.rH)
Yn = np.vstack([np.interp(np.log(np.maximum(xq, xp[0])), t, Y[i]) for i in range(3)])
beta = float(np.interp(np.log(b.x1*(b0.rho/b.rho if mode != 'x' else 1)), t, Y[1]))
for i in np.where(x < 3*b.x0)[0]:
    Yn[:, i] = b.family(beta, x[i])
pn = np.array([beta, p[1], p[2]])
t0 = time.time()
sol = b.solve(tn, Yn, pn, tol=1e-6, max_nodes=300000, verbose=1)
print(mode, sol.status, sol.message, len(sol.x), sol.p, 0.75*sol.p[1], f'{time.time()-t0:.0f}s')
rr = sol.rms_residuals
idx = np.argsort(rr)[-8:]
for i in sorted(idx):
    xm = np.exp(0.5*(sol.x[i]+sol.x[i+1]))
    print(f'interval {i}: x={xm:.4e} r={b.rH+xm:.5f} rms residual {rr[i]:.2e}  Y={sol.y[:, i]}')
import collections
xs = np.exp(sol.x)
print('node density: x<1e-2:', np.sum(xs < 1e-2), ' 1e-2..0.1:', np.sum((xs >= 1e-2) & (xs < 0.1)), ' 0.1..1:', np.sum((xs >= 0.1) & (xs < 1)), ' >1:', np.sum(xs >= 1))
