import numpy as np, mpmath as mp, time
from scipy.integrate import solve_ivp
from from_kkr import load_row, fields, J0
from build_crhs import crhs
from build_qrhs import qintegrate
mp.mp.dps = 40
sv, row = load_row(3.0); hi = row['high']; L = (J0/hi['J'])**(1/3); al = L**2*hi['alpha']
s = np.linspace(1e-3, 0.95, 20000)
R, fR, h, dRds, _ = fields(row, s); R, h = L*R, L**2*h; hp = np.gradient(h, R)
i0 = np.argmin(abs(R-1.3)); i1 = np.argmin(abs(R-3.0))
y0 = [fR[i0], h[i0], hp[i0]]
for tol in (1e-20, 1e-26, 1e-30):
    t = time.time(); code, r, y, n = qintegrate(R[i0], R[i1], al, y0, tol=tol)
    print('quad tol', tol, 'code', code, 'r', mp.nstr(r, 12), [mp.nstr(v, 20) for v in y], 'steps', n, f'{time.time()-t:.2f}s')
sol = solve_ivp(crhs, (R[i0], R[i1]), y0, args=(al,), method='DOP853', rtol=1e-13, atol=1e-15)
print('double', sol.y[:, -1]); print('spectral', fR[i1], h[i1], hp[i1])
