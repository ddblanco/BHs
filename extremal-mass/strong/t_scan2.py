import numpy as np, sys, time
from extremal_shoot import Shooter
alpha = float(sys.argv[1]); x0f = float(sys.argv[2])
sh = Shooter(alpha, 12, x0=x0f*np.sqrt(1.0), R=12.)
sh.x0 = x0f*sh.S.rH
print('gamma', sh.gam, 'nv', sh.nv, 'x0', sh.x0, flush=True)
for B in [-1e6, -1e5, -1e4, -1e3, -1e2, 1e2, 1e3, 1e4, 1e5, 1e6]:
    t = time.time()
    sol = sh.run(B)
    r = sol.t[-1]; f, h, hp = sol.y[:, -1]
    print(f'B={B:6}: status {sol.status} r_end={r:.4f} f={f:.4e} h/r^2={h/r**2:.4e} events {[e.size for e in sol.t_events]} nfev {sol.nfev} {time.time()-t:.1f}s', flush=True)
