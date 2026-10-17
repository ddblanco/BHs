import numpy as np, sys, time
from extremal_shoot import Shooter
alpha = float(sys.argv[1])
sh = Shooter(alpha, 12, R=12.)
sh.x0 = 0.003*sh.S.rH
print('gamma', sh.gam, 'nv', sh.nv, 'x0', sh.x0, flush=True)
for B in [float(v) for v in sys.argv[2].split(',')]:
    t = time.time()
    sol = sh.run(B)
    r = sol.t[-1]; f, h, hp = sol.y[:, -1]
    print(f'B={B:8.3f}: status {sol.status} r_end={r:.4f} f={f:.6e} h/r^2={h/r**2:.6e} r^2(1-f)={r**2*(1-f):.6f} T={sh.T(B) if sol.status==0 else 0:+.3e} events {[e.size for e in sol.t_events]} nfev {sol.nfev} {time.time()-t:.1f}s', flush=True)
