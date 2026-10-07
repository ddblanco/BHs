import numpy as np
from kkr_solver import KKR, myers_perry, invariants
for n in (24, 40, 64):
    k = KKR(n, 1)
    mp = myers_perry(n)
    k.set_weights(mp, 0.)
    s = np.linalg.svd(k.jacobian(mp, 0.), compute_uv=False)
    print(f'alpha=0 N={n}: cond {s[0]/s[-1]:.2e} smallest {s[-2:]/s[0]}')
k = KKR(40, 1); flat = myers_perry(40)
for a in np.geomspace(1e-3, .5, 25):
    flat, info = k.solve(a, flat)
    o = k.observables(flat, a); inv = invariants(o)
    print(f"alpha={a:.4f} res {info['residual']:.1e} cond {info['condition']:.1e} y {inv['y']:.5f} mu {inv['mu']:.9f} "
          f"x {inv['x']:.4f} j {inv['j']:.5f} C {o['constraint']:.1e} E_W(0) {o['E_W_horizon']:.1e}")
