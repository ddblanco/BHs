"""Does the smooth zero mode persist at finite alpha (fixed a = 1)?"""
import numpy as np
from kkr_solver import KKR, myers_perry
xs = np.array([0, .01, .1, .3, .5, .7, .9, .99, 1])
k = KKR(40, 1)
flat = myers_perry(40)
for a in np.geomspace(1e-3, .04, 10):
    flat, info = k.solve(a, flat)
for n in (40, 64, 96):
    flat = k.interpolate(flat, n)
    k = KKR(n, 1)
    flat, info = k.solve(.04, flat)
    k.set_weights(flat, .04)
    U, s, Vt = np.linalg.svd(k.jacobian(flat, .04))
    v = Vt[-1].reshape(4, n)
    t = 1-2*k.x
    vals = np.array([np.polynomial.chebyshev.chebval(1-2*xs, np.polynomial.chebyshev.chebfit(t, vi, n-1)) for vi in v])
    vals /= vals[1, 4]
    print(f'alpha=0.04 N={n}: sigma_min/max {s[-1]/s[0]:.1e}, next {s[-2]/s[0]:.1e}')
    for name, row in zip(('P1', 'P2', 'P3', 'W'), vals):
        print(f'  {name}', ' '.join(f'{c:+.4f}' for c in row))
