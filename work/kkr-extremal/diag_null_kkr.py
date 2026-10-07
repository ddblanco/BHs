import numpy as np
from kkr_solver import KKR, myers_perry
xs = np.array([0, 1e-3, 1e-2, .05, .1, .3, .5, .7, .9, .97, .99, .999, 1])
for n in (40, 64, 96):
    k = KKR(n, 1)
    mp = myers_perry(n)
    k.set_weights(mp, 0.)
    U, s, Vt = np.linalg.svd(k.jacobian(mp, 0.))
    v = Vt[-1].reshape(4, n)
    t = 1-2*k.x
    vals = np.array([np.polynomial.chebyshev.chebval(1-2*xs, np.polynomial.chebyshev.chebfit(t, vi, n-1)) for vi in v])
    vals /= vals[1, 6]
    print(f'N={n} sigma {s[-1]/s[0]:.1e}')
    for name, row in zip(('P1', 'P2', 'P3', 'W'), vals):
        print(f'  {name}', ' '.join(f'{c:+.3f}' for c in row))
    left = U[:, -1].reshape(4, n)
    i = np.unravel_index(np.argmax(np.abs(left)), left.shape)
    print('  left null vector peaks at equation', i[0], 'node xi =', k.x[i[1]])
