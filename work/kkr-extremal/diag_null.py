"""Smallest singular values of the extremal Jacobian at exact Myers-Perry and
the shape of the near-null vector (which field, where)."""
import numpy as np
from extremal_solver import Extremal, myers_perry

for n in (24, 32, 40):
    ex = Extremal(n)
    mp = myers_perry(ex.x)
    ex.set_weights(mp, 0.)
    A = ex.jacobian(mp, 0.)
    U, s, Vt = np.linalg.svd(A)
    print(n, 'smallest singular values', s[-4:]/s[0])
    v = Vt[-1].reshape(4, n)
    for name, vi in zip('BFHW', v):
        print(f'   null {name}: at horizon {vi[0]:+.3e}, at infinity {vi[-1]:+.3e}, max {np.max(np.abs(vi)):.3e}')
    # residual row with largest left-null weight
    left = U[:, -1].reshape(4, n)
    k = np.unravel_index(np.argmax(np.abs(left)), left.shape)
    print('   left null vector peaks at equation', 'bghw'[k[0]], 'node', k[1], 'x=', ex.x[k[1]])
    x = ex.x
    z = 1-x
    # compare null vector with candidate symmetries: rescaling of t (b -> lam^2 b, w -> lam w)
    print('   null B/B_MP', np.round(v[0]/(mp.reshape(4, n)[0]), 4)[::max(1, n//8)])
    print('   null W/W_MP', np.round(v[3]/(mp.reshape(4, n)[3]), 4)[::max(1, n//8)])
