import numpy as np
from extremal_solver import Extremal, myers_perry
xs = np.linspace(0, 1, 11)
for n in (24, 32, 40, 48):
    ex = Extremal(n)
    mp = myers_perry(ex.x)
    ex.set_weights(mp, 0.)
    A = ex.jacobian(mp, 0.)
    U, s, Vt = np.linalg.svd(A)
    v = Vt[-1].reshape(4, n)
    v = v/v[2, -1]
    t = 1-2*ex.x
    print(n, 'sigma_min', s[-1]/s[0])
    for name, vi in zip('BFHW', v):
        c = np.polynomial.chebyshev.chebfit(t, vi, n-1)
        print(f'  {name}', np.round(np.polynomial.chebyshev.chebval(1-2*xs, c), 4))
