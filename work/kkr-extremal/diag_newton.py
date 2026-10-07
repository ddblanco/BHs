import numpy as np
from extremal_solver import Extremal, myers_perry
n = 32
ex = Extremal(n)
mp = myers_perry(ex.x)
ex.set_weights(mp, 0.)
for eps in (0., 1e-6, 1e-3):
    x = np.tile(ex.x, 4)
    guess = mp*(1+eps*np.sin(3*x+np.repeat(np.arange(4), n)))
    sol, info = ex.solve(0., guess, verbose=True, max_iter=8)
    print(eps, info, 'dev', np.max(np.abs(sol-mp).reshape(4, n), axis=1))
