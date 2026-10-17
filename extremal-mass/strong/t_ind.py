import numpy as np
from horizon_series import HorizonSeries
from shoot import Indicial
for alpha, H1g in [(0.5, 3.1), (1.5, 1.53), (5.0, 0.727)]:
    S = HorizonSeries(alpha, 14)
    S.solve_sequential(min(S.H1_roots(-8, 8, 321), key=lambda v: abs(v-H1g)))
    I = Indicial(S, mmax=14, deg=6)
    ms, M = I.ms, I.mats
    for i in range(2):
        for j in range(2):
            y = M[:, i, j]
            # finite differences to find polynomial degree
            d = y.copy(); degs = []
            for k in range(7):
                degs.append(np.max(np.abs(d))/max(1e-300, np.max(np.abs(y))))
                d = np.diff(d)
            print(alpha, (i, j), 'rel size of k-th differences:', ' '.join(f'{v:.1e}' for v in degs))
    print('throat gamma', S.T['gamma'])
