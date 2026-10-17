import numpy as np
from horizon_series import HorizonSeries
from shoot import Indicial
for alpha, H1g in [(0.3, 4.17), (1.5, 1.53), (5.0, 0.727), (50., 0.2014)]:
    S = HorizonSeries(alpha, 8)
    S.solve_sequential(min(S.H1_roots(-8, 8, 321), key=lambda v: abs(v-H1g)))
    I = Indicial(S, mmax=8)
    print(alpha, 'det roots', np.sort(I.det_roots()), 'throat gamma', S.T['gamma'], '-1-gamma', -1-S.T['gamma'], 'null', I.null(S.T['gamma']))
