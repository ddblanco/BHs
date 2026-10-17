import numpy as np, time
from horizon_series import HorizonSeries
from shoot import Indicial
for alpha in [0.01, 0.3, 1.5, 5.0, 50.]:
    S = HorizonSeries(alpha, 8)
    roots = S.H1_roots(-12, 12, 481)
    for r in roots:
        S.solve_sequential(r)
        I = Indicial(S, mmax=8)
        print(f"alpha={alpha}: H1={S.H[1]:.8f} gamma_throat={S.T['gamma']:.6f} det roots {np.round(np.sort(I.det_roots().real), 6)}", flush=True)
