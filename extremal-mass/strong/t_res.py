import numpy as np, sys
from extremal_shoot import build
from horizon_series import residual_series
alpha = float(sys.argv[1])
S, I, gerr = build(alpha, 12)
b, h_ = residual_series(S.rH, S.F.astype(complex), S.H.astype(complex), alpha, 16)
# scale: magnitude of individual contributions ~ use derivative columns
for k in range(15):
    print(k, f'Nb={abs(b[k]):.3e}  Nh={abs(h_[k]):.3e}')
print('kappa', {m: round(v, 3) for m, v in S.kappa.items()})
print('F', S.F); print('H', S.H)
