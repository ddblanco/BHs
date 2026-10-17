"""Do the exact d_n follow the large-order form set by the throat branch point s*?
   d_n ~ 2 Re[ C s*^-n n^beta ]  (complex C).  For each beta, fit C on (n-1, n) and predict d_(n+1)."""
import json, mpmath as mp
from pathlib import Path
mp.mp.dps = 30
d = {int(k): mp.mpf(v['value']) for k, v in json.loads((Path(__file__).parent/'coefficients.json').read_text()).items()}
d[1] = mp.mpf(1)
ss = mp.mpc('0.0059231771', '0.0548188529')
def fitC(n1, n2, beta):
    # unknowns Re C, Im C:  d_n = 2 (ReC Re w_n - ImC Im w_n), w_n = s*^-n n^beta
    w1 = ss**-n1*mp.mpf(n1)**beta; w2 = ss**-n2*mp.mpf(n2)**beta
    A = mp.matrix([[2*mp.re(w1), -2*mp.im(w1)], [2*mp.re(w2), -2*mp.im(w2)]])
    x = mp.lu_solve(A, mp.matrix([d[n1], d[n2]]))
    return mp.mpc(x[0], x[1])
N = max(d)
for beta in [-1.5, -2.5, -3.5]:
    C = fitC(N-2, N-1, beta)
    pred = 2*mp.re(C*ss**-N*mp.mpf(N)**beta)
    print(f'beta={beta}: C={mp.nstr(C, 6)}  predicted d_{N} = {mp.nstr(pred, 8)}  actual {mp.nstr(d[N], 8)}')
# best beta from 3-term fit on the last three
f = lambda b: 2*mp.re(fitC(N-2, N-1, b)*ss**-N*mp.mpf(N)**b)-d[N]
for b0 in [-1, -2, -3, -4]:
    try:
        print('beta solving the last three exactly:', mp.nstr(mp.findroot(f, b0), 8))
    except Exception as e:
        pass
