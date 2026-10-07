import json, numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ
from branch_rq import record, acceptable
CH = np.polynomial.chebyshev
d = json.loads(open('branch_rq.json').read())
last = d['rows'][-1]
for n, key in ((64, 'coeffs_low'), (96, 'coeffs_high')):
    e = ExtremalRQ(n)
    f = np.array([CH.chebval(1-2*e.x, c) for c in last[key]]).ravel()
    sol, info = e.solve(0.0068, f)
    r = record(e, sol, 0.0068, info)
    print(n, info, 'constraint', r['constraint'], 'tails', r['tails'], 'acceptable', acceptable(r))
