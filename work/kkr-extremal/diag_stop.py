import json, numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ
from branch_rq import robust_solve
CH = np.polynomial.chebyshev
d = json.loads(open('branch_rq.json').read())
last = d['rows'][-1]
a = last['alpha']*1.01
for n, key in ((48, 'coeffs_low'), (72, 'coeffs_high')):
    e = ExtremalRQ(n)
    f = np.array([CH.chebval(1-2*e.x, c) for c in last[key]]).ravel()
    sol, rec = robust_solve(e, a, f)
    print(n, 'res %.1e C %.1e tails %s mu %.10f' % (rec['residual'], rec['constraint'], ['%.0e' % t for t in rec['tails']], rec['mu']))
