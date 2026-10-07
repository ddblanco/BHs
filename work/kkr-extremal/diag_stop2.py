import json, numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ, _constraint
from branch_rq import robust_solve
CH = np.polynomial.chebyshev
d = json.loads(open('branch_rq.json').read())
last = d['rows'][-1]
a = 0.527
e = ExtremalRQ(48)
f = np.array([CH.chebval(1-2*e.x, c) for c in last['coeffs_low']]).ravel()
f, rec = robust_solve(e, a, f)
for n in (48, 64, 80, 96, 128):
    f = e.interpolate(f, n); e = ExtremalRQ(n)
    f, rec = robust_solve(e, a, f)
    C = np.abs(np.array(_constraint(e.x, e.zeta, *e.qjets(f.reshape(4, n)), a)))
    k = np.argmax(C[1:-1])+1
    print(f"N={n}: res {rec['residual']:.1e} C {rec['constraint']:.1e} at s={e.x[k]:.4f} tails {max(rec['tails']):.0e} "
          f"mu {rec['mu']:.10f} y {rec['y']:.9f} J {rec['J']:.10f} M {rec['M']:.10f}")
