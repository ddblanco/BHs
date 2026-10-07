import json, numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ
from branch_rq import robust_solve, record
CH = np.polynomial.chebyshev
d = json.loads(open('branch_rq.json').read())
last = d['rows'][-1]
print('last alpha', last['alpha'], 'N', last['n_low'], last['n_high'], 'step', last['step'])
for frac in (1.0, 0.5, 0.25):
    a = last['alpha']+frac*min(last['step']*1.4, 0.05*last['alpha'])
    e = ExtremalRQ(last['n_low'])
    f = np.array([CH.chebval(1-2*e.x, c) for c in last['coeffs_low']]).ravel()
    sol, info = e.solve(a, f)
    r = record(e, sol, a, info)
    print(f"alpha {a:.5f}: res {r['residual']:.1e} it {r['iterations']} C {r['constraint']:.1e} tails {max(r['tails']):.0e} mu {r['mu']:.9f}")

print('--- with the tangent predictor q + da dq/dalpha')
from extremal_rq import H_STEP
e = ExtremalRQ(last['n_low'])
f0 = np.array([CH.chebval(1-2*e.x, c) for c in last['coeffs_low']]).ravel()
a0 = last['alpha']
e.set_weights(f0, a0)
A = e.jacobian(f0, a0)
dR = e.residual(f0.astype(complex), a0+1j*H_STEP).imag/H_STEP
dq = np.linalg.solve(A, -dR)
for da in (0.0156, 0.0312, 0.0625):
    sol, info = e.solve(a0+da, f0+da*dq)
    r = record(e, sol, a0+da, info)
    print(f"alpha {a0+da:.5f}: res {r['residual']:.1e} it {r['iterations']} C {r['constraint']:.1e} tails {max(r['tails']):.0e} mu {r['mu']:.9f}")
