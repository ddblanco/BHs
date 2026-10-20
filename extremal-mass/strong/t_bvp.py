import json, time, numpy as np, pickle, sys
from bvp import guess_from_shooting
a = float(sys.argv[1])
r = {round(d['alpha'], 6): d for d in json.load(open('sweep.json'))}[a]
t0 = time.time()
b, t, Y, p = guess_from_shooting(a, r['B'], r['mu'], r['b4'], n=3000)
pickle.dump((t, Y, p), open(f'bvp_guess_{a}.pkl', 'wb'))
F = b.fun(t, Y, p); dY = np.gradient(Y, t, axis=1)
res = np.abs(dY-F)/(np.abs(F)+np.abs(Y)+1e-30)
print('guess residual max', res.max(), 'at r', b.rH+np.exp(t[res.max(axis=0).argmax()]), 'max|F|', np.abs(F).max(), f'{time.time()-t0:.0f}s', flush=True)
sol = b.solve(t, Y, p, tol=1e-6, verbose=0)
print('tol 1e-6:', sol.status, sol.message, len(sol.x), sol.p, 'm', 0.75*sol.p[1], 'shooting m', r['m'], flush=True)
for tol in (1e-8, 1e-10):
    sol = b.solve(sol.x, sol.y, sol.p, tol=tol)
    print(f'tol {tol}:', sol.status, len(sol.x), sol.p, 'm %.12f' % (0.75*sol.p[1]), flush=True)
pickle.dump((sol.x, sol.y, sol.p), open(f'bvp_sol_{a}.pkl', 'wb'))
