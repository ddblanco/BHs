"""debug: re-solve one alpha with Rfac=12 from the secant of two good v2 profiles, verbose Newton"""
import sys, pickle
from ms import MS
a0, a1, al = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
kw = eval(sys.argv[4]) if len(sys.argv) > 4 else {}
nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 14
L = lambda a: pickle.load(open(f'ms_sols_v2/{a:.8f}.pkl', 'rb'))
P = []
for a in (a0, a1):
    t, Y, p, _ = L(a); P.append((t, Y, p[0], p[1], p[2]))
kw.setdefault('Rfac', 12)
M = MS(al, nproc=nproc, **kw)
try:
    u0 = M.guess_secant(P[0], P[1], a0, a1)
    print('guess mu, b4', u0[-2], u0[-1], flush=True)
    M.stag = 0.9
    u = M.solve(u0, maxit=20, verbose=True)
    print('RESULT', al, 0.75*u[-2], u[-1], M.res, flush=True)
    import numpy as np
    r, F = M.residual(u); r3 = np.abs(r).reshape(-1, 3).max(1); x = np.exp(M.t)
    for i in np.argsort(-r3)[:8]:
        print(f'  seg {i} r={M.b.rH+x[i]:.4g}..{M.b.rH+x[i+1]:.4g} res={r3[i]:.2e}', flush=True)
    print('K', M.K, 'R', M.b.R, 'rho', M.b.rho, flush=True)
finally:
    M.pool.close(); M.pool.join()
