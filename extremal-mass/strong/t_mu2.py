import numpy as np, sys
from extremal_shoot import Shooter
alpha = float(sys.argv[1]); Blo, Bhi = float(sys.argv[2]), float(sys.argv[3])
for rtol, x0f in [(1e-12, 0.003), (1e-13, 0.003), (1e-13, 0.0015)]:
    sh = Shooter(alpha, 12, R=60., rtol=rtol); sh.x0 = x0f*sh.S.rH
    sh.solve(Blo, Bhi)
    sol = sh.run(sh.B, R=sh.best.t[-1], dense=True)
    print(f'rtol={rtol} x0f={x0f} B*={sh.B:.15g}')
    for lo, hi in [(3, 8), (4, 10), (5, 12), (6, 15)]:
        r = np.linspace(lo, hi, 400); f = sol.sol(r)[0]; y = r**2*(1-f)
        res = []
        for K in (3, 4, 5, 6):
            A = np.vstack([r**(-2*k) for k in range(K)]).T
            res.append(np.linalg.lstsq(A, y, rcond=None)[0][0])
        print(f'   window [{lo},{hi}] mu(K=3..6) =', ' '.join(f'{v:.9f}' for v in res))
print('KKR spectral value mu = 4 m/3 =', 4*3.04100567/3)
