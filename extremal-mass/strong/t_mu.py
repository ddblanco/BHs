import numpy as np, sys
from extremal_shoot import Shooter
alpha = float(sys.argv[1]); Blo, Bhi = float(sys.argv[2]), float(sys.argv[3])
sh = Shooter(alpha, 12, R=400.); sh.x0 = 0.003*sh.S.rH
sh.solve(Blo, Bhi)
sol = sh.run(sh.B, R=sh.best.t[-1], dense=True)
for r in [2, 5, 10, 20, 40, 80, 150, 250, 400]:
    if r > sol.t[-1]: break
    f, h, hp = sol.sol(r)
    eta = np.log(h/r**2)
    print(f'r={r}: r^2(1-f)={r**2*(1-f):.12f}  eta r^4={eta*r**4:.8f}  (r eta\'+4eta)={r*(hp/h-2/r)+4*eta:.3e}')
