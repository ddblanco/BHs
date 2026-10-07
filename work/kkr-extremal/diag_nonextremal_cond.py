"""Condition number of the project's non-extremal collocation Jacobian, for
comparison with the extremal one (same variables, simple-zero horizon)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from rotating_bh import egb_rotating_bvp as bvp

for omega in (0.5, 0.68, 0.70):
    sol = bvp.solve(omega, 0.0, resolution=32, tol=1e-11)
    n = len(sol.nodes)
    x = bvp._cheb(n)[0]
    flat = sol.evaluate(np.sort(x)[::-1]).ravel() if False else sol.evaluate(x).ravel()
    def res(v):
        return bvp._residual(v, n, omega, 0.0, 1-x)
    J = np.zeros((4*n, 4*n))
    for c in range(4*n):
        v = flat.astype(complex); v[c] += 1e-30j
        J[:, c] = res(v).imag/1e-30
    s = np.linalg.svd(J, compute_uv=False)
    print(f'omega_h={omega}: |R|={np.max(np.abs(res(flat))):.1e}  cond={s[0]/s[-1]:.2e}  smallest rel {s[-3:]/s[0]}')
