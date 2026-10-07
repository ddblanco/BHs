"""Resolution test of the g_H = 1 formulation: does the constraint E_f vanish
as N grows, and do M, J, mu converge?"""
import numpy as np
from kkr_solver import KKR, myers_perry, invariants
for alpha_target in (0.05, 0.2, 0.5):
    k = KKR(32, 1); flat = myers_perry(32)
    for a in np.linspace(0, alpha_target, 12)[1:]:
        flat, info = k.solve(a, flat)
    for n in (32, 48, 64, 96, 128, 160):
        flat = k.interpolate(flat, n); k = KKR(n, 1)
        flat, info = k.solve(alpha_target, flat)
        o = k.observables(flat, alpha_target); inv = invariants(o)
        print(f"alpha={alpha_target} N={n:3d} res {info['residual']:.0e} cond {info['condition']:.1e} "
              f"M {o['M']:.12f} J {o['J']:.12f} y {inv['y']:.10f} mu {inv['mu']:.10f} C {o['constraint']:.1e} EW0 {o['E_W_horizon']:.1e}")
