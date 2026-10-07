import numpy as np
from extremal_r_solver import ExtremalR, myers_perry, invariants
n = 48
e = ExtremalR(n)
x = np.tile(e.x, 4)
for amp in (1e-6, 1e-4, 1e-3, 1e-2):
    guess = myers_perry(n)+amp*np.sin(2*x+np.repeat(np.arange(4), n))*x*(1-x)
    sol, info = e.solve(0., guess)
    print(f'amp {amp:.0e}: {info}, deviation {np.max(np.abs(sol-myers_perry(n))):.1e}')
flat = myers_perry(n)
for a in (1e-3, 1e-2, 3e-2, .1):
    flat, info = e.solve(a, flat, verbose=False)
    o = e.observables(flat, a); inv = invariants(o)
    print(f"alpha={a}: {info} y {inv['y']:.6f} mu {inv['mu']:.10f} J {o['J']:.10f} J_inf {o['J_inf']:.10f} "
          f"spread {o['J_spread']:.0e} P1'(1) {o['P1_slope_inf']:.0e} C {o['constraint']:.0e} rows {o['replaced_rows']}")
