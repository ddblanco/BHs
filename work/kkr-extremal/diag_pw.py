import numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_r_solver import ExtremalR, myers_perry, invariants
from _pw_generated_p0 import J_of
e = ExtremalR(48); flat = myers_perry(48)
for a in (1e-3, 1e-2, 3e-2):
    flat, info = e.solve(a, flat)
for n in (48, 64, 96, 128, 160):
    flat = e.interpolate(flat, n); e = ExtremalR(n)
    flat, info = e.solve(.03, flat)
    o = e.observables(flat, .03)
    u = flat.reshape(4, n); D = e.D
    Jn = J_of(e.x[1:-1], u[0,1:-1], (D@u[0])[1:-1], u[1,1:-1], (D@u[1])[1:-1], u[2,1:-1], (D@u[2])[1:-1], u[3,1:-1], (D@u[3])[1:-1], .03)
    k = [n//8, n//4, n//2, 3*n//4, 7*n//8]
    print(f"N={n}: res {info['residual']:.0e} M {o['M']:.10f} J(med) {o['J']:.10f} J_inf {o['J_inf']:.10f} spread {o['J_spread']:.1e} "
          f"J at s={np.round(e.x[k],3)}: {np.round(Jn[np.array(k)-1], 8)}")
