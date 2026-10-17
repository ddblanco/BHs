import numpy as np, sympy as sp
from tps import TPS
import _ode_generated as G
TPS.K = 10
x = sp.Symbol('x')
def ser(expr):
    sr = sp.series(expr, x, 0, TPS.K+1).removeO()
    return TPS([float(sr.coeff(x, k)) for k in range(TPS.K+1)])
r = 1+x
fs = ser((1-1/r**2)**2); hs = ser(r**2+1/r**2)
R = TPS([1., 1.])
hp, hpp, fp = hs.deriv(), hs.deriv().deriv(), fs.deriv()
for nm in ('eb', 'eh'):
    A, B, C = (getattr(G, f'{l}_{nm}')(R, fs, hs, hp, 0.0) for l in 'ABC')
    res = A*fp+B*hpp+C
    print(nm, np.max(np.abs(res.c[:TPS.K-2])), 'A', np.round(A.c[:4], 4), 'B', np.round(B.c[:4], 4), 'C', np.round(C.c[:4],4))
