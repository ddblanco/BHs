import numpy as np
from tps import TPS
import _ode_generated as G
TPS.K = 12
def resid(rH, F, H, alpha):
    x = TPS([0., 1.])
    R = x+rH
    f = TPS(F).shift(2); h = TPS(H)
    out = []
    for nm in ('eb', 'eh'):
        A, B, C = (getattr(G, f'{l}_{nm}')(R, f, h, h.deriv(), alpha) for l in 'ABC')
        out.append((A*f.deriv()+B*h.deriv().deriv()+C).c)
    return out
# MP point
import sympy as sp
xs = sp.Symbol('x'); r = 1+xs
Fmp = [float(c) for c in [sp.series((1-1/r**2)**2/xs**2, xs, 0, 13).removeO().coeff(xs, k) for k in range(13)]]
Hmp = [float(c) for c in [sp.series(r**2+1/r**2, xs, 0, 13).removeO().coeff(xs, k) for k in range(13)]]
nb, nh = resid(1., Fmp, Hmp, 0.)
print('MP residual', np.abs(nb[:9]).max(), np.abs(nh[:9]).max())
# Jacobian of first n orders wrt unknowns
eps = 1e-7
unk = ['rH']+[f'F{k}' for k in range(8)]+[f'H{k}' for k in range(8)]
base = [1.]+Fmp[:8]+Hmp[:8]
def vec(u, alpha=0.):
    F = list(u[1:9])+Fmp[8:]; H = list(u[9:17])+Hmp[8:]
    nb, nh = resid(u[0], F, H, alpha)
    return nb, nh
nb0, nh0 = vec(base)
for i, name in enumerate(unk):
    u = list(base); u[i] += eps
    nb, nh = vec(u)
    db = (nb-nb0)/eps; dh = (nh-nh0)/eps
    fb = next((k for k in range(10) if abs(db[k]) > 1e-6), None)
    fh = next((k for k in range(10) if abs(dh[k]) > 1e-6), None)
    print(f'{name}: first order in Nb {fb} ({db[fb] if fb is not None else 0:.3g}), Nh {fh} ({dh[fh] if fh is not None else 0:.3g})')
