import numpy as np, math
from from_kkr import load_row, fields, J0
from extremal_shoot import build
sv, row = load_row(3.0)
hi = row['high']; L = (J0/hi['J'])**(1/3); al = L**2*hi['alpha']
S, I, gerr = build(al, 12)
gam = S.T['gamma']; nv = I.null(gam)
print('alpha', al, 'series rH', S.rH, 'gamma', gam, 'H1', S.H[1], 'nv', nv)
s = np.linspace(2e-4, 0.03, 300)
R, fR, h, dRds, _ = fields(row, s); R, h = L*R, L**2*h
x = R-S.rH
k = np.arange(len(S.H)); kf = np.arange(len(S.F))
hs = np.array([np.sum(S.H*xx**k) for xx in x])
fs = np.array([xx**2*np.sum(S.F*xx**kf) for xx in x])
dh = h-hs; df = fR-fs
for i in range(0, 300, 30):
    print(f'x={x[i]:.3e}: h-hser={dh[i]:+.3e}  (h-hser)/x^gamma={dh[i]/x[i]**gam:+.4e}  (f-fser)/x^(gamma+1)={df[i]/x[i]**(gam+1):+.4e} ratio phi={df[i]/x[i]**(gam+1)/(dh[i]/x[i]**gam):+.4f}')
