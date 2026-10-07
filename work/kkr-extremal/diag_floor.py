import numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ, myers_perry_q
n = 48
e = ExtremalRQ(n)
flat = myers_perry_q(n)
for a in (2.5e-4, 5e-4, 1e-3):
    flat, info = e.solve(a, flat)
R = e.residual(flat, 1e-3).reshape(4, n)
print('final', info)
for i in range(4):
    k = np.argsort(np.abs(R[i]))[-3:]
    print(f'eq {i}: largest weighted residuals at nodes {k} (s={np.round(e.x[k], 4)}): {R[i, k]}')
print('weights at those nodes', [e.weights[i, np.argmax(np.abs(R[i]))] for i in range(4)])
