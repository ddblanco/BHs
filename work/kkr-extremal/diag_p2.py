"""Null space of the p = 2 Jacobian at Myers-Perry, and the effect of
replacing the horizon rows by Neumann conditions."""
import numpy as np
from kkr_solver import KKR, myers_perry

for n in (24, 40):
    k = KKR(n, 2)
    mp = myers_perry(n)
    k.weights = None
    A = k.jacobian(mp, 0.)
    s = np.linalg.svd(A, compute_uv=False)
    print(f'N={n}: singular values / max below 1e-12: {np.sum(s/s[0] < 1e-12)}')
    rows0 = A.reshape(4, n, 4*n)[:, 0, :]
    print('   horizon rows norms:', np.linalg.norm(rows0, axis=1))
    # Neumann at s = 0 in place of the equation rows there
    B = A.reshape(4, n, 4, n).copy()
    B[:, 0] = 0
    for i in range(4):
        B[i, 0, i, :] = k.D[0]
    B = B.reshape(4*n, 4*n)
    s = np.linalg.svd(B, compute_uv=False)
    print(f'   with Neumann rows: cond {s[0]/s[-1]:.2e}, smallest {s[-3:]/s[0]}')

print('variant: Neumann for P1, P2, P3 at s = 0, keep the E_W row')
for n in (24, 40, 64):
    k = KKR(n, 2)
    mp = myers_perry(n)
    k.weights = None
    A = k.jacobian(mp, 0.).reshape(4, n, 4, n).copy()
    for i in range(3):
        A[i, 0] = 0
        A[i, 0, i, :] = k.D[0]
    A = A.reshape(4*n, 4*n)
    # row equilibration
    A = A/np.abs(A).max(axis=1, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    print(f'   N={n}: cond {s[0]/s[-1]:.2e}, smallest {s[-3:]/s[0]}')
