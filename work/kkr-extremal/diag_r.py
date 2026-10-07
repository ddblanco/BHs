"""Structure of the r-coordinate (power 0) system at extremal Myers-Perry:
which horizon rows vanish identically, and the null space after Neumann
replacement."""
import numpy as np
from kkr_solver import KKR, myers_perry

for n in (32, 48):
    k = KKR(n, 0)
    mp = myers_perry(n)
    k.weights = None
    E = k.raw(mp.reshape(4, n), 0.)
    print(f'N={n}: max |residual| at MP {np.abs(E[:, :-1]).max():.1e}')
    # raw Jacobian without any row replacement
    u = mp.reshape(4, n)
    Jac = np.zeros((4, n, 4, n))
    from kkr_solver import JET_MAP
    for (field, order), dE in zip(JET_MAP, k.jet_derivatives(u, 0.)):
        for e in range(4):
            Jac[e, :, field, :] += dE[e][:, None]*k.mats[order]
    print('   horizon row norms', np.linalg.norm(Jac[:, 0].reshape(4, -1), axis=1))
    print('   infinity row norms', np.linalg.norm(Jac[:, -1].reshape(4, -1), axis=1))
    # Neumann for P1, P2, P3 at s = 0, Dirichlet at infinity
    A = Jac.copy()
    for i in range(3):
        A[i, 0] = 0; A[i, 0, i, :] = k.D[0]
    A[:, -1] = 0
    for i in range(4):
        A[i, -1, i, -1] = 1
    A = A.reshape(4*n, 4*n)
    A = A/np.abs(A).max(axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(A)
    print('   smallest singular values', s[-5:]/s[0])
    for m in (1, 2):
        v = Vt[-m].reshape(4, n)
        left = U[:, -m].reshape(4, n)
        i = np.unravel_index(np.argmax(np.abs(left)), left.shape)
        print(f'   null {m}: left peaks at eq {i[0]} node s={k.x[i[1]]:.3f}; right at horizon {np.round(v[:, 0], 3)}, '
              f'max per field {np.round(np.abs(v).max(axis=1), 3)}')

print('with the two zero modes fixed: rows (W, node 1) -> P2(0)=1, (P2, node 1) -> P2_s(1)=0')
for n in (32, 48, 64, 96):
    k = KKR(n, 0)
    mp = myers_perry(n)
    u = mp.reshape(4, n)
    Jac = np.zeros((4, n, 4, n))
    from kkr_solver import JET_MAP
    for (field, order), dE in zip(JET_MAP, k.jet_derivatives(u, 0.)):
        for e in range(4):
            Jac[e, :, field, :] += dE[e][:, None]*k.mats[order]
    A = Jac.copy()
    for i in range(3):
        A[i, 0] = 0; A[i, 0, i, :] = k.D[0]
    A[:, -1] = 0
    for i in range(4):
        A[i, -1, i, -1] = 1
    A[3, 1] = 0; A[3, 1, 1, 0] = 1
    A[1, 1] = 0; A[1, 1, 1, :] = k.D[-1]
    A = A.reshape(4*n, 4*n)
    A = A/np.abs(A).max(axis=1, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    print(f'   N={n}: cond {s[0]/s[-1]:.2e}, smallest {s[-3:]/s[0]}')
