import numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ, myers_perry_q, invariants
for n in (24, 40, 64):
    e = ExtremalRQ(n)
    q0 = myers_perry_q(n)
    p = q0+1e-2*np.sin(np.arange(4*n))
    e.weights = None
    A, B = e.jacobian(p, 0.), e.dense_jacobian(p, 0.)
    err = np.max(np.abs(A-B)/np.maximum(np.abs(B).max(axis=1, keepdims=True), 1e-300))
    e.set_weights(q0, 0.)
    U, s, Vt = np.linalg.svd(e.jacobian(q0, 0.))
    left = U[:, -1].reshape(4, n); i = np.unravel_index(np.argmax(np.abs(left)), left.shape)
    print(f'N={n}: Jacobian check {err:.1e}; cond {s[0]/s[-1]:.1e}; smallest {s[-3:]/s[0]}; left null peak eq {i[0]} node {i[1]}')
