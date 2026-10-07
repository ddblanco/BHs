"""Is the near-null vector of the p = 1 KKR Jacobian the residual radial gauge
mode r -> r + c sqrt(b f) of the gauge b f = fixed?  At alpha = 0 (MP):
dP1 = -eta B2'/B2, dP2 = -eta u'/u, dP3 = -eta B3'/B3, dW = -eta w0',
eta = sqrt(B1 B2) = r^3/sqrt(u (u^2+1)), derivatives in r."""
import numpy as np
from kkr_solver import KKR, myers_perry

for n in (24, 40, 64):
    k = KKR(n, 1)
    mp = myers_perry(n)
    k.set_weights(mp, 0.)
    A = k.jacobian(mp, 0.)
    U, s, Vt = np.linalg.svd(A)
    v = Vt[-1]
    xi = k.x[:-1]                          # drop infinity (r finite)
    r2 = xi/(1-xi); r = np.sqrt(r2); u = r2+1
    eta = r**3/np.sqrt(u*(u*u+1))
    B2p_over_B2 = 4/r-4*r*u/(u*u+1)        # d/dr log(r^4/(u^2+1))
    up_over_u = 2*r/u
    B3 = 1+1/u**2
    B3p_over_B3 = (-4*r/u**3)/B3
    w0p = -np.sqrt(2)*4*r*u/(u*u+1)**2
    g = np.zeros((4, n))
    g[0, :-1] = -eta*B2p_over_B2
    g[1, :-1] = -eta*up_over_u
    g[2, :-1] = -eta*B3p_over_B3
    g[3, :-1] = -eta*w0p
    # the gauge mode tends to a nonzero constant shift effect at infinity; the
    # Dirichlet rows there pin the nodal value, so compare on interior nodes
    mask = np.zeros((4, n), bool); mask[:, 1:-1] = True
    gv, vv = g[mask], v.reshape(4, n)[mask]
    cos = abs(gv@vv)/np.linalg.norm(gv)/np.linalg.norm(vv)
    print(f'N={n}: sigma_min/sigma_max {s[-1]/s[0]:.1e}; |cos(null, gauge mode)| on interior nodes = {cos:.4f}')
