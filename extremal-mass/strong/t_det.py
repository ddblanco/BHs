import numpy as np, sys
from scipy.integrate import solve_ivp
from extremal_shoot import build
from shoot import rhs, initial_data
from ode_poly import coeffs
alpha = float(sys.argv[1])
S, I, gerr = build(alpha, 12)
gam = S.T['gamma']; nv = I.null(gam)
print('rH', S.rH, 'gamma', gam, 'H1', S.H[1])
# evaluate det and numerators along the series itself (B=0) for small x
for x in [1e-3, 3e-3, 1e-2, 3e-2, 0.05, 0.1]:
    y = initial_data(S, gam, nv, 0., x)
    Ab, Bb, Cb, Ah, Bh, Ch = coeffs(np.array(S.rH+x), np.array(y[0]), np.array(y[1]), np.array(y[2]), alpha)
    det = Ab*Bh-Ah*Bb; n1 = -Cb*Bh+Ch*Bb; n2 = -Ab*Ch+Ah*Cb
    fp_series = 2*x*S.F[0] + 3*x**2*S.F[1] + 4*x**3*S.F[2]
    print(f'x={x:.0e}: det={det:.3e} f\'(ode)={n1/det:.6e} f\'(series)={fp_series:.6e}  h\'\'(ode)={n2/det:.6e} h\'\'(series)={2*S.H[2]+6*S.H[3]*x+12*S.H[4]*x**2:.6e}')
y0 = initial_data(S, gam, nv, 0., 0.02*S.rH)
sol = solve_ivp(rhs, (S.rH+0.02*S.rH, S.rH+1.), y0, args=(alpha,), method='DOP853', rtol=1e-10, atol=1e-13, dense_output=True)
print('status', sol.status, sol.message, 'last r', sol.t[-1])
for r in np.linspace(sol.t[0], sol.t[-1], 12):
    f, h, hp = sol.sol(r)
    Ab, Bb, Cb, Ah, Bh, Ch = coeffs(np.array(r), np.array(f), np.array(h), np.array(hp), alpha)
    print(f'r={r:.5f} f={f:.6e} h={h:.6f} hp={hp:.6f} det={Ab*Bh-Ah*Bb:.3e} Bb={Bb:.3e} Bh={Bh:.3e}')
