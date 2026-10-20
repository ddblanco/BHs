import sys, numpy as np, mpmath as mp
from scipy.integrate import solve_ivp
from mp_series import cached
from build_crhs import crhs
from extremal_shoot import ev_h, ev_big, ev_det
alpha = float(sys.argv[1]); lo, hi = float(sys.argv[2]), float(sys.argv[3]); xf = 0.03
S = cached(alpha); rho = S.radius(); x0 = xf*rho; sc = float(rho)**(-float(S.gamma))
rend = 6*(1.5*float(S.rH)+0.5*np.sqrt(alpha))
def run(Bh):
    r0, y0 = S.start(Bh*sc, x0)
    sol = solve_ivp(crhs, (float(r0), rend), [float(v) for v in y0], args=(alpha,), method='DOP853',
                    rtol=1e-13, atol=1e-15, events=[ev_h, ev_big, ev_det])
    r = sol.t[-1]; f, h, hp = sol.y[:, -1]
    return r, h/r**2, r**2*(1-f)
rlo = run(lo); side_lo = np.sign(rlo[1]-1)
for it in range(60):
    mid = 0.5*(lo+hi); r, hr, mu = run(mid)
    if np.sign(hr-1) == side_lo: lo = mid
    else: hi = mid
    if it % 4 == 0 or it > 50:
        print(f'it {it}: Bhat={mid:.17g} r_end={r:.4f} h/r^2={hr:.5f} r^2(1-f)={mu:.6f}', flush=True)
print('B* =', mid*sc, 'Bhat =', mid)
