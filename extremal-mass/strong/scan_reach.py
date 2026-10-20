"""Diagnostic: reach of outward trajectories vs normalised gamma-mode amplitude Bhat = B rho^gamma.
60-digit start data (mp_series.cached), double DOP853 integration with escape events."""
import sys, numpy as np, mpmath as mp
from scipy.integrate import solve_ivp
from mp_series import cached
from build_crhs import crhs
from extremal_shoot import ev_h, ev_big, ev_det
alpha = float(sys.argv[1]); xf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
S = cached(alpha)
rho = S.radius(); x0 = xf*rho; sc = float(rho)**(-float(S.gamma))
rend = 6*(1.5*float(S.rH)+0.5*np.sqrt(alpha))
def run(B):
    r0, y0 = S.start(B, x0)
    sol = solve_ivp(crhs, (float(r0), rend), [float(v) for v in y0], args=(alpha,), method='DOP853',
                    rtol=1e-12, atol=1e-15, events=[ev_h, ev_big, ev_det])
    r = sol.t[-1]; f, h, hp = sol.y[:, -1]
    return r, sol.status, h/r**2, r**2*(1-f)
print('gamma', mp.nstr(S.gamma, 8), 'rho', mp.nstr(rho, 5), 'x0', mp.nstr(x0, 4), 'rend', rend)
for Bh in [float(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else np.r_[-np.logspace(2, -3, 11), np.logspace(-3, 2, 11)]:
    r, st, hr, mu = run(Bh*sc)
    print(f'Bhat={Bh:+.4e}: r_end={r:8.3f} status={st} h/r^2={hr:.4f} r^2(1-f)={mu:.4f}', flush=True)
