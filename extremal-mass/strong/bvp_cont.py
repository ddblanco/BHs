"""BVP continuation in alpha, seeded by the spectral solution of work/kkr-extremal (largest s on that branch).
Results appended to bvp_results.json; solutions pickled in bvp_sols/.
Usage: python bvp_cont.py alpha1,alpha2,...   (first run seeds from the spectral branch)"""
import json, pickle, sys, time
import numpy as np
from pathlib import Path
from bvp import BVP
HERE = Path(__file__).resolve().parent
OUT = HERE/'bvp_results_beta.json'
SOLS = HERE/'bvp_sols_beta'; SOLS.mkdir(exist_ok=True)


def seed_from_spectral(target_s=5.24, n=3000):
    from from_kkr import load_row, fields, J0
    sv, row = load_row(target_s)
    hi = row['high']; L = (J0/hi['J'])**(1/3); al = L**2*hi['alpha']
    s = np.linspace(1e-4, 0.9995, 40000)
    R, fR, h, dRds, _ = fields(row, s); R, h = L*R, L**2*h
    hp = np.gradient(h, R)
    b = BVP(al)
    t = np.linspace(b.ta, b.tb, n); r = b.rH+np.exp(t)
    Y = np.vstack([np.interp(r, R, fR), np.interp(r, R, h), np.interp(r, R, hp)])
    # near the horizon use the series (spectral data are least accurate there)
    near = np.exp(t) < 3*b.x0
    for i in np.where(near)[0]:
        Y[:, i] = [float(v) for v in b.S.start(0.0, np.exp(t[i]))[1]]
    mu0 = 4*(hi['M']*L**2*2/np.pi)/3
    b.left_lin()
    beta0 = float(np.interp(b.x1, R-b.rH, h))
    print(f'seed: spectral row s={sv:.5f} -> code alpha={al:.8f}, m={0.75*mu0:.8f}', flush=True)
    return al, b, t, Y, np.array([beta0, mu0, 0.0])


def remap(prev, b, n=3000):
    """previous solution (t, Y, p, rH) onto the new domain, as a function of x = r - r_H.
    The gamma-mode amplitude is re-fitted (least squares on h over x in [x0, 6 x0]) instead of carried over:
    across a resonance gamma = integer it passes through a pole while the profile stays continuous."""
    tp, Yp, pp, rHp = prev
    xp = np.exp(tp)
    t = np.linspace(b.ta, b.tb, n); x = np.exp(t)
    Y = np.vstack([np.interp(np.log(x), np.log(xp), Yp[i]) for i in range(3)])
    U, V = b.left_lin()
    beta = float(np.interp(np.log(b.x1), np.log(xp), Yp[1]))
    near = x < 3*b.x0
    for i in np.where(near)[0]:
        Y[:, i] = b.family(beta, x[i])
    p = pp.copy(); p[0] = beta
    return t, Y, p


def solve_one(b, t, Y, p, tols=(1e-6, 1e-8)):
    out = []
    for tol in tols:
        sol = b.solve(t, Y, p, tol=tol, max_nodes=400000)
        out.append((tol, sol.status, len(sol.x), 0.75*sol.p[1]))
        print(f'   tol {tol}: status {sol.status} nodes {len(sol.x)} p={sol.p} m={0.75*sol.p[1]:.13f}', flush=True)
        if sol.status not in (0,):
            break
        t, Y, p = sol.x, sol.y, sol.p
    return sol, out


if __name__ == '__main__':
    alphas = [float(v) for v in sys.argv[1].split(',')]
    res = json.loads(OUT.read_text()) if OUT.exists() else []
    prev = None
    if res:
        last = max([d for d in res if d['status'] in (0, 1)], key=lambda d: d['alpha'])
        tp, Yp, pp, rHp = pickle.load(open(SOLS/f"{last['alpha']:.8f}.pkl", 'rb'))
        prev = (tp, Yp, np.array(pp), rHp)
    else:
        al, b, t, Y, p = seed_from_spectral()
        t0 = time.time()
        sol, hist = solve_one(b, t, Y, p)
        res.append(dict(alpha=al, s=2*al, m=0.75*sol.p[1], p=list(sol.p), status=sol.status, hist=hist, seconds=time.time()-t0))
        OUT.write_text(json.dumps(res, indent=1))
        pickle.dump((sol.x, sol.y, list(sol.p), b.rH), open(SOLS/f'{al:.8f}.pkl', 'wb'))
        prev = (sol.x, sol.y, sol.p, b.rH)
    for al in alphas:
        if any(abs(d['alpha']-al) < 1e-9 and d['status'] in (0, 1) for d in res):
            continue
        import os
        from mp_series import cached
        cached(al)      # computes (5 min) or loads the 60-digit horizon data
        t0 = time.time()
        b = BVP(al)
        t, Y, p = remap(prev, b)
        print(f'alpha={al} s={2*al} gamma={b.gam:.4f} x0={b.x0:.3e} R={b.R:.1f}', flush=True)
        sol, hist = solve_one(b, t, Y, p)
        rec = dict(alpha=al, s=2*al, m=0.75*sol.p[1], p=list(sol.p), status=sol.status, hist=hist, seconds=time.time()-t0)
        res.append(rec); OUT.write_text(json.dumps(res, indent=1))
        if sol.status in (0, 1) and np.all(np.isfinite(sol.p)) and 0 < sol.p[1] < 10*al+10:
            pickle.dump((sol.x, sol.y, list(sol.p), b.rH), open(SOLS/f'{al:.8f}.pkl', 'wb'))
            prev = (sol.x, sol.y, sol.p, b.rH)
        print(f'RESULT alpha={al} s={2*al} m={rec["m"]:.13f} status={sol.status} [{rec["seconds"]:.0f}s]', flush=True)
