"""Continuation in alpha with multiple shooting (ms.MS).  Seed: a converged profile (pickle (t, Y, p, rH)).
Results -> ms_results.json, profiles -> ms_sols/.  On failure a midpoint alpha is inserted (depth <= 3).
Usage: python ms_cont.py seed.pkl alpha_seed alpha1,alpha2,..."""
import json, pickle, sys, time
import numpy as np
from pathlib import Path
from ms import MS
HERE = Path(__file__).resolve().parent
OUT = HERE/'ms_results.json'
SOLS = HERE/'ms_sols'; SOLS.mkdir(exist_ok=True)


HIST = []      # last converged (alpha, (t, Y, beta, mu, b4))


def attempt(al, prev):
    t, Y, mu, b4 = prev
    t0 = time.time()
    M = MS(al)
    try:
        if len(HIST) >= 2:
            (a0, P0), (a1, P1) = HIST[-2], HIST[-1]
            u0 = M.guess_secant(P0, P1, a0, a1)
        else:
            u0 = M.guess_from_profile(t, Y, mu, b4, alpha_prev=prev_alpha[0])
        u = M.solve(u0, verbose=False)
        ok = M.res < 1e-8 and 0 < u[-2] < 10*al+20
        rec = dict(alpha=al, s=2*al, m=0.75*u[-2], mu=u[-2], b4=u[-1], beta=u[0], res=M.res, K=M.K,
                   seconds=time.time()-t0, ok=bool(ok))
        prof = M.profile() if ok else None
    finally:
        if M.pool is not None:
            M.pool.close(); M.pool.join()
    return rec, prof


def step(al, prev, res, depth=0):
    from mp_series import cached
    cached(al)
    rec, prof = attempt(al, prev)
    print(f"{'  '*depth}alpha={al:.6f} s={2*al:.5f} m={rec['m']:.13f} res={rec['res']:.1e} K={rec['K']} ok={rec['ok']} [{rec['seconds']:.0f}s]", flush=True)
    if rec['ok']:
        res.append(rec); OUT.write_text(json.dumps(res, indent=1))
        t, Y, beta, mu, b4 = prof
        HIST.append((al, prof))
        pickle.dump((t, Y, [beta, mu, b4], None), open(SOLS/f'{al:.8f}.pkl', 'wb'))
        return (t, Y, mu, b4)
    if depth >= 3:
        raise RuntimeError(f'continuation failed at alpha={al}')
    a0 = prev_alpha[0]
    mid = round(float(np.sqrt(a0*al)), 6)
    p2 = step(mid, prev, res, depth+1)
    prev_alpha[0] = mid
    return step(al, p2, res, depth+1)


prev_alpha = [None]

if __name__ == '__main__':
    seed, a_seed = sys.argv[1], float(sys.argv[2])
    alphas = [float(v) for v in sys.argv[3].split(',')]
    res = json.loads(OUT.read_text()) if OUT.exists() else []
    done = sorted(res, key=lambda d: d['alpha'])
    if done:
        last = done[-1]
        t, Y, p, _ = pickle.load(open(SOLS/f"{last['alpha']:.8f}.pkl", 'rb'))
        prev = (t, Y, p[1], p[2]); prev_alpha[0] = last['alpha']
        for d in done[-2:]:
            tt, YY, pp, _ = pickle.load(open(SOLS/f"{d['alpha']:.8f}.pkl", 'rb'))
            HIST.append((d['alpha'], (tt, YY, pp[0], pp[1], pp[2])))
    else:
        t, Y, p, _ = pickle.load(open(seed, 'rb'))
        prev = (t, Y, p[1], p[2]); prev_alpha[0] = a_seed
    for al in alphas:
        if al <= prev_alpha[0]+1e-12:
            continue
        prev = step(al, prev, res)
        prev_alpha[0] = al
