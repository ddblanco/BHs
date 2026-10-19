"""Sweep alpha with the two-sided solver; resumable, results appended to sweep.json.
Robust start: if the continuation guess fails, bisect the outward escape classification for B*,
then Newton for (B, mu, b4).  Error estimate at each point: rerun with x0/2, r_m*1.4, R*1.5."""
import json, sys, time
import numpy as np
from pathlib import Path
from twoside import TwoSide
OUT = Path(__file__).resolve().parent/'sweep.json'
RES = [0.757001, 4.830610, 13.967932, 31.657915, 62.114554, 110.381663, 182.340493, 289.0, 440.0]


def robust_guess(T, Bs):
    sh = T.sh
    sh.R = 4*T.rm
    signs = [(B, np.sign(sh.T(B)[0])) for B in Bs]
    for (b1, s1), (b2, s2) in zip(signs, signs[1:]):
        if s1 != s2:
            sh.solve(b1, b2)
            sol = sh.run(sh.B, R=sh.best.t[-1], dense=True)
            r = min(sol.t[-1], 3*T.rm)
            f = sol.sol(r)[0]
            return [sh.B, r**2*(1-f), 0.0]
    raise RuntimeError('no B bracket in scan')


def run_point(alpha, guess=None, check=True):
    t = time.time()
    T = TwoSide(alpha)
    p = None
    if guess is not None:
        try:
            p = T.solve(guess)
            if T.err > 1e-6:
                p = None
        except Exception:
            p = None
    if p is None:
        g = T.guess()
        p = T.solve(g)
    rec = dict(alpha=alpha, s=2*alpha, m=0.75*p[1], mu=p[1], B=p[0], b4=p[2], gamma=T.sh.gam,
               err=T.err, cond=T.cond, rm=T.rm, R=T.R, x0=T.sh.x0)
    if check:
        dm = []
        for kw in (dict(x0f=0.0015), dict(rm=1.4*T.rm), dict(R=1.5*T.R)):
            T2 = TwoSide(alpha, **kw)
            try:
                p2 = T2.solve(p)
                if T2.err > 1e-6:
                    raise RuntimeError
            except Exception:
                p2 = T2.solve(T2.guess())
            dm.append(0.75*(p2[1]-p[1]))
        rec['dm_variants'] = dm
    rec['seconds'] = time.time()-t
    return rec


if __name__ == '__main__':
    alphas = [float(v) for v in sys.argv[1].split(',')]
    data = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {round(d['alpha'], 12) for d in data}
    guess = None
    for idx, al in enumerate(alphas):
        if round(al, 12) in done:
            prev = [d for d in data if round(d['alpha'], 12) == round(al, 12)][0]
            guess = [prev['B'], prev['mu'], prev['b4']]
            continue
        if any(abs(al/r-1) < 0.03 for r in RES):
            print('skip (near gamma resonance)', al, flush=True); continue
        import signal
        def _alarm(*a): raise TimeoutError('point exceeded time limit')
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(900)
        try:
            rec = run_point(al, guess, check=(idx % 4 == 0))
            signal.alarm(0)
        except Exception as ex:
            signal.alarm(0)
            print('FAILED', al, repr(ex), flush=True); guess = None; continue
        data.append(rec); OUT.write_text(json.dumps(data, indent=1))
        guess = [rec['B'], rec['mu'], rec['b4']]
        print(f"alpha={al:.6g} s={2*al:.6g} m={rec['m']:.12f} B={rec['B']:.6g} b4={rec['b4']:.4g} gamma={rec['gamma']:.4f} err={rec['err']:.1e} dm={['%.1e' % v for v in rec.get('dm_variants', [])]} [{rec['seconds']:.0f}s]", flush=True)
