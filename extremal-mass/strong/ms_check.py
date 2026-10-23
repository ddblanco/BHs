"""Accuracy check of multiple-shooting points: re-solve a converged alpha with changed numerical
parameters (finer nodes / tighter rtol, different left point xf) starting from its saved profile.
Usage: python ms_check.py alpha [nproc] [variant,variant,...].  Appends to ms_check.json."""
import json, pickle, sys, time
from pathlib import Path
from ms import MS
HERE = Path(__file__).resolve().parent
OUT = HERE/'ms_check.json'

VARIANTS = {'fine': dict(dphi=4.0, nh=18, nfar=40, rtol=1e-13),
            'xf2': dict(xf=0.002),
            'xf45': dict(xf=0.0045),
            'rtol13': dict(rtol=1e-13), 'dphi4': dict(dphi=4.0), 'nhfar': dict(nh=18, nfar=40), 'nh18': dict(nh=18), 'nfar40': dict(nfar=40), 'nh24': dict(nh=24), 'nfar60': dict(nfar=60), 'nfar80': dict(nfar=80), 'R12': dict(R=12), 'R8': dict(R=8), 'R12n40': dict(R=12, nfar=40),
            'dphi3': dict(dphi=3.0), 'fine2': dict(dphi=3.0, nh=18, nfar=40, rtol=1e-14)}

if __name__ == '__main__':
    al = float(sys.argv[1]); nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    base = next(d for d in json.loads((HERE/'ms_results.json').read_text()) if abs(d['alpha']-al) < 1e-7)
    t, Y, p, _ = pickle.load(open(HERE/'ms_sols'/f'{base["alpha"]:.8f}.pkl', 'rb'))
    res = json.loads(OUT.read_text()) if OUT.exists() else []
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else ['fine', 'xf2', 'xf45']
    for name in names:
        kw = VARIANTS[name]
        t0 = time.time()
        kw = dict(kw)
        if 'R' in kw:
            kw['R'] = max(30., kw['R']*base['alpha']**0.5)
        M = MS(base['alpha'], nproc=nproc, **kw)
        try:
            u = M.solve(M.guess_from_profile(t, Y, p[1], p[2]), verbose=False)
        finally:
            M.pool.close(); M.pool.join()
        rec = dict(alpha=base['alpha'], s=base['s'], variant=name, m=0.75*u[-2], m_base=base['m'],
                   dm=0.75*u[-2]-base['m'], res=M.res, K=M.K, seconds=time.time()-t0)
        print(rec, flush=True)
        res.append(rec); OUT.write_text(json.dumps(res, indent=1))
