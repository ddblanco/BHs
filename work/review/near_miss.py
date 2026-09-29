"""For states whose re-solution misses the 1e-8 relative tensor gate: measure them anyway,
and refine at N=128 and N=160 (seeded by the same solution), to see whether the
observables that enter the T->0 extrapolation depend on the gate outcome."""
import glob, json
from multiprocessing import Pool
import numpy as np
from rotating_bh.egb_rotating_predictor import solve
from rotating_bh.egb_rotating_scale import scaled_diagnose
from rotating_bh.egb_rotating_observables import measure
from rotating_bh.gb_response import predictor_ladder, response_observables
from reproduce_walks import advance

DATA = json.load(open('../../results/egb-extremality.json'))


def job(key):
    d = json.load(open(f'out2/walk-{key}.json'))
    fails = sorted(r['omega_h'] for r in d['rows'] if 'failed' in r)
    if not fails:
        return []
    stored = {round(s['omega_h'], 12): s for s in DATA['walks'][key]}
    alpha = float(key)
    cur = predictor_ladder(.6, alpha, step=.005, resolution=64)
    out = []
    for om in fails:
        s = stored[round(om, 12)]
        n = s['resolution']
        if cur.initial_resolution != n:
            cur = solve(float(cur.omega_h), alpha, resolution=n, tol=1e-11, previous=cur)
        cur = advance(cur, om, alpha, n)
        row = dict(alpha=alpha, omega_h=om, T_H=s['T_H'], stored_rel=s['max_relative_tensor_residual'])
        for N in (n, 128, 160):
            sol = cur if N == n else solve(om, alpha, resolution=N, tol=1e-11, previous=cur)
            m = measure(sol); p = response_observables(sol)['psi_gb']
            rel = scaled_diagnose(sol, alpha, count=51)['max_relative_tensor_residual']
            row[f'N{N}'] = dict(rel=rel, dE=abs(m['E']-s['E'])/s['E'], dT=abs(m['T_H']-s['T_H'])/s['T_H'],
                                dPsi=abs(p-s['psi_gb']))
        out.append(row)
    return out


if __name__ == '__main__':
    keys = [p.split('walk-')[1][:-5] for p in glob.glob('out2/walk-*[0-9].json')]
    rows = []
    with Pool(len(keys)) as pool:
        for part in pool.imap_unordered(job, keys):
            rows += part
    rows.sort(key=lambda r: (r['alpha'], r['omega_h']))
    json.dump(rows, open('out/near_miss.json', 'w'), indent=1)
    print(f"{'alpha':>6} {'Omega':>9} {'T_H':>8} {'stored':>8} | {'N':>4} {'rel':>8} {'dE/E':>8} {'dT/T':>8} {'dPsi':>8}")
    for r in rows:
        for k in [k for k in r if k.startswith('N')]:
            v = r[k]
            print(f"{r['alpha']:6g} {r['omega_h']:9.6f} {r['T_H']:8.1e} {r['stored_rel']:8.1e} | {k[1:]:>4} {v['rel']:8.1e} {v['dE']:8.1e} {v['dT']:8.1e} {v['dPsi']:8.1e}")
