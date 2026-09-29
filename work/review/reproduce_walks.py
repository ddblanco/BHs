"""Re-solve every accepted state of results/egb-extremality.json from scratch.

For each of the 13 couplings: climb in alpha at Omega_H=0.6 from Myers-Perry,
then continue in Omega_H (predictor seeds, adaptive sub-steps) to every stored
Omega_H, at the stored resolution, and recompute E, J, S, T_H and Psi with the
project's measurement code. Stores the Chebyshev coefficients of each profile
so they can be checked by the independent evaluator (check_profiles.py).
Run:  PYTHONPATH=../../src ../../.venv/bin/python reproduce_walks.py
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.fft import dct

from rotating_bh.egb_rotating_bvp import _cheb
from rotating_bh.egb_rotating_predictor import solve
from rotating_bh.extremality import state_at
from rotating_bh.egb_rotating_scale import scaled_diagnose
from rotating_bh.gb_response import predictor_ladder

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('out2')
KEYS = ['E', 'J', 'S', 'T_H', 'psi_gb']


def coefficients(sol):
    n = sol.initial_resolution
    x, _, _ = _cheb(n)
    vals = np.asarray(sol.evaluate(x))
    c = dct(vals, type=1, axis=1)/(n-1)
    c[:, [0, -1]] *= .5
    return c


CAP = 2e-3   # a first version without a cap and without per-step gating jumped onto
             # spurious discrete solutions near extremality (out/reproduce-naive-path.log)


def advance(current, target, alpha, n):
    """Go from current solution to Omega_H=target in gated steps <= CAP, halving on failure.

    Every intermediate solution must pass the project's relative tensor residual at 1e-7
    (21 points); otherwise the step is halved. This keeps the continuation on the
    continuum branch rather than on a spurious solution of the collocation system.
    """
    start = float(current.omega_h)
    pos, step = start, np.sign(target-start)*min(abs(target-start), CAP)
    while abs(target-pos) > 1e-15:
        nxt = pos+step if abs(step) < abs(target-pos) else target
        try:
            sol = solve(nxt, alpha, resolution=n, tol=1e-11, previous=current,
                        seed='predictor', jacobian='analytic')
            if scaled_diagnose(sol, alpha, count=21)['max_relative_tensor_residual'] > 1e-7:
                raise RuntimeError('spurious')
            current, pos = sol, nxt
            step = np.sign(step)*min(CAP, abs(step)*1.2)
        except Exception:
            step /= 2
            if abs(step) < 1e-7:
                raise RuntimeError(f'stuck at Omega={pos} towards {target}')
    return current


def run(key):
    data = json.loads((ROOT/'results/egb-extremality.json').read_text())
    stored = sorted(data['walks'][key], key=lambda s: s['omega_h'])
    alpha = float(key)
    t0 = time.time()
    base = predictor_ladder(.6, alpha, step=.005, resolution=64)
    rows, profiles = [], {}
    up = [s for s in stored if s['omega_h'] >= .6]
    down = [s for s in stored if s['omega_h'] < .6][::-1]
    for branch in (up, down):
        current = base
        for s in branch:
            n = int(s['resolution'])
            try:
                if current.initial_resolution != n:
                    current = solve(float(current.omega_h), alpha, resolution=n, tol=1e-11,
                                    previous=current, jacobian='analytic')
                current = advance(current, float(s['omega_h']), alpha, n)
                mine = state_at(current)
            except Exception as err:
                rows.append(dict(omega_h=s['omega_h'], failed=repr(err)))
                continue
            rel = {k: abs(mine[k]-s[k])/max(abs(s[k]), 1e-300) for k in KEYS}
            rows.append(dict(omega_h=s['omega_h'], resolution=n,
                             stored={k: s[k] for k in KEYS},
                             mine={k: mine[k] for k in KEYS + ['max_relative_tensor_residual']},
                             rel=rel))
            profiles[f"{s['omega_h']:.12f}"] = coefficients(current).tolist()
    OUT.mkdir(exist_ok=True)
    (OUT/f'walk-{key}.json').write_text(json.dumps(dict(alpha=alpha, rows=rows,
                                                        profiles=profiles,
                                                        seconds=time.time()-t0)))
    worst = max((max(r['rel'].values()) for r in rows if 'rel' in r), default=np.nan)
    failed = sum('failed' in r for r in rows)
    return key, len(rows), failed, worst, time.time()-t0


if __name__ == '__main__':
    data = json.loads((ROOT/'results/egb-extremality.json').read_text())
    keys = list(data['walks'])
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    with Pool(min(13, len(keys))) as pool:
        for key, n, failed, worst, sec in pool.imap_unordered(run, keys):
            print(f'alpha={key:6s} states={n:3d} failed={failed} worst_rel={worst:.2e} ({sec:.0f}s)', flush=True)
