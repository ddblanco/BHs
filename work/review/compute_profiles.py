"""Radial profiles b,f,h,w at fixed r_H=1, Omega_H=0.33 for the 13 production couplings.

Same setting as Fig. 1 of arXiv:2607.07418 (r_H=1, Omega_H=0.33), whose coupling
is 4x ours (alpha_theirs in [0,3] <-> alpha in [0,0.75]). One continuation in alpha
from Myers-Perry with predictor seeds; each of the 13 couplings is accepted by the
project's tensor gate (state_at) and later re-checked by check_profiles.py.
Also computes the coldest reached state of each walk? No: that is reproduce_walks.py.
"""
import json
from pathlib import Path

import numpy as np

from rotating_bh.egb_rotating_predictor import solve
from rotating_bh.extremality import state_at
from rotating_bh.gb_potential import VacuumSeed
from reproduce_walks import coefficients

COUPLINGS = [0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
OMEGA = 0.33
N = 64

grid = sorted(set(np.round(np.r_[np.arange(0, 0.5001, 0.005), COUPLINGS], 12)))
current = solve(OMEGA, 0., resolution=N, tol=1e-11, previous=VacuumSeed(OMEGA))
out = {}
for alpha in grid:
    if alpha > 0:
        current = solve(OMEGA, float(alpha), resolution=N, tol=1e-11, previous=current,
                        seed='predictor', jacobian='analytic')
    if any(abs(alpha-c) < 1e-12 for c in COUPLINGS):
        st = state_at(current)
        out[f'{alpha:g}'] = dict(state={k: st[k] for k in ['E', 'J', 'S', 'T_H', 'psi_gb',
                                                             'max_relative_tensor_residual', 'smarr']},
                                 N=N, coefficients=coefficients(current).tolist())
        print(f"alpha={alpha:<6g} E={st['E']:.8f} J={st['J']:.8f} T={st['T_H']:.6f} "
              f"Psi={st['psi_gb']:+.6f} tensor={st['max_relative_tensor_residual']:.1e}", flush=True)
Path('out').mkdir(exist_ok=True)
Path('out/profiles-omega0.33.json').write_text(json.dumps(dict(omega_h=OMEGA, couplings=out)))
