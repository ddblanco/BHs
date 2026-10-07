"""Is the extremal solution non-smooth at the horizon, with the exponent of
nearhorizon.py?

In s = r/(1+r) a term r^(2 gamma) at the horizon makes the Chebyshev
coefficients of Q decay algebraically, |c_n| ~ n^(-(4 gamma + 1)) (endpoint
singularity s^p gives n^(-2p-1), here p = 2 gamma); an analytic solution would
give geometric decay. At three couplings the solution is computed at N = 128
(filtered restart as in branch_rq.py), the envelope of |c_n| of the field
with the largest tail is fitted in log-log and log-linear form over the range
above the round-off plateau, and the fitted power is compared with
4 gamma + 1 from the perturbation analysis at the same alpha/g_H.

Run: python gamma_check.py  ->  gamma_check.json
"""
import json
import warnings
from pathlib import Path

import numpy as np

from extremal_rq import ExtremalRQ, myers_perry_q
from branch_rq import robust_solve
from nearhorizon import NearHorizon

warnings.simplefilter('ignore', RuntimeWarning)
HERE = Path(__file__).resolve().parent


def envelope(c):
    """Running maximum from the right: removes the odd/even zig-zag."""
    a = np.abs(c)
    return np.maximum.accumulate(a[::-1])[::-1]


def main():
    nh = NearHorizon()
    out = []
    for target in (0.03, 0.2, 1.0):
        e = ExtremalRQ(48)
        f = myers_perry_q(48)
        for a in np.linspace(0, target, int(40*max(1, target/0.2)))[1:]:
            f, rec = robust_solve(e, a, f)
        f = e.interpolate(f, 128)
        e = ExtremalRQ(128)
        f, rec = robust_solve(e, target, f)
        s = nh.solve(target, [.25, 2., -.5]) if target < .05 else None
        # exponent gamma at this alpha/g_H (continuation from MP for safety)
        guess = np.array([.25, 2., -.5])
        for a in np.linspace(0, target, 60)[1:]:
            s = nh.solve(a, guess)
            guess = np.array([s['v1'], s['H0'], s['w1']])
        gam = max(x for x in nh.exponents(s) if x < 2.5)
        coeffs = e.coefficients(f)
        best = None
        for name, c in zip(('Q1', 'Q2', 'Q3', 'Q4'), coeffs):
            env = envelope(c)
            n = np.arange(len(c))
            floor = 50*np.median(env[-10:]) if np.median(env[-10:]) > 0 else 1e-300
            k = (n >= 12) & (env > floor)
            if k.sum() < 10:
                continue
            p_alg = np.polyfit(np.log(n[k]), np.log(env[k]), 1)
            p_geo = np.polyfit(n[k], np.log(env[k]), 1)
            r_alg = np.std(np.log(env[k])-np.polyval(p_alg, np.log(n[k])))
            r_geo = np.std(np.log(env[k])-np.polyval(p_geo, n[k]))
            cand = dict(field=name, n_range=[int(n[k].min()), int(n[k].max())],
                        power=float(-p_alg[0]), rms_algebraic=float(r_alg),
                        rate_geometric=float(-p_geo[0]), rms_geometric=float(r_geo))
            if best is None or cand['n_range'][1]-cand['n_range'][0] > best['n_range'][1]-best['n_range'][0]:
                best = cand
        rec = dict(alpha=target, gamma=float(gam), predicted_power=float(4*gam+1), fit=best,
                   constraint=rec['constraint'], tails=rec['tails'])
        out.append(rec)
        print(f"alpha={target}: gamma={gam:.4f} -> predicted |c_n| ~ n^-{4*gam+1:.2f}; fit on {best['field']} "
              f"n in {best['n_range']}: algebraic power {best['power']:.2f} (rms {best['rms_algebraic']:.2f}) vs "
              f"geometric (rms {best['rms_geometric']:.2f})")
    (HERE/'gamma_check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
