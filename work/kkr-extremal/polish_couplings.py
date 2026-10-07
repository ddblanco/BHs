"""High-resolution extremal solutions at the manuscript's couplings.

For each alpha of results/egb-extremality.json (r_H = 1), the nearest branch
solution of branch_rq.json is continued to that alpha and refined through
N = 64, 96, 128 (each seeded by interpolating the previous; filtered restart
as in branch_rq.py). Recorded per N: mu, y, sigma, mu' (implicit
differentiation), Psi_S, Psi_FL, constraint, tails. The reported value is the
N = 128 one and its uncertainty max(|X128 - X96|, |X96 - X64|): a resolution
difference, not a bound.

Run: python polish_couplings.py  ->  polished.json
"""
import json
import warnings
from pathlib import Path

import numpy as np

from extremal_rq import ExtremalRQ
from branch_rq import robust_solve

warnings.simplefilter('ignore', RuntimeWarning)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CH = np.polynomial.chebyshev
KEYS = ('mu', 'y', 'sigma', 'mu_prime', 'psi_smarr', 'psi_first_law', 'x', 'j', 'aH', 's', 'omega', 'M', 'J')


def main():
    rows = json.loads((HERE/'branch_rq.json').read_text())['rows']
    ms = sorted(json.loads((ROOT/'results/egb-extremality.json').read_text())['extremals'],
                key=lambda r: r['alpha_gb'])
    out = []
    for m in ms:
        a = m['alpha_gb']
        if a == 0:
            continue
        near = min(rows, key=lambda r: abs(r['alpha']-a))
        e = ExtremalRQ(near.get('n_high', 72))
        f = np.array([CH.chebval(1-2*e.x, c) for c in near['coeffs_high']]).ravel()
        f, rec = robust_solve(e, a, f)
        per_n = {}
        for n in (64, 96, 128):
            f = e.interpolate(f, n)
            e = ExtremalRQ(n)
            f, rec = robust_solve(e, a, f)
            per_n[n] = {k: rec[k] for k in KEYS+('constraint', 'residual')}
            per_n[n]['tail'] = max(rec['tails'])
        best = per_n[128]
        unc = {k: max(abs(per_n[128][k]-per_n[96][k]), abs(per_n[96][k]-per_n[64][k])) for k in KEYS}
        r = dict(alpha=a, per_n=per_n, value={k: best[k] for k in KEYS}, resolution=unc,
                 manuscript=dict(y=m['y'], mu=m['mu'], mu_spread=m['mu_spread'], sigma=m['sigma'],
                                 sigma_spread=m['sigma_spread'], psi=m['psi_gb']['value'],
                                 psi_envelope=m['psi_uncertainty'], y_spread=m['y_spread']))
        out.append(r)
        print(f"alpha={a:<6} y {best['y']:.9f} (+-{unc['y']:.0e}) ms {m['y']:.9f} | mu {best['mu']:.9f} "
              f"(+-{unc['mu']:.0e}) ms {m['mu']:.9f} (+-{m['mu_spread']:.0e}) | mu' {best['mu_prime']:.7f} "
              f"(+-{unc['mu_prime']:.0e}) ms Psi {m['psi_gb']['value']:.7f} (+-{m['psi_uncertainty']:.0e}) | "
              f"C {best['constraint']:.0e}", flush=True)
        (HERE/'polished.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
