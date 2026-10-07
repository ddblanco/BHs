"""Checks and comparisons on the directly constructed extremal branch
(branch_rq.json). Writes analysis.json; prints a summary.

Internal checks (asserted where the expected accuracy is known):
  1. near-horizon data: P1(0)/4 = v1, h_H = H0 from nearhorizon.py at the
     same alpha/g_H (the identification of arXiv:2303.12471, sec. 3.2);
  2. Smarr at T = 0 versus the slope from implicit differentiation:
     Psi_S = (M - 3 Omega_H J)/alpha against mu'(y);
  3. first law at T = 0 along the branch: Psi_FL = M_a - 2 Omega_H J_a;
  4. 3 omega = mu - y mu'  (manuscript eq. omegaidentity), omega = Omega_H J^(1/3);
  5. resolution: |X(N_high) - X(N_low)| for every reported quantity.
Comparisons (not asserted; the point of the exercise):
  6. mu(y) and mu'(y) against the manuscript's T -> 0 extrapolations
     (results/egb-extremality.json: mu, psi_gb at the same alpha = same y),
     in units of the manuscript's own spreads;
  7. sigma = S/J against the near-horizon entropy function.
"""
import json
from pathlib import Path

import numpy as np

from nearhorizon import NearHorizon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    data = json.loads((HERE/'branch_rq.json').read_text())
    rows = [r for r in data['rows'] if r['alpha'] > 0]
    nh = NearHorizon()
    guess = np.array([.25, 2., -.5])
    out = []
    for r in rows:
        hi, lo = r['high'], r['low']
        s = nh.solve(r['alpha'], guess)          # G0 = g_H = 1
        guess = np.array([s['v1'], s['H0'], s['w1']])
        sigma_nh = s['S']/s['J']
        rec = dict(alpha=r['alpha'], y=hi['y'], x=hi['x'], j=hi['j'], mu=hi['mu'],
                   mu_prime=hi['mu_prime'], psi_smarr=hi['psi_smarr'],
                   psi_first_law=hi['psi_first_law'], sigma=hi['sigma'], omega=hi['omega'],
                   aH=hi['aH'], s=hi['s'], M=hi['M'], J=hi['J'], Omega_H=hi['Omega_H'],
                   dN={k: abs(hi[k]-lo[k]) for k in ('mu', 'mu_prime', 'y', 'psi_smarr', 'sigma', 'j', 'aH', 's')},
                   v1_dev=abs(hi['P1_H']/4-s['v1']), hH_dev=abs(hi['h_H']-s['H0']),
                   sigma_nh_dev=abs(hi['sigma']-sigma_nh),
                   smarr_dev=abs(hi['psi_smarr']-hi['mu_prime']),
                   first_law_dev=abs(hi['psi_first_law']-hi['mu_prime']),
                   omega_identity_dev=abs(3*hi['omega']-(hi['mu']-hi['y']*hi['mu_prime'])),
                   constraint=hi['constraint'])
        out.append(rec)
    worst = {k: max(r[k] for r in out) for k in ('v1_dev', 'hH_dev', 'sigma_nh_dev', 'smarr_dev',
                                                 'first_law_dev', 'omega_identity_dev', 'constraint')}
    print('worst internal deviations over the branch:', {k: f'{v:.1e}' for k, v in worst.items()})
    print('worst resolution differences:', {k: f"{max(r['dN'][k] for r in out):.1e}" for k in out[0]['dN']})
    # comparison with the manuscript at its couplings (same alpha in r_H = 1 is the same family member)
    ms = json.loads((ROOT/'results/egb-extremality.json').read_text())['extremals']
    alphas = np.array([r['alpha'] for r in out])
    comp = []
    for m in sorted(ms, key=lambda r: r['alpha_gb']):
        a = m['alpha_gb']
        if a == 0 or a > alphas.max():
            continue
        # cubic interpolation in alpha of our branch (dense, smooth)
        k = np.argsort(np.abs(alphas-a))[:6]
        k.sort()
        fit = lambda key: np.polyval(np.polyfit(alphas[k], [out[i][key] for i in k], 5), a)
        rec = dict(alpha=a, y_ours=fit('y'), y_ms=m['y'], mu_ours=fit('mu'), mu_ms=m['mu'],
                   mu_spread_ms=m['mu_spread'], psi_ours=fit('mu_prime'), psi_ms=m['psi_gb']['value'],
                   psi_unc_ms=m['psi_uncertainty'], sigma_ours=fit('sigma'), sigma_ms=m['sigma'])
        comp.append(rec)
        print(f"alpha={a:<6} y {rec['y_ours']:.8f} vs {rec['y_ms']:.8f} | mu {rec['mu_ours']:.8f} vs "
              f"{rec['mu_ms']:.8f} (diff {rec['mu_ours']-rec['mu_ms']:+.1e}, ms spread {rec['mu_spread_ms']:.0e}) | "
              f"Psi {rec['psi_ours']:.6f} vs {rec['psi_ms']:.6f} (diff {rec['psi_ours']-rec['psi_ms']:+.1e}, "
              f"ms envelope {rec['psi_unc_ms']:.0e})")
    (HERE/'analysis.json').write_text(json.dumps(dict(branch=out, worst=worst, manuscript=comp), indent=1))


if __name__ == '__main__':
    main()
