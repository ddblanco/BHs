"""Numerical and symbolic validation of the manuscript.

Run from the project root: .venv/Scripts/python reports/revision-manuscript-checks.py
Does not modify production data. Writes separate validation artifacts.
"""
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp

from rotating_bh.consistency import first_law_residual, static_closed_form, goon_penco_residual
from rotating_bh.egb_rotating_predictor import solve
from rotating_bh.egb_rotating_observables import measure
from rotating_bh.egb_rotating_scale import scaled_diagnose
from rotating_bh.gb_response import predictor_ladder


def saved_checks():
    data = json.loads(Path('results/egb-extremality.json').read_text())
    fits = []
    for ext in data['extremals']:
        states = sorted(data['walks'][format(ext['alpha_gb'], '.12g')],
                        key=lambda s: s['T_H'])[:12]
        values = np.array([s['psi_gb'] for s in states])
        row = dict(alpha=ext['alpha_gb'], saved=ext['psi_gb']['value'],
                   quoted_uncertainty=ext['psi_uncertainty'])
        for key in ('T_H', 'tau'):
            coordinate = np.array([s[key] for s in states])
            for count in (6, 12):
                row[f'{key}_{count}'] = float(np.polynomial.polynomial.polyfit(
                    coordinate[:count], values[:count], 3)[0])
        fits.append(row)
    Path('reports/revision-manuscript-fits.json').write_text(
        json.dumps(fits, indent=2)+'\n', encoding='utf-8')
    # Wu-Lu 2405.04576v2, eq. (33): Delta G / alpha in odd D.
    D, r, a, u = sp.symbols('D r a u', positive=True)
    sigma = sp.Symbol('Sigma')
    correction = -(D-3)*sigma/(16*sp.pi*r**4*(a*a+r*r)**((5-D)/2)) * (
        a**4-2*(2*D-3)*a*a*r*r+(D-2)**2*r**4)
    five = sp.simplify(correction.subs({D: 5, sigma: 2*sp.pi**2}))
    psi = sp.simplify(five.subs(a, r*sp.sqrt(u)))
    assert sp.simplify(psi + sp.pi/4*(u*u-14*u+9)) == 0
    print('Wu-Lu D=5 specialization:', psi, flush=True)
    print('Saved intercepts reproduced by cubic T fits to six points:',
          all(abs(row['saved']-row['T_H_6']) < 1e-12 for row in fits), flush=True)


def main():
    saved_checks()
    if '--saved-only' in sys.argv:
        return
    rows = []
    for n in (32, 48, 64):
        current = predictor_ladder(0., .5, step=.025, resolution=n)
        for q in (0., .01, .03, .05, .07, .1):
            if q:
                current = solve(q, .5, resolution=n, tol=1e-11,
                                previous=current, seed='predictor')
            obs = measure(current)
            row = dict(N=n, q=q, M=obs['E'], S=obs['S'], J=obs['J'],
                       mass_minus_static=obs['E']-static_closed_form(.5)['E'],
                       first_law=first_law_residual(current),
                       mass_tail_spread=obs['mass_tail_spread'])
            if q in (0., .07):
                row['tensor'] = scaled_diagnose(current, .5, count=51)
            if q == 0.:
                row['goon_penco_static'] = goon_penco_residual(current)
            rows.append(row)
            print(json.dumps(row), flush=True)
    data = json.loads(Path('results/egb-extremality.json').read_text())
    summary = dict(fresh_small_spin=rows,
                   max_condition_point=max(data['consistency'],
                       key=lambda r: r['conditioning']['condition_number']),
                   static_temperature_at_half=static_closed_form(.5)['t_scaled'])
    for row in rows:
        if not np.isfinite(row['first_law']['relative']):
            row['first_law']['relative'] = None  # all terms vanish at q=0
    Path('reports/revision-manuscript-checks.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False)+ '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
