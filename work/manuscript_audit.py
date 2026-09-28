"""Severe audit of the manuscript's printed algebra and of the saved dataset.

Every claim below is either derived with sympy and asserted to vanish, or read
back from ``results/egb-extremality.json`` and asserted against the number the
paper prints.  Nothing here re-solves the boundary-value problem: the solver
outputs are taken as given and only what the paper says *about* them is tested.

Run from the project root:  python3 work/manuscript_audit.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT/'results/egb-extremality.json').read_text(encoding='utf-8'))
NUMBERS = (ROOT/'manuscript/numbers.tex').read_text(encoding='utf-8')

PASS, FAIL = [], []


def check(name, condition, detail=''):
    (PASS if condition else FAIL).append(f'{name}: {detail}' if detail else name)
    print(('PASS  ' if condition else 'FAIL  ')+name+(f'   {detail}' if detail else ''))


def macro(name):
    """The body of \\Nname as it stands in the generated numbers.tex."""
    found = re.search(r'\\newcommand\{\\N'+name+r'\}\{(.*?)\}\n', NUMBERS, re.S)
    return found.group(1) if found else None


def macro_float(name):
    body = macro(name)
    if body is None:
        return None
    body = body.replace('$', '').strip()
    times = re.match(r'^(-?[\d.]+)\\times 10\^\{(-?\d+)\}$', body)
    if times:
        return float(times.group(1))*10.0**int(times.group(2))
    return float(body)


# ----------------------------------------------------------------- symbolic
r_H, alpha, lam = sp.symbols('r_H alpha lambda', positive=True)

# eq. (14): the Boulware-Deser charges the paper prints.
M_s = 3*sp.pi*(r_H**2+2*alpha)/8
T_s = r_H/(2*sp.pi*(r_H**2+4*alpha))
S_s = sp.pi**2*r_H**3/2+6*sp.pi**2*alpha*r_H

# eq. (15): Psi_static, obtained the way eq. (11) says to obtain it.
psi_s = sp.diff(M_s, alpha)-T_s*sp.diff(S_s, alpha)
printed = sp.Rational(3, 4)*sp.pi*(4*alpha/r_H**2-3)/(1+4*alpha/r_H**2)
check('eq.(15) Psi_static equals (3pi/4)(4a-3)/(1+4a)',
      sp.simplify(psi_s-printed) == 0)
check('eq.(15) -> -9pi/4 as alpha -> 0',
      sp.simplify(sp.limit(psi_s, alpha, 0)+9*sp.pi/4) == 0)
check('eq.(15) changes sign at alpha/r_H^2 = 3/4',
      sp.simplify(psi_s.subs(alpha, 3*r_H**2/4)) == 0)
check('static t_H = sqrt(1+2a)/(1+4a) matches eq.(10) on eq.(14)',
      sp.simplify(4*sp.sqrt(2*sp.pi/3)*T_s*sp.sqrt(M_s)
                  - sp.sqrt(1+2*alpha/r_H**2)/(1+4*alpha/r_H**2)) == 0)

# eq. (8): Smarr, closed on the static solution with the potential just derived.
check('eq.(8) Smarr closes on the static solution at J=0',
      sp.simplify(2*M_s-3*T_s*S_s-2*alpha*psi_s) == 0)

# eq. (8) from Euler's theorem on the weights the paper lists.  The two spins
# are kept distinct so that differentiating does not merge their slots.
S_, J1, J2, a_ = sp.symbols('S J_1 J_2 a', positive=True)
M_fun = sp.Function('M')
scaled = (M_fun(lam**3*S_, lam**3*J1, lam**3*J2, lam**2*a_)
          - lam**2*M_fun(S_, J1, J2, a_))
euler = sp.simplify(sp.diff(scaled, lam).subs(lam, 1))
T_, O1, O2, Psi_, M_ = sp.symbols('T Omega_1 Omega_2 Psi M')
base = M_fun(S_, J1, J2, a_)
euler = euler.subs({sp.Derivative(base, S_): T_, sp.Derivative(base, J1): O1,
                    sp.Derivative(base, J2): O2,
                    sp.Derivative(base, a_): Psi_, base: M_})
check('Euler on the listed weights gives 2M = 3TS + 3*sum(Omega_i J_i) + 2*alpha*Psi',
      sp.simplify(euler-(3*T_*S_+3*O1*J1+3*O2*J2+2*a_*Psi_-2*M_)) == 0)
Om_, J_ = sp.symbols('Omega J', positive=True)
equal_spin = euler.subs({O1: Om_, O2: Om_, J1: J_, J2: J_})
check('eq.(8) 2M = 3TS + 6*Omega_H*J + 2*alpha*Psi is its equal-spin case',
      sp.simplify(equal_spin-(3*T_*S_+6*Om_*J_+2*a_*Psi_-2*M_)) == 0)

# The Gibbs-Duhem constraint the paper prints just below eq. (11).
dS, dT, dJ, dOm, dPsi, da = sp.symbols('dS dT dJ dOmega dPsi dalpha')
d_smarr = 3*(T_*dS+S_*dT)+6*(Om_*dJ+J_*dOm)+2*(a_*dPsi+Psi_*da)
check('differentiated Smarr minus twice the first law gives the printed constraint',
      sp.simplify(d_smarr-2*(T_*dS+2*Om_*dJ+Psi_*da)
                  - (T_*dS+3*S_*dT+2*Om_*dJ+6*J_*dOm+2*a_*dPsi)) == 0)

# eq. (16)-(17): the published perturbative potential, and its two endpoints.
u, q = sp.symbols('u q', nonnegative=True)
psi0 = -sp.pi*(u**2-14*u+9)/4
check('eq.(17) Psi_0(u=0) = -9pi/4, the static limit of eq.(15)',
      sp.simplify(psi0.subs(u, 0)+9*sp.pi/4) == 0)
check('eq.(17) Psi_0(u=1) = pi, the published linear-order slope',
      sp.simplify(psi0.subs(u, 1)-sp.pi) == 0)
root = min(sp.solve(sp.Eq(u**2-14*u+9, 0), u), key=abs)
q_zero = sp.sqrt(root/(1+root))
check('eq.(17) sign change at the q the paper prints',
      abs(float(q_zero)-macro_float('perturbativezero')) < 5e-7,
      f'derived {float(q_zero):.9f} vs printed {macro_float("perturbativezero")}')

# eq. (9): the coordinate change and the extremal spin of the vacuum family.
a_mp, rp = sp.symbols('a_mp r_plus', positive=True)
rH_mp = sp.sqrt(rp**2+a_mp**2)
check('eq.(9) q = a/r_H follows from Omega_H = a/(r_+^2+a^2) and r_H^2 = r_+^2+a^2',
      sp.simplify(rH_mp*a_mp/(rp**2+a_mp**2)-a_mp/rH_mp) == 0)
check('eq.(9) u = q^2/(1-q^2) equals a^2/r_+^2',
      sp.simplify(((a_mp/rH_mp)**2/(1-(a_mp/rH_mp)**2))-a_mp**2/rp**2) == 0)
mu_mp = (rp**2+a_mp**2)**2/rp**2         # equal-spin d=5 Myers-Perry horizon
check('equal-spin Myers-Perry is extremal at r_+ = a',
      sp.simplify(sp.diff(mu_mp, rp).subs(rp, a_mp)) == 0)
check('extremality therefore sits at q = 1/sqrt(2)',
      sp.simplify((a_mp/rH_mp).subs(rp, a_mp)-1/sp.sqrt(2)) == 0)
check('appendix mass formula M = 3pi r_H^2/[8(1-q^2)] is 3pi mu/8',
      sp.simplify(3*sp.pi*rH_mp**2/(8*(1-(a_mp/rH_mp)**2))-3*sp.pi*mu_mp/8) == 0)

# ------------------------------------------------------------------- dataset
walks = {float(k): v for k, v in DATA['walks'].items()}
states = [s for group in walks.values() for s in group]
extremals = sorted(DATA['extremals'], key=lambda rec: rec['y'])

check('13 couplings were walked, as the paper says',
      len(walks) == 13 == int(macro('couplings')),
      f'{sorted(walks)}')
check('every walked state is a solve of the radial problem, not a rescaling',
      all('max_relative_tensor_residual' in s and 'resolution' in s for s in states)
      and {s['resolution'] for s in states} <= {48, 64, 96},
      f"{len(states)} states, resolutions {sorted({s['resolution'] for s in states})}")
check('the state count in the paper matches the dataset',
      len(states) == int(macro('states')), f'{len(states)}')
check('every coupling carries rotating states, not only the static member',
      all(max(s['omega_h'] for s in g) > 0.5 for g in walks.values()))
check('figure 2 plots rotating solutions: each curve has >= 25 spins',
      sum(1 for g in walks.values() if len(g) >= 25) == int(macro('mapcouplings')))
check('the independent tensor residual passes the gate the paper quotes',
      max(s['max_relative_tensor_residual'] for s in states) < 1e-8,
      f"worst {max(s['max_relative_tensor_residual'] for s in states):.3e}")

# The paper's own headline numbers, read back.
lowest = min(extremals, key=lambda rec: rec['psi_gb']['value'])
check('all 13 central Psi_ext are positive',
      all(rec['psi_gb']['value'] > 0 for rec in extremals))
check('the lowest central Psi_ext is the one printed',
      abs(lowest['psi_gb']['value']-macro_float('psimin')) < 5e-6)
check('the last central Psi_ext is the one printed',
      abs(extremals[-1]['psi_gb']['value']-macro_float('psilast')) < 5e-6)
check('mu is strictly increasing across the sampled couplings',
      all(b['mu'] > a['mu'] for a, b in zip(extremals, extremals[1:])))
check('the suppression factor pi/Psi_min is the one printed',
      abs(math.pi/lowest['psi_gb']['value']-macro_float('suppression')) < 5e-3)
check('the total change in mu is the one printed',
      abs((extremals[-1]['mu']-extremals[0]['mu'])-macro_float('mutotal')) < 5e-5)

fall = math.pi-lowest['psi_gb']['value']
rise = extremals[-1]['psi_gb']['value']-lowest['psi_gb']['value']
worst_band = max(rec['psi_uncertainty'] for rec in extremals)
check('the fall and the rise both exceed every quoted sensitivity band',
      fall > 10*worst_band and rise > 2*worst_band,
      f'fall {fall:.4f}, rise {rise:.4f}, worst band {worst_band:.4f}')

# The integral discrepancy: the endpoint and the largest running value are
# two different numbers and the paper now has a macro for each.
routes = DATA['route_comparison']
endpoint = routes['cumulative'][-1]['difference']
check('the endpoint and the largest running discrepancy are told apart',
      abs(macro_float('routecumulativeend')-endpoint) < 5e-3*endpoint
      and abs(macro_float('routecumulative')-routes['worst_cumulative'])
          < 5e-3*routes['worst_cumulative']
      and abs(endpoint-routes['worst_cumulative']) > 1e-9,
      f'endpoint {endpoint:.3e}, maximum running {routes["worst_cumulative"]:.3e}')
check('every interval discrepancy lies within the two routes\' combined spread',
      all(abs(row['difference']) <= row['combined_spread']
          for row in routes['intervals']),
      f'worst ratio {max(abs(r["difference"])/r["combined_spread"] for r in routes["intervals"]):.2f}')

# The Smarr-derived potential: a determination that does not use the response.
worst_rel, worst_abs, count = 0., 0., 0
for coupling, group in walks.items():
    if coupling == 0.:
        continue
    for s in group:
        implied = (2*s['E']-3*s['T_H']*s['S']-6*s['omega_h']*s['J'])/(2*coupling)
        worst_abs = max(worst_abs, abs(implied-s['psi_gb']))
        worst_rel = max(worst_rel, abs(implied-s['psi_gb'])/abs(s['psi_gb']))
        count += 1
check('Smarr-derived Psi reproduces the response Psi at every finite-coupling state',
      worst_rel < 1e-5, f'{count} states, worst relative {worst_rel:.3e}, '
                        f'worst absolute {worst_abs:.3e}')

# The extremal table's sensitivity column is only about Psi, yet y/mu/sigma
# also come from extrapolations that carry their own recorded spreads.
check('y, mu and sigma carry recorded spreads that the paper does not display',
      all({'y_spread', 'mu_spread', 'sigma_spread'} <= set(rec) for rec in extremals),
      f"worst mu spread {max(rec['mu_spread'] for rec in extremals):.3e}")

# The Smarr route at the extremal intercepts, and the fit diagnostics.
import importlib.util
spec = importlib.util.spec_from_file_location(
    'reanalysis', ROOT/'manuscript/reanalysis.py')
reanalysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reanalysis)
worst_intercept = 0.
for record in extremals:
    coupling = record['alpha_gb']
    if coupling == 0.:
        continue
    relabelled = [dict(s, psi_gb=reanalysis.smarr_potential(s, coupling))
                  for s in walks[coupling]]
    intercept = reanalysis.production_fit(relabelled)['intercept']
    central = record['psi_gb']['value']
    worst_intercept = max(worst_intercept, abs(intercept-central)/abs(central))
check('the Smarr route reproduces the printed worst relative agreement',
      abs(worst_intercept-macro_float('smarrextworst'))
      < .02*macro_float('smarrextworst'),
      f'{worst_intercept:.3e}')
check('the Smarr route agrees far inside every fit sensitivity',
      worst_intercept*max(rec['psi_gb']['value'] for rec in extremals)
      < min(rec['psi_uncertainty'] for rec in extremals)/100.)
smallest_band = min(rec['psi_uncertainty'] for rec in extremals)
check('the fit residual and the drop-one shift are below every sensitivity',
      macro_float('fitresidualworst') < smallest_band
      and macro_float('looworst') < smallest_band,
      f'smallest band {smallest_band:.3e}')

# ------------------------------------------------- the radial-resolution study
STUDY_PATH = ROOT/'results/egb-rotating-resolution-study.json'
if not STUDY_PATH.exists():
    check('the radial-resolution study artifact exists', False,
          'run experiments/egb_rotating_resolution_study.py')
else:
    STUDY = json.loads(STUDY_PATH.read_text(encoding='utf-8'))
    rows = STUDY['comparisons']
    couplings = STUDY['extremal_couplings']
    check('the study covers every coupling at every coarse resolution',
          len(rows) == len(couplings)*len(STUDY['coarse_resolutions'])
          and {r['alpha_gb'] for r in rows} == set(couplings),
          f"{len(rows)} sequences over {len(couplings)} couplings at "
          f"N={STUDY['coarse_resolutions']}")
    check('the production side was reproduced from the shipped states, not re-solved',
          STUDY['production_reproduced_to'] < 1e-10,
          f"{STUDY['production_reproduced_to']:.2e} absolute")
    envelopes = {rec['alpha_gb']: rec['psi_uncertainty'] for rec in extremals}
    worst_ratio = max(r['psi_gb']['absolute']/envelopes[r['alpha_gb']] for r in rows)
    check('the resolution difference is below the fit sensitivity everywhere',
          worst_ratio < 1., f'worst ratio {worst_ratio:.2e}')
    fall = math.pi-min(rec['psi_gb']['value'] for rec in extremals)
    worst_psi = max(r['psi_gb']['absolute'] for r in rows)
    check('the resolution difference is far below the fall and the rise',
          worst_psi < fall/1000., f'{worst_psi:.2e} against a fall of {fall:.3f}')
    check('the printed worst-case study numbers match the artifact',
          abs(worst_psi-macro_float('studypsiworst')) < 5e-3*worst_psi
          and abs(max(r['mu']['absolute'] for r in rows)
                  - macro_float('studymuworst')) < 5e-3*macro_float('studymuworst'))
    control = STUDY.get('environment_control')
    if control:
        worst_control = max(r['psi_gb']['absolute'] for r in control)
        check('the environment alone accounts for less than the resolution change',
              worst_control <= worst_psi,
              f'environment {worst_control:.2e} vs resolution+environment '
              f'{worst_psi:.2e}')
    else:
        check('an environment control was recorded', False,
              'rerun the study to add it')

# Resolution coverage of the shipped measurement, which the study supplements.
check('the shipped measurement carries its own same-environment comparison',
      len(DATA['resolution_contrast']) == 3,
      f"at alpha = {[rec['alpha_gb'] for rec in DATA['resolution_contrast']]}")

print()
print(f'{len(PASS)} passed, {len(FAIL)} failed')
for line in FAIL:
    print('  FAILED: '+line)
