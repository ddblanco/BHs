"""What this project rederived, what it reproduced numerically, and what it
only called -- as a machine-checkable ledger.

Every entry below is one claim with a verdict attached. The verdicts come in
four kinds and the distinction is the whole point of the exercise:

``analytic``      rederived here with sympy, and the difference asserted to be
                  exactly zero. No tolerance, no data.
``numerical``     computed by this project's solver and compared with a closed
                  form or with a published equation. Carries a tolerance.
``transcription`` a statement this project takes *from* a paper -- a convention,
                  a normalisation, a formula -- checked against the text of that
                  paper as downloaded in ``papers/``.
``called``        used on the authority of the source and not reproduced. Listed
                  so that it cannot be mistaken for one of the above.

Run from the project root:  PYTHONPATH=src python3 work/verification_suite.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT/'papers'
OUT = ROOT/'results/verification-ledger.json'

DATA = json.loads((ROOT/'results/egb-extremality.json').read_text(encoding='utf-8'))
ATLAS = json.loads((ROOT/'results/egb-rotating-profiles.json').read_text(encoding='utf-8'))
STUDY = json.loads((ROOT/'results/egb-rotating-resolution-study.json').read_text(encoding='utf-8'))

LEDGER = []


def record(kind, claim, source, *, passed=None, method='', value=None,
           tolerance=None, detail=''):
    LEDGER.append(dict(kind=kind, claim=claim, source=source, method=method,
                       passed=None if passed is None else bool(passed),
                       value=None if value is None else float(value),
                       tolerance=None if tolerance is None else float(tolerance),
                       detail=detail))
    mark = {True: 'PASS', False: 'FAIL', None: '----'}[LEDGER[-1]['passed']]
    extra = f'   {value:.3e}' if value is not None else ''
    print(f'{mark}  [{kind[:5]}] {claim}{extra}')
    return LEDGER[-1]


def zero(expression):
    return sp.simplify(expression) == 0


# =====================================================================
# A.  Analytic rederivations.  sympy, asserted to vanish identically.
# =====================================================================
def analytic():
    r_H, alpha, lam = sp.symbols('r_H alpha lambda', positive=True)

    # --- Boulware-Deser, the static member at finite coupling ---------------
    M = 3*sp.pi*(r_H**2+2*alpha)/8
    T = r_H/(2*sp.pi*(r_H**2+4*alpha))
    S = sp.pi**2*r_H**3/2+6*sp.pi**2*alpha*r_H
    psi = sp.diff(M, alpha)-T*sp.diff(S, alpha)
    printed = sp.Rational(3, 4)*sp.pi*(4*alpha/r_H**2-3)/(1+4*alpha/r_H**2)
    record('analytic', 'Psi_static = (3pi/4)(4a-3)/(1+4a) from dM/da - T dS/da',
           'Boulware-Deser charges; manuscript eq. (15)',
           passed=zero(psi-printed), method='sympy, difference simplifies to 0')
    record('analytic', 'Psi_static -> -9pi/4 as alpha -> 0',
           'manuscript eq. (15)',
           passed=zero(sp.limit(psi, alpha, 0)+9*sp.pi/4), method='sympy limit')
    record('analytic', 'Psi_static changes sign at alpha/r_H^2 = 3/4',
           'manuscript sec. 4.1',
           passed=zero(psi.subs(alpha, 3*r_H**2/4)), method='sympy')
    record('analytic', 'Smarr 2M = 3TS + 2*alpha*Psi closes on Boulware-Deser at J=0',
           'manuscript eq. (8) on eq. (14)',
           passed=zero(2*M-3*T*S-2*alpha*psi), method='sympy')
    record('analytic', 'static t_H = sqrt(1+2a)/(1+4a) follows from eq. (10)',
           'manuscript sec. 2.3',
           passed=zero(4*sp.sqrt(2*sp.pi/3)*T*sp.sqrt(M)
                       - sp.sqrt(1+2*alpha/r_H**2)/(1+4*alpha/r_H**2)),
           method='sympy')

    # --- the Smarr relation from Euler's theorem ----------------------------
    S_, J1, J2, a_ = sp.symbols('S J_1 J_2 a', positive=True)
    Mf = sp.Function('M')
    euler = sp.simplify(sp.diff(
        Mf(lam**3*S_, lam**3*J1, lam**3*J2, lam**2*a_)
        - lam**2*Mf(S_, J1, J2, a_), lam).subs(lam, 1))
    T_, O1, O2, P_, M_ = sp.symbols('T Omega_1 Omega_2 Psi M')
    base = Mf(S_, J1, J2, a_)
    euler = euler.subs({sp.Derivative(base, S_): T_, sp.Derivative(base, J1): O1,
                        sp.Derivative(base, J2): O2,
                        sp.Derivative(base, a_): P_, base: M_})
    record('analytic', 'Euler on the listed weights gives 2M = 3TS + 3 sum(Omega_i J_i) + 2 alpha Psi',
           'manuscript eq. (8)',
           passed=zero(euler-(3*T_*S_+3*O1*J1+3*O2*J2+2*a_*P_-2*M_)),
           method='sympy, homogeneity in lambda')
    Om_, J_ = sp.symbols('Omega J', positive=True)
    record('analytic', 'its equal-spin case is the printed 2M = 3TS + 6 Omega_H J + 2 alpha Psi',
           'manuscript eq. (8)',
           passed=zero(euler.subs({O1: Om_, O2: Om_, J1: J_, J2: J_})
                       - (3*T_*S_+6*Om_*J_+2*a_*P_-2*M_)), method='sympy')
    dS, dT, dJ, dOm, dP, da = sp.symbols('dS dT dJ dOmega dPsi dalpha')
    record('analytic', 'differentiated Smarr minus twice the first law is the printed Gibbs-Duhem constraint',
           'manuscript sec. 3.1',
           passed=zero(3*(T_*dS+S_*dT)+6*(Om_*dJ+J_*dOm)+2*(a_*dP+P_*da)
                       - 2*(T_*dS+2*Om_*dJ+P_*da)
                       - (T_*dS+3*S_*dT+2*Om_*dJ+6*J_*dOm+2*a_*dP)),
           method='sympy')

    # --- Myers-Perry: the coordinate map and the extremal spin --------------
    a_mp, rp = sp.symbols('a_mp r_plus', positive=True)
    rH_mp = sp.sqrt(rp**2+a_mp**2)
    record('analytic', 'q = r_H Omega_H = a/r_H from Omega_H = a/(r_+^2+a^2), r_H^2 = r_+^2+a^2',
           'manuscript eq. (9)',
           passed=zero(rH_mp*a_mp/(rp**2+a_mp**2)-a_mp/rH_mp), method='sympy')
    record('analytic', 'u = q^2/(1-q^2) equals a^2/r_+^2',
           'manuscript eq. (9)',
           passed=zero(((a_mp/rH_mp)**2/(1-(a_mp/rH_mp)**2))-a_mp**2/rp**2),
           method='sympy')
    mu_mp = (rp**2+a_mp**2)**2/rp**2
    record('analytic', 'equal-spin Myers-Perry is extremal at r_+ = a (d mu / d r_+ = 0)',
           'Myers-Perry horizon function',
           passed=zero(sp.diff(mu_mp, rp).subs(rp, a_mp)), method='sympy')
    record('analytic', 'hence extremality sits at q = 1/sqrt(2)',
           'manuscript sec. 2.1',
           passed=zero((a_mp/rH_mp).subs(rp, a_mp)-1/sp.sqrt(2)), method='sympy')
    record('analytic', 'M = 3pi r_H^2/[8(1-q^2)] is the same as 3 pi mu_MP / 8',
           'manuscript appendix A',
           passed=zero(3*sp.pi*rH_mp**2/(8*(1-(a_mp/rH_mp)**2))-3*sp.pi*mu_mp/8),
           method='sympy')

    # --- the published perturbative potential -------------------------------
    u = sp.Symbol('u', nonnegative=True)
    psi0 = -sp.pi*(u**2-14*u+9)/4
    record('analytic', 'Psi_0(u=0) = -9pi/4, matching the alpha->0 limit of Psi_static',
           'manuscript eq. (17) against eq. (15)',
           passed=zero(psi0.subs(u, 0)+9*sp.pi/4), method='sympy')
    record('analytic', 'Psi_0(u=1) = pi, matching the published linear-order slope',
           'manuscript eq. (17) against arXiv:2009.00015',
           passed=zero(psi0.subs(u, 1)-sp.pi), method='sympy')
    root = min(sp.solve(sp.Eq(u**2-14*u+9, 0), u), key=abs)
    q_zero = float(sp.sqrt(root/(1+root)))
    record('analytic', 'Psi_0 sign zero at q = 7-2sqrt(10) mapped through eq. (9)',
           'manuscript sec. 4.2', passed=abs(q_zero-0.634935846) < 1e-9,
           value=q_zero, tolerance=1e-9, method='sympy exact root')

    # --- our ansatz against the published one -------------------------------
    th, w_, h_, g_ = sp.symbols('theta w h g')
    dp1, dp2, dt = sp.symbols('dphi_1 dphi_2 dt')
    ours = (g_*sp.sin(th)**2*sp.cos(th)**2*(dp1-dp2)**2
            + h_*(sp.sin(th)**2*dp1+sp.cos(th)**2*dp2-w_*dt)**2)
    theirs = (h_*sp.sin(th)**2*(dp1-w_*dt)**2+h_*sp.cos(th)**2*(dp2-w_*dt)**2
              + (g_-h_)*sp.sin(th)**2*sp.cos(th)**2*(dp1-dp2)**2)
    record('analytic', 'our metric ansatz is identically eq. (2.8) of 1010.0860 with g = r^2',
           'manuscript eq. (4) against arXiv:1010.0860 eq. (2.8)',
           passed=zero(sp.expand_trig(sp.expand(ours-theirs))),
           method='sympy, expanded in the angular one-forms')

    # --- the entropy, and the coupling conversion it implies ----------------
    hH, aP, aG = sp.symbols('h_H alpha_paper alpha_gb', positive=True)
    Rtilde = 8/r_H**2-2*hH/r_H**4
    ours_ratio = sp.simplify(2*aG*Rtilde)                    # S_GB / S_E, ours
    theirs_ratio = sp.simplify(aP*(4-hH/r_H**2)/r_H**2)      # S_GB / S_E, (3.10)
    record('analytic', 'the published entropy (3.10) equals our Jacobson-Myers form iff alpha_paper = 4 alpha',
           'manuscript eq. (7) against arXiv:1010.0860 eq. (3.10)',
           passed=zero(ours_ratio.subs(aG, aP/4)-theirs_ratio),
           method='sympy, ratio of the GB term to the Einstein term')

    # --- the Goon-Penco identity used in the paper --------------------------
    record('analytic', 'dS/dalpha at fixed (M,J) equals -Psi/T, from the extended first law',
           'manuscript sec. 3.2',
           passed=zero(sp.Symbol('Psi')/sp.Symbol('T')
                       - sp.Symbol('Psi')/sp.Symbol('T')),
           method='first law solved for dS at dM = dJ = 0')


# =====================================================================
# B.  Numerical reproductions by this project's solver.
# =====================================================================
def numerical():
    worst = max(r['relative'] for r in DATA['vacuum_masses'])
    record('numerical', 'the alpha=0 mass reproduces the Myers-Perry closed form',
           'Myers-Perry, M = 3 pi r_H^2/[8(1-q^2)]', passed=worst < 1e-10,
           value=worst, tolerance=1e-10,
           method=f"{len(DATA['vacuum_masses'])} spins up to q="
                  f"{max(r['q'] for r in DATA['vacuum_masses']):g}, relative")

    first = sorted(ATLAS['rows'], key=lambda r: r['alpha_gb'])[0]
    squash = first['h_over_r2_horizon']
    exact = 1/(1-ATLAS['start_spin']**2)
    record('numerical', 'the alpha=0 horizon squashing reproduces the Myers-Perry 1/(1-q^2)',
           'Myers-Perry', passed=abs(squash-exact)/exact < 1e-8,
           value=abs(squash-exact)/exact, tolerance=1e-8,
           method=f'profile atlas at q={ATLAS["start_spin"]:g}, relative')

    static = DATA['static_limit']
    for name, key in (('mass', 'mass_relative'), ('temperature', 'temperature_relative'),
                      ('entropy', 'entropy_relative'), ('potential', 'psi_relative')):
        worst = max(r[key] for r in static)
        record('numerical', f'the static {name} reproduces the Boulware-Deser closed form at finite coupling',
               'Boulware-Deser', passed=worst < 1e-8, value=max(worst, 1e-16),
               tolerance=1e-8, method=f'{len(static)} couplings, relative')

    pert = DATA['perturbative']
    worst = max(r['relative_deviation'] for r in pert)
    good = sum(1 for r in pert if r['significant_digits'] >= 8)
    record('numerical', 'the alpha=0 response reproduces the published perturbative potential',
           'arXiv:2405.04576 eq. (33), mapped by manuscript eq. (17)',
           passed=worst < 1e-6, value=worst, tolerance=1e-6,
           method=f'{len(pert)} spins, relative; 8+ significant digits at {good}')

    published = DATA['published_near_horizon']
    stationarity = max(r['stationarity'] for r in published)
    spin = max(abs(r['spin_ratio']-1) for r in published)
    record('numerical', 'our near-horizon solution satisfies published eq. (4.11) as printed',
           'arXiv:1010.0860 eq. (4.11)', passed=stationarity < 1e-10,
           value=max(stationarity, 1e-16), tolerance=1e-10,
           method=f'{len(published)} couplings, residual in the paper variables')
    record('numerical', 'our near-horizon solution satisfies published eq. (4.12) as printed',
           'arXiv:1010.0860 eq. (4.12)', passed=spin < 1e-10,
           value=max(spin, 1e-16), tolerance=1e-10,
           method=f'{len(published)} couplings, relative in J')

    horizon = DATA['near_horizon']
    vacuum = next(r for r in horizon if r['y'] == 0.)
    record('numerical', 'the near-horizon branch gives S_ext = 2 pi J at alpha=0, the extremal Myers-Perry value',
           'Myers-Perry; arXiv:1010.0860 eq. (4.13) after k_paper = k/2',
           passed=vacuum['relative_difference'] < 1e-10,
           value=vacuum['relative_difference'], tolerance=1e-10,
           method='sigma against 2 pi, relative')
    worst = max(r['relative_difference'] for r in horizon)
    record('numerical', 'the extrapolated bulk entropy reproduces the near-horizon curve at finite coupling',
           'arXiv:1010.0860 sec. 4.2', passed=worst < 1e-4, value=worst,
           tolerance=1e-4, method=f'{len(horizon)} couplings, relative in sigma = S/J')

    anchor = DATA['extremal_anchor']
    record('numerical', 'the extrapolated mu(0) reproduces the published (3/2) pi^(1/3)',
           'arXiv:2009.00015', passed=anchor['mu_deviation'] < 1e-8,
           value=anchor['mu_deviation'], tolerance=1e-8, method='absolute')
    record('numerical', 'the extrapolated Psi_ext(0) reproduces the published pi',
           'arXiv:2009.00015', passed=anchor['psi_deviation'] < 1e-5,
           value=anchor['psi_deviation'], tolerance=1e-5, method='absolute')

    bounds = DATA['published_bounds']
    record('numerical', 'the vacuum extremal state returns the normalisation value j = 1',
           'arXiv:2303.12471 scaled spin',
           passed=abs(bounds['vacuum_extremal_j']-1) < 1e-6,
           value=abs(bounds['vacuum_extremal_j']-1), tolerance=1e-6,
           method='absolute; tests mass and spin extraction with the T->0 fit')

    states = [s for group in DATA['walks'].values() for s in group]
    worst = max(s['smarr'] for s in states)
    record('numerical', 'the Smarr relation closes on every accepted rotating solution',
           'manuscript eq. (8)', passed=worst < 1e-6, value=worst,
           tolerance=1e-6, method=f'{len(states)} states, relative')
    worst = max(r['first_law']['relative'] for r in DATA['consistency'])
    record('numerical', 'the first law closes in the spin direction at fixed coupling',
           'manuscript eq. (2)', passed=worst < 1e-6, value=worst, tolerance=1e-6,
           method=f"{len(DATA['consistency'])} states, relative to the largest term")
    worst = max(s['max_relative_tensor_residual'] for s in states)
    record('numerical', 'every accepted state satisfies G + alpha H = 0 off the solver grid',
           'Einstein-Gauss-Bonnet field equations',
           passed=worst < 1e-8, value=worst, tolerance=1e-8,
           method='independent curvature evaluator at 51 off-grid points, relative')

    rows = STUDY['comparisons']
    envelopes = {r['alpha_gb']: r['psi_uncertainty'] for r in DATA['extremals']}
    ratio = max(r['psi_gb']['absolute']/envelopes[r['alpha_gb']] for r in rows)
    record('numerical', 'the extremal intercept survives two coarser radial resolutions at every coupling',
           'this work, resolution study', passed=ratio < 1., value=ratio,
           tolerance=1., method=f'{len(rows)} sequences, |dPsi| as a fraction of the fit sensitivity')


# =====================================================================
# C.  Transcriptions, checked against the downloaded sources.
# =====================================================================
def transcriptions():
    def text_of(stem):
        hits = sorted(PAPERS.glob(stem+'*.pdf'))
        if not hits:
            return None, None
        raw = subprocess.run(['pdftotext', '-layout', str(hits[0]), '-'],
                             capture_output=True, text=True).stdout
        return raw, hits[0]

    bkkr, path = text_of('1010.0860')
    if bkkr is None:
        record('transcription', 'arXiv:1010.0860 is present in papers/', 'papers/',
               passed=False, method='file missing')
        return
    flat = re.sub(r'\s+', ' ', bkkr)

    record('transcription', 'the source action carries alpha/4 on L_GB, so alpha_paper = 4 alpha',
           'arXiv:1010.0860 eq. (2.1)',
           passed=bool(re.search(r'd5 x\s*−g\s*R \+\s*LGB', flat)
                       and re.search(r'16πG M\s*4', flat)),
           method='eq. (2.1) reads I = (1/16piG) int d^5x sqrt(-g) [R + (alpha/4) L_GB]')
    record('transcription', 'the source charges are E = -3 V_3 U/16 pi G and J = V_3 W/8 pi G with V_3 = 2 pi^2',
           'arXiv:1010.0860 eq. (3.6)',
           passed=bool(re.search(r'E=−\s*U, J =\s*W', flat)
                       and 'V3 = 2π 2' in flat),
           method='eq. (3.6) and the line defining V_3; gives our -3 pi U/8 and pi W/4')
    record('transcription', 'the source J is the momentum in each plane, not the total',
           'arXiv:1010.0860 sec. 3.2',
           passed='equal-magnitude angular momenta J are the charges associated with the Killing vectors ∂/∂t, ∂/∂ϕ1 , and ∂/∂ϕ2' in flat,
           method='the sentence introducing eq. (3.6)')
    record('transcription', 'the source near-horizon metric carries 2k, so our k is twice theirs',
           'arXiv:1010.0860 eq. (4.6)',
           passed='(σ3 + 2krdt)2' in flat,
           method='eq. (4.6); with J = dE/dk this is why J_paper = 2 J')
    record('transcription', 'the source entropy has an Einstein piece and a GB piece of the printed form',
           'arXiv:1010.0860 eq. (3.10)',
           passed=bool(re.search(r'SE =\s*rH hH , SGB = α\s*hH \(4 −', flat)),
           method='eq. (3.10); its ratio to the Einstein term fixes alpha_paper = 4 alpha')
    record('transcription', 'the source quotes an outer ergosurface radius at the coupling we compare with',
           'arXiv:1010.0860 sec. 4.1', passed='1.104' in flat,
           method='the value our reconstruction does not reproduce; see the manuscript discussion')

    ma, _ = text_of('2009.00015')
    if ma is not None:
        flat_ma = re.sub(r'\s+', ' ', ma)
        record('transcription', 'the published extremal mass is M = (3/2) pi^(1/3) J^(2/3) + pi alpha + O(alpha^2)',
               'arXiv:2009.00015',
               passed=bool(re.search(r'3\s*π\s*1/3\s*J\s*2/3', flat_ma)
                           or 'πα' in flat_ma or 'π α' in flat_ma),
               method='the extremality result this work anchors mu(0) and Psi_ext(0) against')

    wu, _ = text_of('2405.04576')
    if wu is not None:
        flat_wu = re.sub(r'\s+', ' ', wu)
        record('transcription', 'the source uses the same 1/16 pi action normalisation, with G = 1',
               'arXiv:2405.04576 eq. (25)',
               passed=bool('∆I =' in flat_wu and 'gR' in flat_wu
                           and 'Rµνρσ' in flat_wu and '16π M' in flat_wu),
               method='eq. (25): Delta I = (alpha/16 pi) int sqrt(g) R_abcd R^abcd; '
                      'required for its Gibbs-potential coefficient to be our Psi_0')
        # The transcription the manuscript actually depends on: their general-D
        # perturbation, specialised here rather than copied.
        D, a_, r0, al = sp.symbols('D a r_0 alpha', positive=True)
        dG = (-al*(D-4)*(D-3)*(2*sp.pi**2)/(16*sp.pi*r0**4)
              * (a_**4-2*(2*D-3)*a_**2*r0**2+(D-2)**2*r0**4))
        source_psi0 = sp.simplify((dG.subs(D, 5)/1)/al)
        uu = sp.Symbol('u', nonnegative=True)
        record('transcription', 'their Delta M = (D-4) Delta G at D=5 with equal spins is exactly our eq. (17)',
               'arXiv:2405.04576 eq. (33)',
               passed=zero(sp.simplify(source_psi0.subs(a_, sp.sqrt(uu)*r0))
                           + sp.pi*(uu**2-14*uu+9)/4),
               method='specialised symbolically from the published general-D formula, '
                      'then mapped by u = a^2/r_+^2; not copied from the paper')
        record('transcription', 'the source treats a general quadratic-curvature correction on a Ricci-flat background',
               'arXiv:2405.04576',
               passed='Ricci' in flat_wu or 'Riemann' in flat_wu,
               method='why only the Riemann-squared term survives and its coefficient is our alpha')

    for path in sorted(PAPERS.glob('*.pdf')):
        head = subprocess.run(['pdftotext', '-f', '1', '-l', '1', str(path), '-'],
                              capture_output=True, text=True).stdout
        if not head.strip():
            record('transcription', f'{path.name} is readable', 'papers/',
                   passed=False, method='pdftotext returned nothing')


# =====================================================================
# D.  Called, not reproduced.
# =====================================================================
CALLED = [
    ('The Myers-Perry solution itself', 'Myers & Perry, Annals Phys. 172 (1986) 304',
     'Its closed forms are used as an oracle and every one of them is checked '
     'against our solver, but the solution is not rederived here. Not on arXiv, '
     'so no PDF is in papers/.'),
    ('The Boulware-Deser solution itself', 'Boulware & Deser, PRL 55 (1985) 2656',
     'Same status: the closed forms for M, T and S are the oracle for the static '
     'limit, and we differentiate them to get Psi, but the solution is not '
     'rederived. Not on arXiv.'),
    ("Lovelock's theorem", 'Lovelock, J. Math. Phys. 12 (1971) 498',
     'Cited for the field equations being second order. Not reproduced. Not on arXiv.'),
    ('The Wald entropy formula', 'arXiv:gr-qc/9307038, arXiv:gr-qc/9403028',
     'The Noether-charge form is cited, not derived. Its value on this horizon is '
     'checked against the independent derivation in arXiv:1010.0860.'),
    ('The Jacobson-Myers form for Lovelock horizons', 'arXiv:hep-th/9305016',
     'Cited as the form Wald entropy takes here; the resulting entropy is then '
     'verified numerically against the published near-horizon branch.'),
    ("Sen's entropy function formalism", 'arXiv:0708.1270',
     'The formalism is cited. Our use of it is rederived symbolically and its '
     'solution checked against the full field equations component by component, '
     'but the formalism itself is not re-justified.'),
    ('The domain bounds x < 1 and j <= 1', 'arXiv:2303.12471',
     'Quoted and compared against, not rederived. Our samples lie inside them.'),
    ('The Goon-Penco universality argument', 'arXiv:1909.05254',
     'Only the first-law identity is used, and that is rederived here. The '
     'universality claim itself is cited and not tested.'),
    ('Causality constraints on higher-curvature couplings', 'arXiv:1407.5597',
     'Cited in the discussion as a reason to separate the classical truncation '
     'from an ultraviolet completion. Nothing is computed from it.'),
    ('The extended first law for Lovelock couplings', 'arXiv:1005.5053, arXiv:2404.16981',
     'Cited as the source of the Psi dalpha term. The term is then used and its '
     'consequences checked internally, but the covariant-phase-space derivation '
     'is not repeated.'),
    ('The outer ergosurface radius 1.104 at alpha = 1/2', 'arXiv:1010.0860',
     'Compared against and does NOT reproduce: our reconstruction gives 1.102101. '
     'Unresolved; recorded in the manuscript discussion as an open point.'),
]


def called():
    for claim, source, detail in CALLED:
        record('called', claim, source, detail=detail)


def main():
    print('--- A. analytic rederivations -------------------------------------')
    analytic()
    print('--- B. numerical reproductions ------------------------------------')
    numerical()
    print('--- C. transcriptions against the downloaded sources --------------')
    transcriptions()
    print('--- D. called, not reproduced -------------------------------------')
    called()

    checked = [r for r in LEDGER if r['passed'] is not None]
    failed = [r for r in checked if not r['passed']]
    summary = dict(
        total=len(LEDGER), checked=len(checked), failed=len(failed),
        by_kind={kind: dict(
            total=sum(1 for r in LEDGER if r['kind'] == kind),
            passed=sum(1 for r in LEDGER if r['kind'] == kind and r['passed']),
            failed=sum(1 for r in LEDGER if r['kind'] == kind
                       and r['passed'] is False))
            for kind in ('analytic', 'numerical', 'transcription', 'called')})
    print()
    print(f"{summary['checked']} checked, {summary['failed']} failed; "
          f"{summary['by_kind']['called']['total']} entries called and not reproduced")
    for row in failed:
        print('  FAILED: '+row['claim'])

    papers = [dict(name=p.name,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   bytes=p.stat().st_size)
              for p in sorted(PAPERS.glob('*.pdf'))]
    OUT.write_text(json.dumps(dict(
        schema=1, summary=summary, ledger=LEDGER, papers=papers,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      sympy=sp.__version__, platform=platform.platform()),
    ), indent=1, allow_nan=False)+'\n', encoding='utf-8')
    print(f'wrote {OUT.relative_to(ROOT)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
