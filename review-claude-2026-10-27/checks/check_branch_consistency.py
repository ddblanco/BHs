"""Cross-check of three independent determinations of the extremal mass function:
  (A) manuscript T->0 extrapolation      results/egb-extremality.json  (mu(y), Psi_ext(y) = mu'(y))
  (B) direct T=0 spectral solver         work/kkr-extremal/analysis.json (mu, mu_prime, omega)
  (C) extremal-mass: exact weak series   extremal-mass/series/coefficients.json (d_1..d_6)
      and the global approximant         extremal-mass/strong/global2.json (True_10)
      and the strong-coupling numerics   extremal-mass/strong/numerics.tex table (s >= 2)
Conversions (derived in check_thermo.py):  s = pi^(2/3) y,  m = mu/pi^(1/3),  Psi_ext = pi m'(s),
and 3 omega = mu - y mu' (Smarr at T = 0).
Writes a receipt to review-claude-2026-10-27/checks/branch_consistency.json.
"""
import json, re
from pathlib import Path
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P13, P23 = np.pi**(1/3), np.pi**(2/3)
out = {}

# ---- (A) manuscript extremal table
ext = json.loads((ROOT/'results/egb-extremality.json').read_text())
rows = ext['extremals']
print('manuscript extremal rows keys:', sorted(rows[0].keys()))
A = [(r['alpha_gb'] if 'alpha_gb' in r else r.get('alpha'), r['y'], r['mu'], r['psi_gb']['value']) for r in rows]
A.sort(key=lambda z: z[1])

# ---- (C) exact weak series and the approximant
coef = json.loads((ROOT/'extremal-mass/series/coefficients.json').read_text())
d = {int(k): float(v['value']) for k, v in coef.items()}
glob = json.loads((ROOT/'extremal-mass/strong/global2.json').read_text())['True_10']
kappa, c = glob['kappa'], np.array(glob['c'])


def m_series(s_, nmax=6):
    return 1.5 + sum(d[n]*s_**n for n in range(1, nmax + 1))


def m_glob(s_):
    tt = (1 + s_/kappa)**-0.5
    return 0.75*s_ + sum(c[k-1]*tt**k for k in range(1, 11))


def dm_glob(s_):
    tt = (1 + s_/kappa)**-0.5
    dt = -0.5*(1 + s_/kappa)**-1.5/kappa
    return 0.75 + sum(k*c[k-1]*tt**(k-1) for k in range(1, 11))*dt


# approximant constraints
print(f'approximant: m(0) = {m_glob(0):.12f} (3/2),  m\'(0) = {dm_glob(0):.12f} (1),  c1 sqrt(kappa) = {c[0]*np.sqrt(kappa):.9f} (4/3), c2 kappa = {c[1]*kappa:.9f}')
out['approximant_constraints'] = dict(m0=m_glob(0), dm0=dm_glob(0), a1=c[0]*np.sqrt(kappa), a2=c[1]*kappa)
assert abs(m_glob(0) - 1.5) < 1e-9 and abs(dm_glob(0) - 1) < 1e-9 and abs(c[0]*np.sqrt(kappa) - 4/3) < 1e-9

# ---- (B) spectral branch (direct T = 0)
spec = json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch']
spec = sorted(spec, key=lambda r: r['y'])
ys = np.array([r['y'] for r in spec]); mus = np.array([r['mu'] for r in spec]); mps = np.array([r['mu_prime'] for r in spec])
oms = np.array([r['omega'] for r in spec])


from scipy.interpolate import CubicSpline
def interp(xv, yv, x):
    cs_ = CubicSpline(xv, yv)
    return float(cs_(x))


print('\n alpha     y        s      mu_ms      m_ms      m_series6   m_glob    |dm_ser|   |dm_glob|  Psi_ms   pi m\'_glob  dPsi     mu_spec    Psi_spec')
tab = []
for alpha, yv, muv, psiv in A:
    sv = P23*yv
    mm = muv/P13
    ms6 = m_series(sv) if sv < 0.0551 else float('nan')
    mg = m_glob(sv)
    psig = np.pi*dm_glob(sv)
    musp = interp(ys, mus, yv) if ys.min() <= yv <= ys.max() else float('nan')
    psisp = interp(ys, mps, yv) if ys.min() <= yv <= ys.max() else float('nan')
    print(f'{alpha:6.3f} {yv:8.5f} {sv:8.5f} {muv:10.6f} {mm:10.7f} {ms6:11.7f} {mg:10.7f} {abs(mm-ms6):9.1e} {abs(mm-mg):9.1e} {psiv:8.5f} {psig:9.5f} {psiv-psig:8.1e} {musp:10.6f} {psisp:9.5f}')
    tab.append(dict(alpha=alpha, y=yv, s=sv, mu=muv, m=mm, m_series6=ms6, m_glob=mg, psi=psiv, psi_glob=psig, mu_spec=musp, psi_spec=psisp))
out['rows'] = tab
# worst deviations
rel_glob = max(abs(r['m'] - r['m_glob'])/r['m'] for r in tab)
rel_ser = max(abs(r['m'] - r['m_series6'])/r['m'] for r in tab if r['s'] < 0.0551)
dpsi = max(abs(r['psi'] - r['psi_glob']) for r in tab)
dmu_spec = max(abs(r['mu'] - r['mu_spec']) for r in tab if r['mu_spec'] == r['mu_spec'] and r['y'] > 0)
dpsi_spec = max(abs(r['psi'] - r['psi_spec']) for r in tab if r['psi_spec'] == r['psi_spec'] and r['y'] > 0)
print(f'\nworst |m_manuscript - m_series6|/m for s < 0.055 : {rel_ser:.2e}')
print(f'worst |m_manuscript - m_approximant|/m        : {rel_glob:.2e}')
print(f'worst |Psi_manuscript - pi m\'_approximant|     : {dpsi:.2e}')
print(f'worst |mu_manuscript - mu_spectral| (interp)   : {dmu_spec:.2e}')
print(f'worst |Psi_manuscript - Psi_spectral| (interp) : {dpsi_spec:.2e}')
out['worst'] = dict(series6=rel_ser, approximant=rel_glob, psi_vs_approximant=dpsi, mu_vs_spectral=dmu_spec, psi_vs_spectral=dpsi_spec)

# ---- minimum of the slope from the approximant and from the spectral branch
sg = np.linspace(0.05, 3, 200001)
dmg = dm_glob(sg)
i = np.argmin(dmg)
print(f'\napproximant: min of Psi = pi m\'(s) = {np.pi*dmg[i]:.5f} at s = {sg[i]:.5f}, y = {sg[i]/P23:.5f}')
j = np.argmin(mps)
print(f'spectral branch: min mu_prime = {mps[j]:.5f} at y = {ys[j]:.5f}  (grid point)')
# local quadratic through the three lowest spectral points
k = slice(max(j-3, 0), j+4)
pp = np.polyfit(ys[k], mps[k], 2)
ymin = -pp[1]/(2*pp[0])
print(f'spectral branch: quadratic fit minimum {np.polyval(pp, ymin):.5f} at y = {ymin:.5f}')
print(f'manuscript: lowest sampled Psi_ext = {min(r["psi"] for r in tab):.5f} at y = {[r["y"] for r in tab if r["psi"]==min(rr["psi"] for rr in tab)][0]:.4f}')
out['minimum'] = dict(approximant=dict(psi=float(np.pi*dmg[i]), s=float(sg[i]), y=float(sg[i]/P23)),
                      spectral_grid=dict(psi=float(mps[j]), y=float(ys[j])), spectral_quadratic=dict(psi=float(np.polyval(pp, ymin)), y=float(ymin)))

# ---- Smarr identity on the spectral branch: 3 omega = mu - y mu'
res = np.abs(3*oms - (mus - ys*mps))
print(f'\nspectral branch: worst |3 omega - (mu - y mu\')| = {res.max():.2e} at y = {ys[np.argmax(res)]:.4f}; median {np.median(res):.2e}; worst for y < 1: {res[ys < 1].max():.2e}')
jo = np.argmax(oms)
print(f'spectral branch: max omega = {oms[jo]:.5f} at y = {ys[jo]:.5f}   (manuscript: 0.762 at y = 0.175; identity says max omega <-> min mu\')')
out['omega'] = dict(worst_identity=float(res.max()), omega_max=float(oms[jo]), y_at_max=float(ys[jo]))

# ---- the strong-coupling table of extremal-mass/strong/numerics.tex vs the approximant and the spectral branch
txt = (ROOT/'extremal-mass/strong/numerics.tex').read_text()
pts = re.findall(r'^(\d+\.?\d*) & (\d+\.\d+) & ', txt, re.M)
print('\n s        m_table       m_approx     rel err    m_spectral(interp)')
worst = 0
for sv, mv in pts:
    sv, mv = float(sv), float(mv)
    mg = m_glob(sv)
    yv = sv/P23
    msp = interp(ys, mus, yv)/P13 if yv <= ys.max() else float('nan')
    worst = max(worst, abs(mg-mv)/mv)
    print(f'{sv:8.4f} {mv:13.8f} {mg:13.8f} {abs(mg-mv)/mv:9.1e} {msp:12.7f}')
print(f'worst relative error of the approximant on the printed table: {worst:.2e}')
out['table_vs_approximant_worst'] = worst

# ---- weak series convergence illustration: partial sums at the first three manuscript couplings
print('\npartial sums of the weak series at the three smallest couplings:')
for r_ in tab[1:4]:
    sv = r_['s']
    ps = [1.5 + sum(d[n]*sv**n for n in range(1, N+1)) for N in range(1, 7)]
    print(f'  s = {sv:.5f}: ' + ' '.join(f'{p:.8f}' for p in ps) + f'   manuscript m = {r_["m"]:.8f}')

# ---- large-order behaviour of d_n versus the throat branch point s*
sst = mp.mpc('0.0059231771', '0.0548188529')
print('\n n   d_n          sign d_n  sign Re s*^-n   |d_n| |s*|^n')
for n_ in range(1, 7):
    v = d[n_]
    print(f' {n_}  {v:12.6f}    {"+" if v>0 else "-"}        {"+" if mp.re(sst**-n_)>0 else "-"}       {float(abs(v)*abs(sst)**n_):.4f}')
(HERE/'branch_consistency.json').write_text(json.dumps(out, indent=1))
print('\nwrote', HERE/'branch_consistency.json')
