"""Figures for the review report: the three independent determinations of the extremal mass
function against the global approximant, the slope near its minimum, and the weak series.
Reads only recorded project outputs.  Writes review-claude-2026-10-27/report/figures/*.pdf."""
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT/'review-claude-2026-10-27/report/figures'; OUT.mkdir(parents=True, exist_ok=True)
P13, P23 = np.pi**(1/3), np.pi**(2/3)
C = dict(ms='#2a78d6', spec='#eb6834', strong='#1baf7a', approx='#0b0b0b', series='#6250d6', text='#52514e')
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.spines.top': False, 'axes.spines.right': False, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'legend.frameon': False})

ext = json.loads((ROOT/'results/egb-extremality.json').read_text())['extremals']
ms = sorted([(r['y'], r['mu'], r['psi_gb']['value'], r['psi_uncertainty']) for r in ext])
y_ms = np.array([r[0] for r in ms]); mu_ms = np.array([r[1] for r in ms]); psi_ms = np.array([r[2] for r in ms]); dpsi_ms = np.array([r[3] for r in ms])
spec = sorted(json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch'], key=lambda r: r['y'])
y_sp = np.array([r['y'] for r in spec]); mu_sp = np.array([r['mu'] for r in spec]); psi_sp = np.array([r['mu_prime'] for r in spec])
strong = [(float(a), float(b)) for a, b in re.findall(r'^(\d+\.?\d*) & (\d+\.\d+) & ', (ROOT/'extremal-mass/strong/numerics.tex').read_text(), re.M)]
s_st = np.array([p[0] for p in strong]); m_st = np.array([p[1] for p in strong])
glob = json.loads((ROOT/'extremal-mass/strong/global2.json').read_text())['True_10']; kappa, c = glob['kappa'], np.array(glob['c'])
d = {int(k): float(v['value']) for k, v in json.loads((ROOT/'extremal-mass/series/coefficients.json').read_text()).items()}


def m_glob(s):
    t = (1 + s/kappa)**-0.5
    return 0.75*s + sum(c[k-1]*t**k for k in range(1, 11))


def dm_glob(s):
    t = (1 + s/kappa)**-0.5; dt = -0.5*(1 + s/kappa)**-1.5/kappa
    return 0.75 + sum(k*c[k-1]*t**(k-1) for k in range(1, 11))*dt


# ---------------- figure 1: m(s) and residuals
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.3], hspace=0.08))
sg = np.linspace(0, 3.2, 600)
a1.plot(sg, m_glob(sg), color=C['approx'], lw=1.2, label='aproximante global de 10 términos (extremal-mass)')
a1.plot(P23*y_sp, mu_sp/P13, color=C['spec'], lw=2, alpha=0.9, label='solver directo a $T=0$ (work/kkr-extremal), 268 puntos')
a1.plot(P23*y_ms, mu_ms/P13, 'o', color=C['ms'], ms=5, mfc='white', mew=1.4, label='manuscrito, extrapolación $T\\to0$ (13 acoplamientos)')
a1.plot(s_st, m_st, 's', color=C['strong'], ms=4.5, mfc='white', mew=1.4, label='shooting / multiple shooting (extremal-mass/strong)')
a1.set_xlim(0, 3.2); a1.set_ylim(1.4, 3.3); a1.set_ylabel('$m(s)=M_{\\rm ext}/(\\pi^{1/3}J^{2/3})$'); a1.grid(True, axis='y')
a1.legend(loc='upper left', fontsize=7.5)
a1.text(0.05, 1.46, 'Myers–Perry: $m(0)=3/2$', color=C['text'], fontsize=7.5)
res_ms = (mu_ms/P13 - m_glob(P23*y_ms))/(mu_ms/P13)
res_sp = (mu_sp/P13 - m_glob(P23*y_sp))/(mu_sp/P13)
res_st = (m_st - m_glob(s_st))/m_st
a2.axhline(0, color=C['approx'], lw=0.8)
a2.plot(P23*y_sp, 1e6*res_sp, color=C['spec'], lw=1.2)
a2.plot(P23*y_ms, 1e6*res_ms, 'o', color=C['ms'], ms=5, mfc='white', mew=1.4)
a2.plot(s_st, 1e6*res_st, 's', color=C['strong'], ms=4.5, mfc='white', mew=1.4)
a2.set_ylim(-14, 14); a2.set_ylabel('residuo relativo\nal aproximante $[10^{-6}]$'); a2.set_xlabel('$s=\\pi^{2/3}\\alpha/J^{2/3}$'); a2.grid(True, axis='y')
fig.savefig(OUT/'fig-three-routes.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- figure 2: the slope Psi_ext = mu'(y) = pi m'(s)
fig, ax = plt.subplots(figsize=(5.6, 3.3))
yg = np.linspace(0, 1.3, 800)
ax.plot(yg, np.pi*dm_glob(P23*yg), color=C['approx'], lw=1.2, label='$\\pi\\,m\'(s)$ del aproximante global')
ax.plot(y_sp, psi_sp, color=C['spec'], lw=2, alpha=0.9, label='solver directo a $T=0$: $\\mu\'(y)$ por diferenciación implícita')
ax.errorbar(y_ms, psi_ms, yerr=dpsi_ms, fmt='o', color=C['ms'], ms=5, mfc='white', mew=1.4, capsize=2, lw=0.9, label='manuscrito: $\\Psi_{\\rm ext}$ (barra = envolvente de ajustes)')
ax.axhline(np.pi, color=C['text'], lw=0.7, ls=':'); ax.text(0.02, np.pi+0.04, '$\\pi$ (Ma–Li–Lü, orden 1)', color=C['text'], fontsize=7.5)
ax.axhline(3*np.pi/4, color=C['text'], lw=0.7, ls=':'); ax.text(0.02, 3*np.pi/4-0.12, '$3\\pi/4$ (asíntota $s\\to\\infty$)', color=C['text'], fontsize=7.5)
ax.set_xlim(0, 1.3); ax.set_ylim(0.9, 3.4); ax.set_xlabel('$y=\\alpha/J^{2/3}$'); ax.set_ylabel('$\\partial M_{\\rm ext}/\\partial\\alpha|_J$'); ax.grid(True, axis='y')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.80), fontsize=7.5)
# inset: the minimum
ins = ax.inset_axes([0.42, 0.14, 0.33, 0.36])
ins.plot(yg, np.pi*dm_glob(P23*yg), color=C['approx'], lw=1.0)
ins.plot(y_sp, psi_sp, color=C['spec'], lw=1.6)
ins.errorbar(y_ms, psi_ms, yerr=dpsi_ms, fmt='o', color=C['ms'], ms=4, mfc='white', mew=1.2, capsize=1.5, lw=0.8)
ins.set_xlim(0.1, 0.3); ins.set_ylim(1.05, 1.13); ins.tick_params(labelsize=6.5); ins.set_title('mínimo', fontsize=7, color=C['text'])
fig.savefig(OUT/'fig-slope.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- figure 3: weak series partial sums and Pade against the numerics
fig, ax = plt.subplots(figsize=(5.6, 3.2))
sg = np.linspace(0, 0.32, 400)
cs_sp = CubicSpline(P23*y_sp, mu_sp/P13)
ax.plot(sg, cs_sp(sg) - 1.5 - sg, color=C['spec'], lw=2, label='numérico (solver directo)')
ax.plot(P23*y_ms, mu_ms/P13 - 1.5 - P23*y_ms, 'o', color=C['ms'], ms=5, mfc='white', mew=1.4, label='numérico (manuscrito)')
for N, ls in ((2, (0, (1, 1))), (4, (0, (3, 1))), (6, '-')):
    ax.plot(sg, sum(d[n]*sg**n for n in range(2, N+1)), color=C['series'], lw=1.0, ls=ls, label=f'serie débil truncada en $s^{N}$')
mp.mp.dps = 30
pn, pd_ = mp.pade([mp.mpf('1.5')] + [mp.mpf(d[n]) for n in range(1, 7)], 3, 3)
pade = np.array([float(mp.polyval(pn[::-1], x)/mp.polyval(pd_[::-1], x)) for x in sg])
ax.plot(sg, pade - 1.5 - sg, color=C['strong'], lw=1.0, ls='--', label='Padé [3/3] de la serie')
ax.axvline(0.05514, color=C['text'], lw=0.7, ls=':'); ax.text(0.057, -0.075, '$|s_*|=0.055$', color=C['text'], fontsize=7.5)
ax.set_xlim(0, 0.32); ax.set_ylim(-0.08, 0.01); ax.set_xlabel('$s$'); ax.set_ylabel('$m(s)-3/2-s$'); ax.grid(True, axis='y')
ax.legend(loc='lower left', fontsize=7)
fig.savefig(OUT/'fig-weak-series.pdf', bbox_inches='tight'); plt.close(fig)
print('wrote', sorted(p.name for p in OUT.glob('*.pdf')))
