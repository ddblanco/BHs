"""Figures for the verification report: every external comparison, with the
residual it leaves drawn underneath the comparison itself.

Each panel pairs a closed form or a published curve with what this project's
solver produced, and each lower panel shows the deviation against the tolerance
the assertion in `work/verification_suite.py` actually used. Nothing is typed:
all of it is read from `results/`.

Run from the project root:  PYTHONPATH=src python3 verification-report/make_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = json.loads((ROOT/'results/egb-extremality.json').read_text(encoding='utf-8'))
LEDGER = json.loads((ROOT/'results/verification-ledger.json').read_text(encoding='utf-8'))
FIGURES = HERE/'figures'

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 9.5,
    'legend.fontsize': 8.2, 'xtick.labelsize': 8.2, 'ytick.labelsize': 8.2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
    'lines.linewidth': 1.15, 'pdf.fonttype': 42, 'figure.dpi': 200,
})
INK, ACCENT, MUTED, GOOD = '#1a1a1a', '#b03a1a', '#5b6b76', '#2f6f3e'


def style(ax):
    ax.tick_params(length=3, width=.7, colors=INK)
    ax.grid(alpha=.18, linewidth=.55)
    for spine in ax.spines.values():
        spine.set_color('#9aa7b0')


def tolerance_of(fragment):
    for row in LEDGER['ledger']:
        if fragment in row['claim'] and row['tolerance']:
            return row['tolerance']
    return None


def pair(height=(2.0, 1.0), size=(5.4, 3.5)):
    figure, axes = plt.subplots(2, 1, figsize=size, sharex=True,
                                gridspec_kw=dict(height_ratios=list(height),
                                                 hspace=.11))
    return figure, axes


def residual_axis(ax, tolerance, label='relative deviation', extra=()):
    """Log residual panel with the assertion tolerance drawn as a dashed line.

    One legend, built here, so that a caller adding its own series does not
    silently replace the line that gives the panel its meaning.
    """
    ax.set_yscale('log')
    handles = list(extra)
    if tolerance:
        ax.axhline(tolerance, color=ACCENT, linewidth=.9, linestyle='--')
        handles.append(plt.Line2D([], [], color=ACCENT, linewidth=.9,
                                  linestyle='--',
                                  label=f'assertion tolerance {tolerance:g}'))
    ax.set_ylabel(label)
    style(ax)
    # A clear band above the data, so the legend never sits on a marker or on
    # the tolerance line. The axis is logarithmic, so this costs only decades.
    low, high = ax.get_ylim()
    ax.set_ylim(low, max(high, (tolerance or high))*3e4)
    if handles:
        ax.legend(handles=handles, frameon=False, loc='upper right',
                  handlelength=1.6, fontsize=7.6, borderpad=.1,
                  ncol=len(handles), columnspacing=1.1)


def main():
    FIGURES.mkdir(exist_ok=True)

    # ---------------------------------------------- 1. Myers-Perry, alpha = 0
    rows = sorted(DATA['vacuum_masses'], key=lambda r: r['q'])
    figure, (top, low) = pair()
    grid = np.linspace(0., max(r['q'] for r in rows)*1.04, 300)
    top.plot(grid, 3*np.pi/(8*(1-grid**2)), '-', color=MUTED,
             label='Myers\u2013Perry, '+r'$M=3\pi r_H^2/[8(1-q^2)]$')
    top.plot([r['q'] for r in rows], [r['E'] for r in rows], 'o', markersize=4.2,
             markerfacecolor='white', markeredgewidth=1.1, color=INK,
             label='this solver, $\\alpha=0$')
    top.set_ylabel(r'$M$   (units $r_H=1$)')
    top.legend(frameon=False, loc='upper left', handlelength=1.7)
    style(top)
    low.plot([r['q'] for r in rows], [max(r['relative'], 1e-18) for r in rows],
             'o-', markersize=3.4, linewidth=.8, color=INK)
    residual_axis(low, tolerance_of('Myers-Perry closed form'))
    low.set_xlabel(r'$q=r_H\Omega_H$')
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-myers-perry.pdf')
    plt.close(figure)

    # ------------------------------------- 2. Boulware-Deser at finite alpha
    rows = sorted(DATA['static_limit'], key=lambda r: r['alpha_gb'])
    figure, (top, low) = pair()
    grid = np.linspace(0., max(r['alpha_gb'] for r in rows)*1.04, 300)
    top.plot(grid, .75*np.pi*(4*grid-3)/(1+4*grid), '-', color=MUTED,
             label='Boulware\u2013Deser, '+r'$\Psi=\frac{3\pi}{4}\frac{4\bar\alpha-3}{1+4\bar\alpha}$')
    top.plot([r['alpha_gb'] for r in rows], [r['psi_gb'] for r in rows], 'o',
             markersize=4.2, markerfacecolor='white', markeredgewidth=1.1,
             color=INK, label='this solver, zero spin')
    top.axhline(-9*np.pi/4, color=GOOD, linewidth=.8, linestyle=':',
                label=r'$-9\pi/4$, the $\alpha\to0$ limit')
    top.set_ylabel(r'$\Psi_{\rm static}$')
    top.legend(frameon=False, loc='lower right', handlelength=1.7)
    style(top)
    series = []
    for key, marker, colour, label in (
            ('mass_relative', 'o', INK, '$M$'),
            ('temperature_relative', 's', '#244a75', '$T$'),
            ('psi_relative', '^', GOOD, r'$\Psi$')):
        low.plot([r['alpha_gb'] for r in rows],
                 [max(r[key], 1e-18) for r in rows], marker, markersize=3.6,
                 markerfacecolor='none', markeredgewidth=1., color=colour)
        series.append(plt.Line2D([], [], marker=marker, linestyle='none',
                                 markersize=3.6, markerfacecolor='none',
                                 markeredgewidth=1., color=colour, label=label))
    residual_axis(low, tolerance_of('static potential reproduces'), extra=series)
    low.set_xlabel(r'$\bar\alpha=\alpha/r_H^2$')
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-boulware-deser.pdf')
    plt.close(figure)

    # ------------------------------ 3. the published perturbative potential
    rows = sorted(DATA['perturbative'], key=lambda r: r['q'])
    figure, (top, low) = pair()
    grid = np.linspace(0., max(r['q'] for r in rows)*1.02, 400)
    u = grid**2/(1-grid**2)
    top.plot(grid, -np.pi*(u**2-14*u+9)/4, '-', color=MUTED,
             label='Wu–Lü, '+r'$\Psi_0=-\frac{\pi}{4}(u^2-14u+9)$')
    top.plot([r['q'] for r in rows], [r['measured'] for r in rows], 'o',
             markersize=4.2, markerfacecolor='white', markeredgewidth=1.1,
             color=INK, label='this solver, implicit response at $\\alpha=0$')
    top.axhline(0., color=MUTED, linewidth=.7)
    top.plot([1/math.sqrt(2)], [np.pi], '*', markersize=11, color=GOOD,
             clip_on=False, zorder=6,
             label=r'$q=1/\sqrt{2}$, $\Psi_0=\pi$ (extremal)')
    top.set_ylabel(r'$\Psi_0$')
    top.legend(frameon=False, loc='upper left', handlelength=1.7)
    style(top)
    low.plot([r['q'] for r in rows],
             [max(r['relative_deviation'], 1e-18) for r in rows], 'o-',
             markersize=3.4, linewidth=.8, color=INK)
    residual_axis(low, tolerance_of('published perturbative potential'))
    low.set_xlabel(r'$q=r_H\Omega_H$')
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-perturbative.pdf')
    plt.close(figure)

    # ------------------------------------ 4. the published near-horizon branch
    rows = sorted(DATA['near_horizon'], key=lambda r: r['y'])
    figure, (top, low) = pair()
    top.plot([r['y'] for r in rows], [r['near_horizon'] for r in rows], '-',
             color=MUTED, label='near-horizon branch, rederived from [1] sec. 4.2')
    top.plot([r['y'] for r in rows], [r['sigma'] for r in rows], 'o',
             markersize=4.2, markerfacecolor='white', markeredgewidth=1.1,
             color=INK, label=r'bulk solutions extrapolated to $T=0$')
    top.plot([0.], [2*np.pi], '*', markersize=11, color=GOOD, clip_on=False,
             zorder=6, label='$2\\pi$, extremal Myers\u2013Perry')
    top.set_ylabel(r'$\sigma=S_\mathrm{ext}/J$')
    top.legend(frameon=False, loc='upper left', handlelength=1.7)
    style(top)
    low.plot([r['y'] for r in rows],
             [max(r['relative_difference'], 1e-18) for r in rows], 'o-',
             markersize=3.4, linewidth=.8, color=INK)
    residual_axis(low, tolerance_of('near-horizon curve at finite coupling'))
    low.set_xlabel(r'$y=\alpha/J^{2/3}$')
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-near-horizon.pdf')
    plt.close(figure)

    for name in sorted(FIGURES.glob('*.pdf')):
        print(f'wrote {name.relative_to(ROOT)}  {name.stat().st_size} bytes')


if __name__ == '__main__':
    main()
