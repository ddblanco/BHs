"""Build the manuscript's figures, tables and numbers from the measurement.

Single rule, inherited from the rest of the project: **no number that appears in
the paper is typed into the paper.** Everything is read here from
`../results/egb-extremality.json` and emitted as

  * `figures/fig-profiles.pdf`    -- the four metric functions at every
                                     coupling, from `../results/egb-rotating-profiles.json`,
  * `figures/fig-potential.pdf`   -- Psi across the family,
  * `figures/fig-shift.pdf`       -- the extremality shift, both routes,
  * `figures/fig-entropy.pdf`     -- the extremal entropy against [1, sec. 4.2],
  * `figures/fig-extrapolation.pdf` -- the T->0 extrapolation itself: the fitted
                                     points, the production cubic, its residuals
                                     and the competing intercepts,
  * `figures/fig-resolution.pdf`  -- the radial-resolution study, from
                                     `../results/egb-rotating-resolution-study.json`,
  * `numbers.tex`                 -- one LaTeX macro per quantity quoted in the
                                     prose, plus the two data tables,
  * `supplementary/solutions.csv` -- every accepted solution, so that the
                                     figures are reproducible from the paper,

so `main.tex` contains prose and `\\Nsomething` macros and nothing else
numerical. Rerunning the measurement and rebuilding propagates every change;
nothing can silently go stale.

Run from this directory:  ../.venv/Scripts/python make_figures.py
"""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reanalysis import (leave_one_out, matched_window, production_fit,
                        sensitivity, smarr_potential)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'results/egb-extremality.json'
PROFILES = HERE.parent/'results/egb-rotating-profiles.json'
RESOLUTION_STUDY = HERE.parent/'results/egb-rotating-resolution-study.json'
ONSHELL = HERE.parent/'results/egb-onshell-potential.json'
FIGURES = HERE/'figures'
NUMBERS = HERE/'numbers.tex'

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 9.5,
    'legend.fontsize': 8.2,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': .7,
    'lines.linewidth': 1.15,
    'pdf.fonttype': 42,
    'figure.dpi': 200,
})
INK, ACCENT, MUTED = '#1a1a1a', '#b03a1a', '#5b6b76'


def latex_float(value, digits=2):
    """A float as LaTeX maths: 1.19e-07 -> 1.19\\times 10^{-7}."""
    # An exact zero prints as `0`: a mantissa would suggest a measured
    # smallness that an identically vanishing spread does not have.
    if value == 0.:
        return '0'
    text = f'{value:.{digits}e}'
    mantissa, exponent = text.split('e')
    power = int(exponent)
    if power == 0:
        return mantissa
    return rf'{mantissa}\times 10^{{{power}}}'


def fixed(value, digits):
    return f'{value:.{digits}f}'


def style(ax):
    ax.tick_params(length=3, width=.7, colors=INK)
    ax.grid(alpha=.18, linewidth=.55)
    for spine in ax.spines.values():
        spine.set_color('#9aa7b0')


def main():
    FIGURES.mkdir(exist_ok=True)
    data = json.loads(DATA.read_text(encoding='utf-8'))
    if not PROFILES.exists():
        raise SystemExit(f'missing {PROFILES.name}; run '
                         'experiments/egb_rotating_profile_atlas.py first')
    atlas = json.loads(PROFILES.read_text(encoding='utf-8'))
    if not RESOLUTION_STUDY.exists():
        raise SystemExit(f'missing {RESOLUTION_STUDY.name}; run '
                         'experiments/egb_rotating_resolution_study.py first')
    study = json.loads(RESOLUTION_STUDY.read_text(encoding='utf-8'))
    if not ONSHELL.exists():
        raise SystemExit(f'missing {ONSHELL.name}; run '
                         'experiments/egb_onshell_potential.py first')
    onshell = json.loads(ONSHELL.read_text(encoding='utf-8'))
    walks = {float(k): v for k, v in data['walks'].items()}
    extremals = sorted(data['extremals'], key=lambda r: r['y'])
    fit_audit = []
    for record in extremals:
        group = walks[record['alpha_gb']]
        fitted = sensitivity(group, record['psi_gb']['value'])
        record['psi_uncertainty'] = max(record['psi_gb']['spread'],
                                         fitted['envelope'])
        cold = sorted(group, key=lambda s: s['T_H'])[:12]
        # What the production fit actually did, and what it depends on: the
        # referee asked for the points, the window, the residuals, the
        # competing intercepts and the effect of dropping a fitted point.
        fitted.update(alpha_gb=record['alpha_gb'],
                      envelope=record['psi_uncertainty'],
                      temperature_min=cold[0]['T_H'],
                      temperature_range=cold[-1]['T_H']/cold[0]['T_H'],
                      production=production_fit(group),
                      leave_one_out=leave_one_out(
                          group, record['psi_gb']['value']))
        fit_audit.append(fitted)
    # ------------------------------------------- the radial-resolution study
    # One entry per (coupling, coarse resolution). Everything here is read from
    # the study artifact except the matched-window refit, which is done on its
    # saved walks so that the manuscript can separate the resolution from the
    # distance the two fits had to travel.
    study_walks = {key: value for key, value in study['walks'].items()}
    resolution_audit = []
    for row in study['comparisons']:
        coupling, coarse = row['alpha_gb'], row['resolution']
        matched = matched_window(walks[coupling],
                                 study_walks[f"{coarse}|{coupling:.12g}"])
        envelope = next(r['psi_uncertainty'] for r in extremals
                        if r['alpha_gb'] == coupling)
        resolution_audit.append(dict(
            row, matched=matched, envelope=envelope,
            below_envelope=bool(row['psi_gb']['absolute'] < envelope),
            escalated_on_both=bool(
                str(data['refined_resolution']) in row['fine_census']
                and str(data['refined_resolution']) in row['coarse_census'])))

    anchor = data['extremal_anchor']
    curve = data['mass_curve']
    routes = data['route_comparison']
    horizon = sorted(data['near_horizon'], key=lambda r: r['y'])
    perturbative = data['perturbative']
    refinement = data['step_refinement']
    bounds = data['published_bounds']
    contrast = data['resolution_contrast']
    locus = [r for r in data['sign_locus'] if r['bracketed'] and r['fully_covered']]
    locus.sort(key=lambda r: r['alpha_gb'])
    static = data['static_limit']
    tails = data['tail_diagnostics']
    consistency = data['consistency']
    models = fit_audit
    scaled = {r['alpha_gb']: r for r in bounds['extremal']}

    states = [s for group in walks.values() for s in group]
    atlas_rows = sorted(atlas['rows'], key=lambda r: r['alpha_gb'])
    wide = {k: v for k, v in sorted(walks.items()) if len(v) >= 25}
    lowest = min(extremals, key=lambda r: r['psi_gb']['value'])

    # ------------------------------------ the extremal horizon angular velocity
    # The discussion reads the slope through omega = Omega_H J^(1/3), which the
    # Smarr relation at T=0 ties to the mass and the slope by 3 omega = mu - y
    # mu'. Extrapolating omega on its own, with the production model, turns that
    # tie into something testable instead of a rearrangement: the fitted omega
    # never enters the right-hand side.
    for group in walks.values():
        for state in group:
            state['omega_scaled'] = state['omega_h']*state['J']**(1/3)
    omega_rows = []
    for record in extremals:
        fitted = production_fit(walks[record['alpha_gb']], 'omega_scaled')
        implied = (record['mu']-record['y']*record['psi_gb']['value'])/3
        omega_rows.append(dict(y=record['y'], alpha_gb=record['alpha_gb'],
                               omega=fitted['intercept'], implied=implied,
                               difference=abs(fitted['intercept']-implied)))
    omega_peak = max(omega_rows, key=lambda r: r['omega'])

    # --------------------------------------------------- figure 1: the profiles
    # One colour family per metric function, one shade per coupling, so that the
    # reader can see both which function a curve is and which coupling it came
    # from. The couplings are the rows of the extremal table; the spin is the one
    # they share.
    # b and f are close enough over most of the range to hide one another, so b
    # is drawn as -b, the convention of fig. 1 of arXiv:1010.0860, which also
    # makes the two figures directly comparable.
    families = [('h_over_r2', 'Greens', r'$h/r^2$', 1.), ('f', 'Oranges', r'$f$', 1.),
                ('w', 'Purples', r'$w$', 1.), ('b', 'Blues', r'$-b$', -1.)]
    figure, ax = plt.subplots(figsize=(5.4, 3.6))
    shades = np.linspace(.28, .95, len(atlas_rows))
    for field, cmap, _, sign in families:
        palette = plt.get_cmap(cmap)(shades)
        for colour, row in zip(palette, atlas_rows):
            ax.plot(row['profile']['r'],
                    [sign*value for value in row['profile'][field]],
                    color=colour, linewidth=.95, solid_joinstyle='round')
    ax.axhline(0., color=MUTED, linewidth=.7, zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(1., 10.)
    ax.set_xticks([1., 1.5, 2., 3., 5., 7., 10.])
    ax.set_xticklabels(['1', '1.5', '2', '3', '5', '7', '10'])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r'$r$   (units $r_H=1$)')
    ax.set_ylabel('metric functions')
    style(ax)
    # Two legends: one naming the functions, one giving the shade scale.
    function_handles = [plt.Line2D([], [], color=plt.get_cmap(cmap)(.72),
                                   linewidth=1.5, label=label)
                        for _, cmap, label, _ in families]
    first = ax.legend(handles=function_handles, frameon=False, ncol=4,
                      loc='upper left', handlelength=1.3, columnspacing=1.1)
    ax.add_artist(first)
    # A discrete colour bar rather than two labelled ends: the shades are one
    # per coupling and the couplings are not evenly spaced, so the bar carries
    # one cell per row of the extremal table and every cell is labelled.
    scale = matplotlib.colors.ListedColormap(plt.get_cmap('Greys')(shades))
    cells = matplotlib.colors.BoundaryNorm(np.arange(len(atlas_rows)+1)-.5,
                                           len(atlas_rows))
    # Between w, which has settled to zero, and -b, which has settled to -1:
    # the one band of the plot no curve crosses.
    bar_axes = ax.inset_axes([.495, .295, .455, .034])
    bar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=cells, cmap=scale),
                          cax=bar_axes, orientation='horizontal',
                          ticks=np.arange(len(atlas_rows)))
    bar.outline.set_linewidth(.5)
    bar.outline.set_edgecolor('#9aa7b0')
    # Every cell is a coupling; labelling all thirteen would collide, so the
    # ticks are drawn for all and the text is written on alternate cells.
    bar.set_ticklabels([rf"${row['alpha_gb']:g}$" if index % 2 == 0 else ''
                        for index, row in enumerate(atlas_rows)])
    bar.ax.tick_params(length=2, width=.5, labelsize=6.4, pad=1.5)
    bar.set_label(rf"$\alpha$   ({len(atlas_rows)} couplings, "
                  rf"all at $q={atlas['start_spin']:g}$)", size=7.4, labelpad=2.5)
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-profiles.pdf')
    plt.close(figure)

    # ---------------------------------------------------------------- figure 2
    figure, ax = plt.subplots(figsize=(5.4, 3.3))
    colours = plt.cm.viridis(np.linspace(.05, .82, len(wide)))
    for colour, (coupling, group) in zip(colours, wide.items()):
        ordered = sorted(group, key=lambda s: s['t_scaled'])
        # Markers as well as the line: the line joins discrete solutions and
        # the sampling density is part of what the figure has to show.
        ax.plot([s['t_scaled'] for s in ordered], [s['psi_gb'] for s in ordered],
                color=colour, marker='o', markersize=1.9, markeredgewidth=0.,
                label=rf'$\alpha={coupling:g}$')
    for record in extremals:
        if record['alpha_gb'] not in wide:
            continue
        ax.plot([0.], [record['psi_gb']['value']], 'o', markersize=3.1,
                color=ACCENT, zorder=5, clip_on=False)
    exact = [(r['exact']['t_scaled'], r['exact']['psi_gb']) for r in
             sorted(data['static_limit'], key=lambda r: r['alpha_gb'])
             if r['alpha_gb'] in wide]
    ax.plot([t for t, _ in exact], [p for _, p in exact], 'x', markersize=5,
            markeredgewidth=1.1, color='#2f6f3e', zorder=6,
            label='static, closed form')
    ax.axhline(0., color=MUTED, linewidth=.7, zorder=1)
    ax.set_xlabel(r'$t_H$')
    ax.set_ylabel(r'$\Psi$')
    ax.set_xlim(-.025, 1.01)
    style(ax)
    ax.legend(frameon=False, ncol=2, loc='lower left', handlelength=1.4)
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-potential.pdf')
    plt.close(figure)

    # ---------------------------------------------------------------- figure 3
    figure, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.errorbar([r['y'] for r in extremals],
                [r['psi_gb']['value'] for r in extremals],
                yerr=[r['psi_uncertainty'] for r in extremals],
                fmt='o', markersize=3.6, capsize=2., linewidth=.9, color=INK,
                label=r'$\lim_{T\to0}\Psi$  (thermodynamic)')
    ax.plot([r['y'] for r in routes['intervals']],
            [r['mass_only'] for r in routes['intervals']], 's', markersize=3.2,
            markerfacecolor='none', markeredgewidth=.9, color=ACCENT,
            label=r'$\Delta\mu/\Delta y$  (mass-curve secants)')
    ax.plot([0.], [curve['published_shift']], '*', markersize=10,
            color='#2f6f3e', zorder=6, clip_on=False,
            label=r'$\pi$, published linear order')
    ax.set_xlabel(r'$y=\alpha/J^{2/3}$')
    ax.set_ylabel(r'$\partial M_{\rm ext}/\partial\alpha\,|_J$')
    ax.set_xlim(-.012, .53)
    style(ax)
    ax.legend(frameon=False, loc='upper right', handlelength=1.6)
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-shift.pdf')
    plt.close(figure)

    # ---------------------------------------------------------------- figure 4
    figure, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot([r['y'] for r in horizon], [r['near_horizon'] for r in horizon],
            '-', color=ACCENT, label='published near-horizon entropy function')
    ax.plot([r['y'] for r in horizon], [r['sigma'] for r in horizon], 'o',
            markersize=3.8, markerfacecolor='white', markeredgewidth=.9,
            color=INK, label=r'$T\to0$ limit of the bulk solutions')
    ax.set_xlabel(r'$y=\alpha/J^{2/3}$')
    ax.set_ylabel(r'$\sigma=S_{\rm ext}/J$')
    style(ax)
    ax.legend(frameon=False, loc='upper left', handlelength=1.6)
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-entropy.pdf')
    plt.close(figure)

    # ---------------------------------------------------------------- figure 5
    # The extrapolation itself, which is the paper's central modelling step and
    # was previously invisible: the fitted points, the production cubic, the
    # intercepts the other models return, and the residuals of the production
    # fit. Psi_ext is subtracted so that thirteen couplings spanning
    # -1 < Psi < 3 can share one pair of axes and every curve ends at the
    # origin by construction; what the reader judges is how far it had to
    # travel to get there.
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(5.7, 4.2), sharex=True,
        gridspec_kw=dict(height_ratios=[2.35, 1.], hspace=.12))
    fit_colours = plt.cm.viridis(np.linspace(.05, .88, len(fit_audit)))
    ordered_fits = sorted(fit_audit, key=lambda r: r['alpha_gb'])
    for colour, row in zip(fit_colours, ordered_fits):
        run = row['production']
        central = row['central']
        # The abscissa is the coordinate the fit itself uses: the temperature
        # divided by the temperature of the warmest fitted state. Every window
        # then runs from its own coldest state to 1, and thirteen windows
        # spanning four decades of T become directly comparable. The physical
        # temperatures are the T_min column of table 4 and the tau_min column
        # of table 2.
        scaled_T = np.array(run['temperature'])/run['scale']
        grid = np.linspace(0., 1., 160)
        model = np.polynomial.polynomial.polyval(grid, run['coefficients'])
        upper.plot(grid, model-central, '-', color=colour, linewidth=.85)
        upper.plot(scaled_T, np.array(run['values'])-central, 'o',
                   markersize=3.1, color=colour, markeredgewidth=0.)
        # The competing intercepts, drawn on the axis they land on: T = 0.
        others = [v-central for v in row['intercepts'].values()]
        upper.plot(np.zeros(len(others)), others, '_', markersize=6.,
                   markeredgewidth=.8, color=colour, alpha=.8, clip_on=False)
        lower.plot(scaled_T, run['residuals'], 'o-', markersize=2.8,
                   linewidth=.65, color=colour, markeredgewidth=0.)
    upper.axhline(0., color=MUTED, linewidth=.7, zorder=1)
    lower.axhline(0., color=MUTED, linewidth=.7, zorder=1)
    upper.set_ylabel(r'$\Psi(T)-\Psi_\mathrm{ext}$')
    lower.set_ylabel('fit residual')
    lower.set_xlabel(r'$T/T_{\rm window}$,  the coordinate the production fit uses')
    upper.set_xlim(-.012, 1.02)
    lower.set_yscale('symlog', linthresh=1e-8)
    lower.set_yticks([-1e-6, 0., 1e-6])
    style(upper)
    style(lower)
    handles = [plt.Line2D([], [], color=fit_colours[0], marker='o',
                          markersize=3.2, linewidth=.85, markeredgewidth=0.,
                          label='six fitted states and the production cubic'),
               plt.Line2D([], [], color=fit_colours[0], marker='_',
                          markersize=6., linestyle='none', markeredgewidth=.8,
                          label='intercepts of the twenty sensitivity fits')]
    upper.legend(handles=handles, frameon=False, loc='lower left',
                 handlelength=1.6, bbox_to_anchor=(.015, .015))
    scale = matplotlib.colors.ListedColormap(fit_colours)
    cells = matplotlib.colors.BoundaryNorm(np.arange(len(ordered_fits)+1)-.5,
                                           len(ordered_fits))
    figure.tight_layout(pad=.35, rect=(0., 0., .875, 1.))
    bar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=cells, cmap=scale),
                          ax=[upper, lower], fraction=.045, pad=.018,
                          ticks=np.arange(len(ordered_fits)))
    bar.outline.set_linewidth(.5)
    bar.outline.set_edgecolor('#9aa7b0')
    bar.set_ticklabels([rf"${row['alpha_gb']:g}$" for row in ordered_fits])
    bar.ax.tick_params(length=2, width=.5, labelsize=6.8, pad=1.8)
    bar.set_label(r'$\alpha$', size=8.6, labelpad=3.)
    figure.savefig(FIGURES/'fig-extrapolation.pdf')
    plt.close(figure)

    # ---------------------------------------------------------------- figure 6
    # The resolution study, against the thing it has to be compared with: the
    # fit sensitivity already quoted for the same coupling. The message of the
    # figure is the vertical gap between the markers and the line.
    figure, ax = plt.subplots(figsize=(5.4, 3.2))
    ordered_extremals = sorted(extremals, key=lambda r: r['y'])
    ax.plot([r['y'] for r in ordered_extremals],
            [r['psi_uncertainty'] for r in ordered_extremals],
            '-', color=MUTED, linewidth=1.1,
            label='fit sensitivity of the extremal table')
    shapes = {study['coarse_resolutions'][0]: ('o', ACCENT),
              study['coarse_resolutions'][-1]: ('s', '#244a75')}
    for coarse, (marker, colour) in shapes.items():
        rows = sorted((r for r in resolution_audit if r['resolution'] == coarse),
                      key=lambda r: r['y']['fine'])
        ax.plot([r['y']['fine'] for r in rows],
                [r['psi_gb']['absolute'] for r in rows], marker,
                markersize=3.8, markerfacecolor='none', markeredgewidth=1.,
                color=colour,
                label=rf"$|\Psi_{{\rm ext}}^{{N={data['resolution']}}}"
                      rf"-\Psi_{{\rm ext}}^{{N={coarse}}}|$")
    ax.set_yscale('log')
    ax.set_xlabel(r'$y=\alpha/J^{2/3}$')
    ax.set_ylabel('difference in the intercept')
    ax.set_xlim(-.012, .53)
    style(ax)
    ax.legend(frameon=False, loc='center right', handlelength=1.7)
    figure.tight_layout(pad=.35)
    figure.savefig(FIGURES/'fig-resolution.pdf')
    plt.close(figure)

    # ---------------------------------------------------------------- figure 7
    # The extremal mass itself, which the slope figure only describes through
    # its derivative, and the same branch in the domain-of-existence plane the
    # published bounds are stated in.
    figure, (left, right) = plt.subplots(1, 2, figsize=(5.9, 2.85))
    ys = np.array([r['y'] for r in extremals])
    mus = np.array([r['mu'] for r in extremals])
    grid = np.linspace(0., ys[-1]*1.04, 200)
    left.plot(grid, curve['mass_coefficient']+np.pi*grid, '--',
              color=MUTED, linewidth=1.,
              label=r'$\frac{3}{2}\pi^{1/3}+\pi y$  (linear order)')
    left.plot(ys, mus, 'o-', markersize=3.4, color=INK, linewidth=1.05,
              label=r'$\mu(y)$  (this work)')
    left.set_xlabel(r'$y=\alpha/J^{2/3}$')
    left.set_ylabel(r'$\mu=M_{\rm ext}/J^{2/3}$')
    left.set_xlim(-.012, ys[-1]*1.06)
    style(left)
    left.legend(frameon=False, loc='upper left', handlelength=1.6)

    # The same states in the (x, j) plane, where the published domain of
    # existence is bounded by x < 1 and j <= 1 and the extremal set is
    # conjectured to end at (1, 0).
    right.plot([s['x'] for s in states], [s['j'] for s in states], '.',
               markersize=1.4, color='#c3ccd3', zorder=1,
               label='sampled solutions')
    extremal_x = [scaled[r['alpha_gb']]['x'] for r in extremals]
    extremal_j = [scaled[r['alpha_gb']]['j'] for r in extremals]
    right.plot(extremal_x, extremal_j, 'o-', markersize=3.4, color=ACCENT,
               linewidth=1.05, zorder=3, label='extremal branch')
    right.plot([extremal_x[-1], 1.], [extremal_j[-1], 0.], ':', linewidth=1.,
               color=ACCENT, zorder=2)
    right.axhline(1., color=MUTED, linewidth=.8, linestyle=(0, (5, 3)))
    right.axvline(1., color=MUTED, linewidth=.8, linestyle=(0, (5, 3)))
    right.plot([1.], [0.], '*', markersize=8, color='#2f6f3e', zorder=4,
               clip_on=False, label='conjectured endpoint')
    right.set_xlabel(r'$x=3\pi\alpha/(4M)$')
    right.set_ylabel(r'$j$')
    right.set_xlim(-.03, 1.07)
    right.set_ylim(-.05, 1.12)
    style(right)
    right.legend(frameon=False, loc='upper right', handlelength=1.4,
                 markerscale=1.6, borderaxespad=.6)
    figure.tight_layout(pad=.4, w_pad=1.6)
    figure.savefig(FIGURES/'fig-massext.pdf')
    plt.close(figure)

    # ---------------------------------------------------------------- numbers
    ratios = [r for row in refinement for r in row['ratios']]
    oracle_worst = max(r['relative_deviation'] for r in perturbative)
    oracle_good = sum(1 for r in perturbative if r['significant_digits'] >= 8)
    lines = [
        '% Generated by make_figures.py from ../results/egb-extremality.json.',
        '% Do not edit: every macro below is read from the measurement.',
        '',
    ]

    def macro(name, body):
        lines.append(rf'\newcommand{{\N{name}}}{{{body}}}')

    macro('couplings', str(len(walks)))
    macro('states', str(len(states)))
    # Which couplings cover the whole spin range and which do not: the paper
    # distinguishes them, so the split is read from the data and not typed.
    macro('mapcouplings', str(len(wide)))
    macro('narrowcouplings', str(len(walks)-len(wide)))
    # The first entry carries the symbol, so that main.tex can print the list
    # as text: wrapping it in maths there would nest $...$ inside $...$.
    macro('maplist', ', '.join(
        (rf'$\alpha={coupling:g}$' if index == 0 else rf'${coupling:g}$')
        for index, coupling in enumerate(sorted(wide))))
    narrow = [group for coupling, group in walks.items() if coupling not in wide]
    macro('startspin', f"{min(min(s['omega_h'] for s in g) for g in narrow):g}")
    # The independent profile atlas: its agreement with these states, and the
    # horizon squashing it makes visible at the two ends of the coupling range.
    atlas_rows = sorted(atlas['rows'], key=lambda r: r['alpha_gb'])
    macro('profileagreement', latex_float(
        max(max(r['agreement'].values()) for r in atlas_rows if r['agreement'])))
    macro('squashingfirst', fixed(atlas_rows[0]['h_over_r2_horizon'], 4))
    macro('squashinglast', fixed(atlas_rows[-1]['h_over_r2_horizon'], 4))
    macro('resolution', str(data['resolution']))
    macro('refinedresolution', str(data['refined_resolution']))
    macro('coarseresolution', str(data['contrast_resolution']))

    macro('psiextzero', fixed(anchor['psi'], 9))
    macro('psiextzerodev', latex_float(anchor['psi_deviation']))
    macro('muzero', fixed(anchor['mu'], 9))
    macro('muzerodev', latex_float(anchor['mu_deviation']))
    macro('masscoefficient', fixed(curve['mass_coefficient'], 9))

    macro('psimin', fixed(lowest['psi_gb']['value'], 5))
    macro('ymin', fixed(lowest['y'], 4))
    macro('xmin', fixed(scaled[lowest['alpha_gb']]['x'], 4))
    macro('alphamin', f"{lowest['alpha_gb']:g}")
    macro('psilast', fixed(extremals[-1]['psi_gb']['value'], 5))
    macro('ylast', fixed(extremals[-1]['y'], 4))
    macro('xlast', fixed(bounds['worst_x'], 4))
    macro('xextlast', fixed(scaled[extremals[-1]['alpha_gb']]['x'], 4))
    macro('worstj', fixed(bounds['worst_j'], 6))
    macro('sigmagrowth', fixed(extremals[-1]['sigma']/extremals[0]['sigma'], 1))
    # The first coupling at which the shift has fallen below half its
    # linear-order value: quoted in the discussion, so read rather than typed.
    half = next(r for r in extremals if r['psi_gb']['value'] < np.pi/2)
    macro('halfpsi', fixed(half['psi_gb']['value'], 4))
    macro('halfx', fixed(scaled[half['alpha_gb']]['x'], 4))
    macro('psiminshort', fixed(lowest['psi_gb']['value'], 2))
    macro('psilastshort', fixed(extremals[-1]['psi_gb']['value'], 2))
    macro('yminshort', fixed(lowest['y'], 3))
    macro('ylastshort', fixed(extremals[-1]['y'], 3))
    macro('suppression', fixed(np.pi/lowest['psi_gb']['value'], 2))
    macro('griderror', latex_float(lowest['psi_uncertainty']))

    # The extremal mass itself, and how far the linear-order expression is from
    # it at the largest coupling sampled: the two statements figure 7 makes.
    macro('mugrowth', fixed(100*(extremals[-1]['mu']/extremals[0]['mu']-1), 0))
    macro('linearratio', fixed(
        (curve['mass_coefficient']+np.pi*extremals[-1]['y'])/extremals[-1]['mu'], 2))
    # The slope the published mass bound forces the branch towards if the
    # endpoint conjecture of arXiv:2303.12471 holds.
    macro('slopeasymptote', fixed(3*np.pi/4, 3))
    # The extremal horizon angular velocity, extrapolated on its own.
    macro('omegapeak', fixed(omega_peak['omega'], 3))
    macro('omegapeaky', fixed(omega_peak['y'], 3))
    macro('omegaworst', latex_float(max(r['difference'] for r in omega_rows)))
    macro('omegazero', fixed(omega_rows[0]['omega'], 6))
    macro('omegazerodev',
          latex_float(abs(omega_rows[0]['omega']-np.pi**(1/3)/2)))
    # Scale invariance fixes j on the extremal branch once mu is known; the
    # deviation measures the two extrapolations against each other.
    macro('jrelationworst', latex_float(max(
        abs(scaled[r['alpha_gb']]['j']-(extremals[0]['mu']/r['mu'])**1.5)
        for r in extremals)))

    macro('oracledigits', '8')
    macro('oraclegood', str(oracle_good))
    macro('oracletotal', str(len(perturbative)))
    macro('oracleworst', latex_float(oracle_worst))
    macro('perturbativezero', fixed(data['perturbative_zero'], 6))
    macro('locuszero', fixed(locus[0]['q'], 5))
    macro('locuszerodev', latex_float(abs(locus[0]['q']-data['perturbative_zero'])))
    macro('locuszerogap', f"{abs(locus[0]['bracket_high']['q']-locus[0]['bracket_low']['q']):g}")

    macro('ratiolow', fixed(min(ratios), 3))
    macro('ratiohigh', fixed(max(ratios), 3))
    macro('refinementpoints', str(len(refinement)))
    macro('refinementsteps', str(len(refinement[0]['ladder'])))
    macro('largestgap', latex_float(max(abs(e['gap']) for row in refinement
                                        for e in row['ladder'])))

    macro('taufloor', latex_float(max(min(s['tau'] for s in group)
                                      for group in walks.values())))
    macro('taubest', latex_float(min(s['tau'] for s in states)))
    macro('tensorworst', latex_float(max(s['max_relative_tensor_residual']
                                         for s in states)))
    macro('smarrworst', latex_float(max(s['smarr'] for s in states)))
    macro('contrastworst', latex_float(max(r['psi_difference'] for r in contrast)))
    macro('contrastmu', latex_float(max(r['mu_difference'] for r in contrast)))
    macro('contrastcouplings', str(len(contrast)))

    # The resolution study: worst case per observable over every coupling and
    # both coarse resolutions, and the three statements the appendix makes
    # about it.
    coarse_list = study['coarse_resolutions']
    macro('studyresolutions', ' and '.join(str(n) for n in coarse_list))
    macro('studylowresolution', str(coarse_list[0]))
    macro('studyhighresolution', str(coarse_list[-1]))
    macro('studywalks', str(len(resolution_audit)))
    for name, key in (('psi', 'psi_gb'), ('mu', 'mu'), ('y', 'y'),
                      ('sigma', 'sigma'), ('spin', 'j')):
        macro(f'study{name}worst',
              latex_float(max(r[key]['absolute'] for r in resolution_audit)))
    for name, key in (('mass', 'E'), ('momentum', 'J'), ('entropy', 'S')):
        macro(f'study{name}worst',
              latex_float(max(r[key]['relative'] for r in resolution_audit)))
    macro('studybelow', str(sum(1 for r in resolution_audit
                                if r['below_envelope'])))
    macro('studyratio', latex_float(max(
        r['psi_gb']['absolute']/r['envelope'] for r in resolution_audit)))
    macro('studyshrink', str(study['psi_shrinking_couplings']))
    macro('studycouplings', str(len(study['extremal_couplings'])))
    macro('studyescalated', str(sum(1 for r in resolution_audit
                                    if r['escalated_on_both'])))
    matched = [r['matched'] for r in resolution_audit if r['matched']]
    macro('studymatchedwalks', str(len(matched)))
    macro('studymatchedworst',
          latex_float(max(m['absolute'] for m in matched)))
    # How much of the quoted difference is the environment rather than the
    # resolution: the same walk at the production resolution, re-run here.
    macro('studycontrolworst', latex_float(study['worst_environment_difference']))
    macro('studycontrollist', ', '.join(
        rf'$\alpha={value:g}$' if index == 0 else rf'${value:g}$'
        for index, value in enumerate(study['environment_control_couplings'])))
    shipped = study['shipped_versions']
    macro('studyshipped', f"Python {shipped['python']}, NumPy {shipped['numpy']}, "
                          f"SciPy {shipped['scipy']}")
    macro('studyhere', f"Python {study['versions']['python']}, "
                       f"NumPy {study['versions']['numpy']}, "
                       f"SciPy {study['versions']['scipy']}")

    macro('routeworst', latex_float(routes['worst_difference']))
    macro('routeworstrel', latex_float(routes['worst_relative']))
    macro('routetrapezoid', latex_float(routes['worst_trapezoid_difference']))
    # Two different quantities, previously printed as one. `worst_cumulative`
    # is the largest running discrepancy, reached before the last interval;
    # the discrepancy across the whole range is the value at the last interval.
    macro('routecumulative', latex_float(routes['worst_cumulative']))
    macro('routecumulativeend', latex_float(routes['cumulative'][-1]['difference']))
    macro('routecumulativey', fixed(routes['cumulative'][-1]['y'], 4))
    macro('routewithinspread', str(sum(
        1 for row in routes['intervals']
        if abs(row['difference']) <= row['combined_spread'])))
    macro('routeintervals', str(len(routes['intervals'])))
    macro('routespreadratio', fixed(max(abs(row['difference'])/row['combined_spread']
                                        for row in routes['intervals']), 2))
    macro('mutotal', fixed(curve['mu'][-1]-curve['mu'][0], 4))
    macro('massslope', fixed(curve['slope_at_zero'], 6))
    macro('massslopedev', latex_float(curve['slope_at_zero_deviation']))
    macro('massslopespread', latex_float(curve['slope_at_zero_spread']))

    # An evaluation of Psi that does not use the implicit-response solve: the
    # Smarr relation, read as an equation for Psi at every finite-coupling
    # state, and separately extrapolated to T = 0 the production way.
    smarr_states, smarr_worst, smarr_absolute = 0, 0., 0.
    for coupling, group in walks.items():
        if coupling == 0.:
            continue
        for state in group:
            implied = smarr_potential(state, coupling)
            smarr_absolute = max(smarr_absolute, abs(implied-state['psi_gb']))
            smarr_worst = max(smarr_worst,
                              abs(implied-state['psi_gb'])/abs(state['psi_gb']))
            smarr_states += 1
    smarr_rows = []
    for record in extremals:
        if record['alpha_gb'] == 0.:
            continue
        group = [dict(s, psi_gb=smarr_potential(s, record['alpha_gb']))
                 for s in walks[record['alpha_gb']]]
        intercept = production_fit(group)['intercept']
        central = record['psi_gb']['value']
        smarr_rows.append((record['alpha_gb'], central, intercept,
                           abs(intercept-central)/abs(central)))
    # A third route to Psi, sharing neither the response solve nor the
    # asymptotic charge extraction: the on-shell Gauss-Bonnet action.
    macro('onshellstates', str(onshell['states']))
    macro('onshellworst', latex_float(onshell['worst_relative']))
    macro('onshellabsolute', latex_float(onshell['worst_absolute']))
    macro('onshellalphahigh', f"{onshell['alpha_range'][1]:g}")
    macro('onshellspinhigh', f"{onshell['q_range'][1]:g}")
    macro('onshellnodes', str(onshell['nodes']))

    macro('smarrstates', str(smarr_states))
    macro('smarrpsiworst', latex_float(smarr_worst))
    macro('smarrpsiabsolute', latex_float(smarr_absolute))
    macro('smarrextcouplings', str(len(smarr_rows)))
    macro('smarrextworst', latex_float(max(row[3] for row in smarr_rows)))

    # How well the production cubic describes the six states it is given, and
    # how far its intercept moves when one of those six is dropped.
    macro('fitresidualworst',
          latex_float(max(row['production']['worst_residual'] for row in fit_audit)))
    macro('looworst',
          latex_float(max(row['leave_one_out']['worst'] for row in fit_audit)))
    macro('fitwindow', str(len(fit_audit[0]['production']['temperature'])))

    macro('staticpsiworst', latex_float(max(r['psi_relative'] for r in static)))
    macro('staticmassworst', latex_float(max(r['mass_relative'] for r in static)))
    macro('staticentropyworst',
          latex_float(max(max(r['entropy_relative'], 1e-16) for r in static)))
    macro('statictempworst',
          latex_float(max(r['temperature_relative'] for r in static)))
    macro('staticcouplings', str(len(static)))
    macro('vacuummassworst',
          latex_float(max(r['relative'] for r in data['vacuum_masses'])))
    macro('firstlawworst',
          latex_float(max(r['first_law']['relative'] for r in consistency)))
    macro('goonpencoworst',
          latex_float(max(r['goon_penco']['relative'] for r in consistency)))
    macro('conditionworst',
          latex_float(max(r['conditioning']['condition_number']
                          for r in consistency), 1))
    macro('consistencypoints', str(len(consistency)))
    macro('vacuumspin', f"{max(r['q'] for r in data['vacuum_masses']):g}")
    macro('tailspreadworst', latex_float(tails['worst']))
    # The asymptotic fit windows, read from the module that applies them so
    # that the appendix cannot quote a window the measurement did not use.
    from rotating_bh.gb_response import TAIL_WINDOWS
    macro('tailwindows', ', '.join(
        rf'$z\in[{lo:g},{hi:g}]$ at degree ${degree:g}$'
        for lo, hi, degree in TAIL_WINDOWS))
    low_index = extremals.index(lowest)
    macro('minimumlow', fixed(extremals[low_index-1]['y'], 3))
    macro('minimumhigh', fixed(extremals[low_index+1]['y'], 3))
    macro('horizonworst', latex_float(max(r['relative_difference'] for r in horizon)))
    macro('publishedstationarity', latex_float(
        max(r['stationarity'] for r in data['published_near_horizon'])))
    macro('publishedspin', latex_float(
        max(abs(r['spin_ratio']-1) for r in data['published_near_horizon'])))
    macro('vacuumj', fixed(bounds['vacuum_extremal_j'], 8))
    macro('jlast', fixed(bounds['extremal_j_at_largest_coupling'], 5))
    macro('jlocusfirst', fixed(locus[0]['j'], 4))
    macro('jlocuslast', fixed(locus[-1]['j'], 4))
    macro('tlocusfirst', fixed(locus[0]['t_scaled'], 4))
    macro('tlocusmin', fixed(min(r['t_scaled'] for r in locus), 4))
    macro('alphalocusmin', f"{min(locus, key=lambda r: r['t_scaled'])['alpha_gb']:g}")

    # ------------------------------------------------------------ table bodies
    lines += ['', '% Table: the sign-change locus.',
              r'\newcommand{\Nlocustable}{%']
    for row in locus:
        # The crossing is bracketed by two accepted solutions of opposite sign
        # and interpolated between them, so the bracket is printed next to the
        # interpolant. The interpolation is linear in q, which is why the
        # bracket is quoted in q and not in one of the derived coordinates.
        low = min(row['bracket_low']['q'], row['bracket_high']['q'])
        high = max(row['bracket_low']['q'], row['bracket_high']['q'])
        lines.append(rf"  ${row['alpha_gb']:g}$ & ${fixed(row['x'], 4)}$ & "
                     rf"$[{low:g},\,{high:g}]$ & ${fixed(row['q'], 5)}$ & "
                     rf"${fixed(row['t_scaled'], 4)}$ & ${fixed(row['j'], 4)}$ \\")
    lines.append('}')

    lines += ['', '% Table: the extremal branch.',
              r'\newcommand{\Nextremaltable}{%']
    for record in extremals:
        row = scaled[record['alpha_gb']]
        # How cold the walk at this coupling actually got. The row itself is a
        # T=0 intercept, so this is the only temperature in it that was measured.
        tau_min = min(s['tau'] for s in walks[record['alpha_gb']])
        lines.append(
            rf"  ${record['alpha_gb']:g}$ & ${fixed(record['y'], 5)}$ & "
            rf"${fixed(row['x'], 5)}$ & ${latex_float(tau_min)}$ & "
            rf"${fixed(record['mu'], 6)}$ & "
            rf"${fixed(record['psi_gb']['value'], 5)}$ & "
            rf"${latex_float(record['psi_uncertainty'])}$ & "
            rf"${fixed(row['j'], 5)}$ & ${fixed(record['sigma'], 4)}$ \\")
    lines.append('}')

    # The lever-arm table that explains why four of the uncertainties are large.
    lines += ['', '% Table: sensitivity of the T=0 intercept to the fit model.',
              r'\newcommand{\Nmodeltable}{%']
    for row in sorted(models, key=lambda r: r['alpha_gb']):
        sqrt_shift = max(abs(v-row['central']) for k, v in row['intercepts'].items()
                         if '_sqrt_' in k)
        log_shift = max(abs(v-row['central']) for k, v in row['intercepts'].items()
                        if '_log_' in k)
        lines.append(
            rf"  ${row['alpha_gb']:g}$ & ${latex_float(row['temperature_min'])}$ & "
            rf"${fixed(row['temperature_range'], 0)}$ & "
            rf"${latex_float(sqrt_shift)}$ & "
            rf"${latex_float(log_shift)}$ \\")
    lines.append('}')

    # The uncertainty budget: every extrapolated column of the extremal table,
    # not only Psi. y, mu and sigma carry the spread of their own intercepts;
    # j is formed from the extrapolated E and J and is propagated through them.
    lines += ['', '% Table: the recorded spreads of every extrapolated column.',
              r'\newcommand{\Nbudgettable}{%']
    shape = (1.5**1.5)*np.sqrt(np.pi)
    for record in extremals:
        scaled_j = [shape*spin/mass**1.5 for mass, spin
                    in zip(record['E']['intercepts'], record['J']['intercepts'])]
        audit = next(row for row in fit_audit
                     if row['alpha_gb'] == record['alpha_gb'])
        lines.append(
            rf"  ${record['alpha_gb']:g}$ & ${latex_float(record['y_spread'])}$ & "
            rf"${latex_float(record['mu_spread'])}$ & "
            rf"${latex_float(max(scaled_j)-min(scaled_j))}$ & "
            rf"${latex_float(record['sigma_spread'])}$ & "
            rf"${latex_float(record['psi_uncertainty'])}$ & "
            rf"${latex_float(audit['production']['worst_residual'])}$ & "
            rf"${latex_float(audit['leave_one_out']['worst'])}$ \\")
    lines.append('}')

    # The resolution comparison, at the three couplings where it exists. The
    # columns are separated because a small change in Psi does not by itself
    # bound the change in the invariants that the mass-curve check uses.
    lines += ['', '% Table: the resolution comparison walks.',
              r'\newcommand{\Ncontrasttable}{%']
    for row in sorted(contrast, key=lambda r: r['alpha_gb']):
        lines.append(
            rf"  ${row['alpha_gb']:g}$ & ${fixed(row['psi_fine'], 6)}$ & "
            rf"${latex_float(abs(row['psi_difference']))}$ & "
            rf"${fixed(row['mu_fine'], 6)}$ & "
            rf"${latex_float(abs(row['mu_difference']))}$ & "
            rf"${latex_float(abs(row['y_fine']-row['y_coarse']))}$ \\")
    lines.append('}')

    # The real finite-difference comparison, with its locations, step sizes and
    # final errors printed rather than only the ratios between them.
    # The resolution study, one row per coupling and coarse resolution, with
    # M, J, S, Psi, y and mu separated because the referee asked for them
    # separately and because a small change in one does not bound another.
    lines += ['', '% Table: the radial-resolution study.',
              r'\newcommand{\Nresolutiontable}{%']
    for row in sorted(resolution_audit,
                      key=lambda r: (r['alpha_gb'], r['resolution'])):
        lines.append(
            rf"  ${row['alpha_gb']:g}$ & ${row['resolution']}$ & "
            rf"${latex_float(row['E']['relative'])}$ & "
            rf"${latex_float(row['J']['relative'])}$ & "
            rf"${latex_float(row['S']['relative'])}$ & "
            rf"${latex_float(row['psi_gb']['absolute'])}$ & "
            rf"${latex_float(row['y']['absolute'])}$ & "
            rf"${latex_float(row['mu']['absolute'])}$ \\")
    lines.append('}')

    lines += ['', '% Table: response against real central differences.',
              r'\newcommand{\Nrefinementtable}{%']
    for row in sorted(refinement, key=lambda r: (r['alpha_gb'], r['q'])):
        first, last = row['ladder'][0], row['ladder'][-1]
        lines.append(
            rf"  ${row['q']:g}$ & ${row['alpha_gb']:g}$ & "
            rf"${fixed(row['linear_response'], 7)}$ & "
            rf"${first['step']:g}$ & ${latex_float(abs(first['gap']))}$ & "
            rf"${last['step']:g}$ & ${latex_float(abs(last['gap']))}$ & "
            rf"${fixed(min(row['ratios']), 3)}$--${fixed(max(row['ratios']), 3)}$ \\")
    lines.append('}')

    # The Smarr route, coupling by coupling: a value of Psi_ext that the
    # implicit-response solve did not produce.
    lines += ['', '% Table: the Smarr-derived potential against the response.',
              r'\newcommand{\Nsmarrtable}{%']
    for coupling, central, intercept, relative in smarr_rows:
        lines.append(rf"  ${coupling:g}$ & ${fixed(central, 6)}$ & "
                     rf"${fixed(intercept, 6)}$ & ${latex_float(relative)}$ \\")
    lines.append('}')
    lines.append('')

    supplementary = HERE/'supplementary'
    supplementary.mkdir(exist_ok=True)
    source_paths = [DATA, RESOLUTION_STUDY, HERE/'reanalysis.py',
                    Path(__file__).resolve()]
    audit = dict(
        central_method='cubic in T, six coldest states; production values retained',
        envelope_method='max deviation across models, T/tau, six/up-to-twelve states, and original degree spread',
        interpretation='fit sensitivity, not confidence intervals or rigorous error bounds',
        resolution_study=resolution_audit,
        source_sha256={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_paths},
        fits=fit_audit)
    (supplementary/'extrapolation-sensitivity.json').write_text(
        json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    columns = ('alpha_gb', 'omega_h', 'E', 'J', 'S', 'T_H', 'A_H', 'psi_gb',
               'y', 'tau', 'mu', 'sigma', 'x', 'j', 't_scaled', 'resolution',
               'max_tensor_residual', 'max_relative_tensor_residual', 'smarr')
    with (supplementary/'solutions.csv').open('w', encoding='utf-8', newline='') as handle:
        handle.write('# Every accepted solution behind the figures and tables of\n')
        handle.write('# "Extremality shift of rotating black holes at finite '
                     'Gauss-Bonnet coupling".\n')
        handle.write('# Units G_5 = r_H = 1; J is each of the two angular '
                     'momenta; Omega_H = omega_h.\n')
        handle.write(','.join(columns)+'\n')
        for state in sorted(states, key=lambda s: (s['alpha_gb'], s['omega_h'])):
            handle.write(','.join(repr(state[name]) for name in columns)+'\n')
    print(f'wrote supplementary/solutions.csv with {len(states)} rows')

    NUMBERS.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'wrote {NUMBERS.name} with {sum(1 for l in lines if l.startswith(chr(92)+"newcommand"))} macros')
    for name in sorted(FIGURES.glob('*.pdf')):
        print(f'wrote figures/{name.name}  {name.stat().st_size} bytes')


if __name__ == '__main__':
    main()
