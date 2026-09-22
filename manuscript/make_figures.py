"""Build the manuscript's figures, tables and numbers from the measurement.

Single rule, inherited from the rest of the project: **no number that appears in
the paper is typed into the paper.** Everything is read here from
`../results/egb-extremality.json` and emitted as

  * `figures/fig-potential.pdf`   -- Psi across the family,
  * `figures/fig-shift.pdf`       -- the extremality shift, both routes,
  * `figures/fig-entropy.pdf`     -- the extremal entropy against [1, sec. 4.2],
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
from reanalysis import sensitivity

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'results/egb-extremality.json'
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
    walks = {float(k): v for k, v in data['walks'].items()}
    extremals = sorted(data['extremals'], key=lambda r: r['y'])
    fit_audit = []
    for record in extremals:
        group = walks[record['alpha_gb']]
        fitted = sensitivity(group, record['psi_gb']['value'])
        record['psi_uncertainty'] = max(record['psi_gb']['spread'],
                                         fitted['envelope'])
        cold = sorted(group, key=lambda s: s['T_H'])[:12]
        fitted.update(alpha_gb=record['alpha_gb'],
                      envelope=record['psi_uncertainty'],
                      temperature_min=cold[0]['T_H'],
                      temperature_range=cold[-1]['T_H']/cold[0]['T_H'])
        fit_audit.append(fitted)
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
    wide = {k: v for k, v in sorted(walks.items()) if len(v) >= 25}
    lowest = min(extremals, key=lambda r: r['psi_gb']['value'])

    # ---------------------------------------------------------------- figure 1
    figure, ax = plt.subplots(figsize=(5.4, 3.3))
    colours = plt.cm.viridis(np.linspace(.05, .82, len(wide)))
    for colour, (coupling, group) in zip(colours, wide.items()):
        ordered = sorted(group, key=lambda s: s['t_scaled'])
        ax.plot([s['t_scaled'] for s in ordered], [s['psi_gb'] for s in ordered],
                color=colour, label=rf'$\alpha={coupling:g}$')
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

    # ---------------------------------------------------------------- figure 2
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

    # ---------------------------------------------------------------- figure 3
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
    macro('suppression', fixed(np.pi/lowest['psi_gb']['value'], 2))
    macro('griderror', latex_float(lowest['psi_uncertainty']))

    macro('oracledigits', '8')
    macro('oraclegood', str(oracle_good))
    macro('oracletotal', str(len(perturbative)))
    macro('oracleworst', latex_float(oracle_worst))
    macro('perturbativezero', fixed(data['perturbative_zero'], 6))
    macro('locuszero', fixed(locus[0]['q'], 5))
    macro('locuszerodev', latex_float(abs(locus[0]['q']-data['perturbative_zero'])))

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

    macro('routeworst', latex_float(routes['worst_difference']))
    macro('routeworstrel', latex_float(routes['worst_relative']))
    macro('routetrapezoid', latex_float(routes['worst_trapezoid_difference']))
    macro('routecumulative', latex_float(routes['worst_cumulative']))
    macro('mutotal', fixed(curve['mu'][-1]-curve['mu'][0], 4))
    macro('massslope', fixed(curve['slope_at_zero'], 6))
    macro('massslopedev', latex_float(curve['slope_at_zero_deviation']))
    macro('massslopespread', latex_float(curve['slope_at_zero_spread']))

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
    macro('jlocusfirst', fixed(locus[0]['j'], 5))
    macro('jlocuslast', fixed(locus[-1]['j'], 5))
    macro('tlocusfirst', fixed(locus[0]['t_scaled'], 5))
    macro('tlocusmin', fixed(min(r['t_scaled'] for r in locus), 5))
    macro('alphalocusmin', f"{min(locus, key=lambda r: r['t_scaled'])['alpha_gb']:g}")

    # ------------------------------------------------------------ table bodies
    lines += ['', '% Table: the sign-change locus.',
              r'\newcommand{\Nlocustable}{%']
    for row in locus:
        # The crossing is bracketed by two accepted solutions, not resolved to
        # the digits of the interpolant; quote the bracket, not just its centre.
        width = abs(row['bracket_high']['t_scaled'] - row['bracket_low']['t_scaled'])
        lines.append(rf"  ${row['alpha_gb']:g}$ & ${fixed(row['x'], 4)}$ & "
                     rf"${fixed(row['t_scaled'], 5)}$ & ${latex_float(width)}$ & "
                     rf"${fixed(row['j'], 5)}$ & ${fixed(row['q'], 5)}$ \\")
    lines.append('}')

    lines += ['', '% Table: the extremal branch.',
              r'\newcommand{\Nextremaltable}{%']
    for record in extremals:
        row = scaled[record['alpha_gb']]
        lines.append(
            rf"  ${record['alpha_gb']:g}$ & ${fixed(record['y'], 5)}$ & "
            rf"${fixed(row['x'], 5)}$ & ${fixed(record['mu'], 6)}$ & "
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
    lines.append('')

    supplementary = HERE/'supplementary'
    supplementary.mkdir(exist_ok=True)
    source_paths = [DATA, HERE/'reanalysis.py', Path(__file__).resolve()]
    audit = dict(
        central_method='cubic in T, six coldest states; production values retained',
        envelope_method='max deviation across models, T/tau, six/up-to-twelve states, and original degree spread',
        interpretation='fit sensitivity, not confidence intervals or rigorous error bounds',
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
