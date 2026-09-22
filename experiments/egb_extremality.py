"""Hito 8: the extremal mass of the rotating EGB family at finite coupling.

Section 4.2 of arXiv:1010.0860v1 determines the extremal near-horizon geometry
of this family by the entropy function, gets `S_ext` in closed form, and then
says what it cannot do: "finding local solutions in the vicinity of the horizon
does not guarantee the existence of global asymptotically flat solutions.
Further progress in this direction seems to require an explicit construction of
the bulk extremal black hole solutions. For example, this would also allow to
construct the E(J) diagram for such configurations."

This experiment constructs that diagram. What makes it possible is not a new
formalism but two changes of method, both in `plans/2026-09-14-hito-8.md`:

* **Continuation seeded by linear response.** Transporting the previous
  solution along `du/dalpha` and `du/domega` instead of reusing it walks the
  spin to within `1e-5` of extremality at every coupling tried, where seeding
  by the previous solution stalls an order of magnitude earlier. The `q=0.68`
  and `q=0.70` walls recorded in Hito 6 and its addendum were artefacts of the
  starting direction, not boundaries.
* **The coupling derivative taken exactly.** `Psi_GB` comes from one linear
  solve rather than from central differences in the coupling, so it carries no
  truncation error at all. The central differences converge to it at exactly
  second order, which is the evidence that the replacement is legitimate and is
  recorded here rather than asserted.

What is measured, in scale-invariant variables `y = alpha_GB/J^{2/3}` and
`tau = T_H J^{1/3}`:

1. `Psi_0(q)` at `alpha_GB=0`, exactly, against the closed form of
   `reports/originality-precheck.md`.
2. `Psi_GB(y,tau)` over the family, each value with an error budget.
3. `mu(y) = M_ext/J^{2/3}` and `sigma(y) = S_ext/J`, by extrapolating to
   `tau=0`.
4. `dM_ext/dalpha|_J`, twice: as `lim_{tau->0} Psi_GB` (Goon-Penco), and as
   `mu'(y)` from masses and spins alone -- no entropy, no temperature, no first
   law. The second route shares only the solver with the first.
5. The locus where `Psi_GB` changes sign, in `tau` at fixed `y`, which is the
   Hito 6 result restated in variables that do not cross states at different
   distances from extremality.

External anchors, none of them reproduced by construction:
`mu(0) = (3/2) pi^{1/3}` and `mu'(0) = pi` (arXiv:2009.00015), and
`sigma(y)` against eqs. (4.11)-(4.20) of arXiv:1010.0860v1 at finite coupling.

Run with PYTHONPATH=src from project/.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from rotating_bh.consistency import (first_law_residual, goon_penco_residual,
                                     jacobian_conditioning, myers_perry_mass,
                                     static_closed_form)
from rotating_bh.extremality import (BOUNDARY_GATE, MASS_COEFFICIENT,
                                     PUBLISHED_SHIFT, RELATIVE_TENSOR_GATE,
                                     TENSOR_GATE, extrapolate, extremal_state,
                                     invariants, state_at, walk)
from rotating_bh.gb_potential import perturbative_potential, perturbative_zero
from rotating_bh.gb_response import (measured_perturbative_potential,
                                     predictor_ladder, response_observables)
from rotating_bh.near_horizon import coupling_from_invariant, extremal, paper_residuals
from rotating_bh.egb_rotating_observables import measure

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT/'.cache/hito-8-progress.json'

RESOLUTION = 64
# Raised automatically by `walk` when the independent tensor residual refuses a
# state; near extremality that is resolution, not step size (see extremality.py).
REFINED_RESOLUTION = 96
CONTRAST_RESOLUTION = 48
START_SPIN = .60
# Couplings walked all the way to extremality. The spacing is finer at the
# small end because mu(y) is being differentiated there and the published
# anchor mu'(0) = pi lives at y=0.
EXTREMAL_COUPLINGS = [0., .005, .01, .02, .035, .05, .075, .1, .15, .2, .3, .4, .5]
# Couplings that also get the descending branch, so Psi_GB is mapped over the
# whole spin range and not only near extremality.
MAP_COUPLINGS = [0., .02, .05, .1, .2, .5]
# Where the exact potential is compared with its own central-difference
# ancestor. Two of these are the points where Hito 6 was least accurate.
REFINEMENT_POINTS = [(.65, .02), (.65, .06), (.6, .02), (.3, .02)]
REFINEMENT_STEPS = [.008, .004, .002, .001, .0005]
PERTURBATIVE_SPINS = [0., .1, .2, .3, .4, .5, .6, .65, .68, .70, .705]
# The static limit is closed-form at every coupling, so it is the one exact
# test available at alpha != 0. Added in revision after a referee pointed out
# that the first version had exact anchors only at alpha = 0.
STATIC_COUPLINGS = [0., .05, .1, .2, .3, .4, .5]
# Where the first law in the spin direction, the Goon-Penco identity and the
# conditioning of dR/du are evaluated. Chosen to span the family rather than to
# be many: each is exact at the point, so a grid buys coverage, not precision.
# Every entry must have q < 1/sqrt(2): these ladders climb in the coupling at
# fixed spin from the vacuum, where that is the extremal bound. Points beyond it
# exist at finite coupling but are reached by walking in the spin instead.
CONSISTENCY_GRID = [(.2, .02), (.3, .1), (.4, .05), (.5, .2), (.6, .05),
                    (.6, .3), (.65, .02), (.68, .1), (.69, .5), (.70, .2)]
# Spins at which the alpha=0 mass is compared with the Myers-Perry closed form
# along the whole range, not only at the static end.
VACUUM_SPINS = [.05, .2, .4, .6, .68, .70]


class Progress:
    """Disk memo, keyed by what defines a walk. The run is an hour of solves."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                self.entries = {}

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, value):
        self.entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, allow_nan=False), encoding='utf-8')
        return value


def approach(alpha_gb, resolution, cache, descend=False):
    """One coupling: climb to it, then walk the spin in the asked directions."""
    key = f'walk|{alpha_gb:.12g}|{resolution}|{int(descend)}'
    stored = cache.get(key)
    if stored is not None:
        # Invariants are derived, never stored authoritatively: recomputing them
        # on the way out means a change in how they are defined reaches cached
        # walks too, instead of silently applying only to fresh ones.
        for state in stored:
            state.update(invariants(state))
        return stored
    started = time.time()
    ladder = (resolution, REFINED_RESOLUTION) if resolution < REFINED_RESOLUTION         else (resolution,)
    base = predictor_ladder(START_SPIN, alpha_gb, step=.005, resolution=resolution)
    states = [state_at(base)]
    if descend:
        states += walk(base, alpha_gb, direction=-1, resolutions=ladder,
                       stop=lambda spin, _: spin <= 0.)
    states += walk(base, alpha_gb, direction=+1, resolutions=ladder, floor=1e-6)
    states.sort(key=lambda s: s['omega_h'])
    for state in states:
        state.update(invariants(state))
    cache.put(key, states)
    print(f'  alpha_gb={alpha_gb:<6g} N={resolution}+ {len(states):3d} states, '
          f'q up to {states[-1]["omega_h"]:.5f}, tau down to '
          f'{min(s["tau"] for s in states):.2e}  [{time.time()-started:.0f}s]',
          flush=True)
    return states


def static_limit(cache):
    """The q=0 member against its closed form, at finite coupling.

    This is the exact test at `alpha != 0` that the first version of this
    measurement lacked. It exercises the whole chain -- solver, mass and spin
    extraction, Wald entropy, and the linear response of section 3 -- against
    an answer known in closed form, rather than against an extrapolation.
    """
    rows = []
    for alpha_gb in STATIC_COUPLINGS:
        key = f'static|{alpha_gb:.12g}'
        stored = cache.get(key)
        if stored is not None:
            rows.append(stored)
            continue
        solution = predictor_ladder(0., alpha_gb, step=.005, resolution=RESOLUTION)
        observed = measure(solution)
        derivative = response_observables(solution)
        exact = static_closed_form(alpha_gb)
        row = dict(alpha_gb=alpha_gb, exact=exact,
                   E=observed['E'], T_H=observed['T_H'], S=observed['S'],
                   psi_gb=derivative['psi_gb'],
                   mass_relative=abs(observed['E']-exact['E'])/exact['E'],
                   temperature_relative=abs(observed['T_H']-exact['T_H'])/exact['T_H'],
                   entropy_relative=abs(observed['S']-exact['S'])/exact['S'],
                   psi_relative=(abs(derivative['psi_gb']-exact['psi_gb'])
                                 / max(abs(exact['psi_gb']), 1e-30)))
        print(f"  static alpha_gb={alpha_gb:<5g} Psi {row['psi_gb']:+.8f} vs exact "
              f"{exact['psi_gb']:+.8f} (rel {row['psi_relative']:.2e})", flush=True)
        rows.append(cache.put(key, row))
    return rows


def vacuum_masses(cache):
    """The alpha=0 mass along the spin range, against Myers-Perry.

    `M = 3 pi r_H^2/(8(1-q^2))` in the gauge of the ansatz. Reported because it
    bounds the error in the mass extraction directly, without any extrapolation,
    and because that bound is what the digits quoted for `mu` rest on.
    """
    rows = []
    for spin in VACUUM_SPINS:
        key = f'vacuum-mass|{spin:.12g}'
        stored = cache.get(key)
        if stored is not None:
            rows.append(stored)
            continue
        solution = predictor_ladder(spin, 0., step=.005, resolution=RESOLUTION)
        observed = measure(solution)
        exact = myers_perry_mass(spin)
        rows.append(cache.put(key, dict(
            q=spin, E=observed['E'], exact=exact,
            relative=abs(observed['E']-exact)/exact)))
    print(f"  worst mass error against Myers-Perry: "
          f"{max(r['relative'] for r in rows):.2e}", flush=True)
    return rows


def consistency_checks(cache):
    """First law in the spin direction, Goon-Penco, and the conditioning.

    None of these enters the extremal measurement; all three are exact at the
    point at which they are evaluated, because both linear responses are.
    """
    rows = []
    for spin, alpha_gb in CONSISTENCY_GRID:
        key = f'consistency|{spin:.12g}|{alpha_gb:.12g}'
        stored = cache.get(key)
        if stored is not None:
            rows.append(stored)
            continue
        solution = predictor_ladder(spin, alpha_gb, step=.005,
                                    resolution=CONTRAST_RESOLUTION)
        rows.append(cache.put(key, dict(
            q=spin, alpha_gb=alpha_gb,
            first_law=first_law_residual(solution),
            goon_penco=goon_penco_residual(solution),
            conditioning=jacobian_conditioning(solution))))
    print(f"  first law worst {max(r['first_law']['relative'] for r in rows):.2e}; "
          f"Goon-Penco worst "
          f"{max(r['goon_penco']['relative'] for r in rows):.2e}; "
          f"worst cond(dR/du) "
          f"{max(r['conditioning']['condition_number'] for r in rows):.1e}",
          flush=True)
    return rows


def mass_bound(walks):
    """An exact upper bound on `x` that the data violate, and by how much.

    In the static limit `x = 2 abar/(1+2 abar)` exactly, and `x = 3 pi alpha/(4M)`
    decreases as the spin grows, so no member of the family at a given coupling
    may exceed the static value. Where a measured state does, the excess is a
    bound on the error of the mass extraction that owes nothing to our own
    diagnostics -- which is the point of reporting it: at the largest coupling it
    is four orders of magnitude above what the tail-window spread claims there.

    Equivalent to the statement that the measured `M` dips below the exact static
    mass, which the first law forbids: `dM/dq = T dS/dq + 2 Omega dJ/dq > 0` for
    small `q`, since `S` is even in the spin and `J` vanishes linearly.
    """
    rows = []
    for key, states in walks.items():
        alpha_gb = float(key)
        if alpha_gb <= 0.:
            continue
        bound = 2*alpha_gb/(1 + 2*alpha_gb)
        static_mass = 3*np.pi/8*(1 + 2*alpha_gb)
        worst = max(states, key=lambda s: s['x'])
        rows.append(dict(
            alpha_gb=alpha_gb, bound=float(bound),
            q=float(worst['omega_h']), x=float(worst['x']),
            excess=float(worst['x'] - bound),
            mass_relative=float(1 - bound/worst['x']),
            mass_dip=float((static_mass - worst['E'])/static_mass),
            tail_spread_relative=float(worst['mass_tail_spread']/worst['E'])))
    rows.sort(key=lambda r: r['alpha_gb'])
    violated = [r for r in rows if r['excess'] > 0.]
    worst = max(rows, key=lambda r: r['excess'])
    summary = dict(rows=rows, violating=len(violated), couplings=len(rows),
                   worst_excess=worst['excess'],
                   worst_mass_relative=worst['mass_relative'],
                   worst_alpha_gb=worst['alpha_gb'], worst_q=worst['q'],
                   tail_spread_there=worst['tail_spread_relative'],
                   understatement=(worst['mass_relative']
                                   / max(worst['tail_spread_relative'], 1e-30)))
    print(f"  x exceeds its exact static bound at {len(violated)} of "
          f"{len(rows)} couplings; worst {worst['excess']:.2e} at "
          f"alpha_gb={worst['alpha_gb']:g}, q={worst['q']:g}, which is a "
          f"relative error of {worst['mass_relative']:.2e} in M against a "
          f"tail spread of {worst['tail_spread_relative']:.2e} there",
          flush=True)
    return summary


def tail_diagnostics(walks):
    """Worst relative value of the two internal mass diagnostics, over all states."""
    states = [s for group in walks.values() for s in group]
    spread = max(s['mass_tail_spread']/s['E'] for s in states)
    difference = max(s['mass_b_f_difference']/s['E'] for s in states)
    return dict(states=len(states), tail_spread=float(spread),
                b_f_difference=float(difference), worst=float(max(spread, difference)))


def extrapolation_models(walks):
    """Does the T=0 intercept depend on assuming analyticity in the temperature?

    The quoted uncertainty is the spread across polynomial degrees, which cannot
    see a non-analytic term if there is one. Refitting with an explicit
    `tau^{1/2}` or `tau^2 log tau` and reporting how far the intercept moves is
    what turns that into a statement rather than an assumption.
    """
    rows = []
    for coupling, states in sorted(walks.items()):
        ordered = sorted(states, key=lambda s: s['T_H'])[:12]
        temperature = np.array([s['T_H'] for s in ordered])
        values = np.array([s['psi_gb'] for s in ordered])
        scaled = np.array([s['tau'] for s in ordered])
        polynomial = np.polynomial.polynomial.polyfit(temperature, values, 3)[0]
        designs = {
            'sqrt': np.column_stack([np.ones_like(temperature), temperature,
                                     np.sqrt(temperature)]),
            'tlogt': np.column_stack([np.ones_like(temperature), temperature,
                                      temperature**2,
                                      temperature**2*np.log(temperature)]),
        }
        alternatives = {}
        for name, design in designs.items():
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            alternatives[name] = float(coefficients[0])
        spread = max(abs(value-polynomial) for value in alternatives.values())
        rows.append(dict(alpha_gb=coupling, polynomial=float(polynomial),
                         **{f'intercept_{k}': v for k, v in alternatives.items()},
                         tau_min=float(scaled.min()), tau_max=float(scaled.max()),
                         tau_range=float(scaled.max()/scaled.min()),
                         model_spread=float(spread)))
    # The sensitivity tracks the lever arm of the fit, not the coupling: a
    # sqrt(tau) term is nearly degenerate with a constant when the coldest
    # states span only a decade. Reported so that the reader can see which it is.
    short = [r for r in rows if r['tau_range'] < 50]
    long_ = [r for r in rows if r['tau_range'] >= 50]
    print(f"  intercept movement across fit models: "
          f"{max(r['model_spread'] for r in rows):.2e} worst; "
          f"{max((r['model_spread'] for r in long_), default=0):.2e} over the "
          f"{len(long_)} walks spanning more than a factor 50 in tau, "
          f"{max((r['model_spread'] for r in short), default=0):.2e} over the "
          f"{len(short)} that do not", flush=True)
    print(f"  the physically motivated tau^2 log tau term moves it by at most "
          f"{max(abs(r['intercept_tlogt']-r['polynomial']) for r in rows):.2e}",
          flush=True)
    return rows


def minimum_location(extremals, models):
    """Fit the minimum of the shift rather than quoting the lowest grid point.

    A parabola through the three couplings around the smallest measured value
    locates the minimum; the uncertainty on both the position and the value
    comes from redoing it with four and five points, and from the extrapolation
    uncertainty of the points themselves.
    """
    ordered = sorted(extremals, key=lambda r: r['y'])
    y = np.array([r['y'] for r in ordered])
    psi = np.array([r['psi_gb']['value'] for r in ordered])
    index = int(np.argmin(psi))
    by_model = {r['alpha_gb']: r['model_spread'] for r in models}
    estimates = []
    for half in (1, 2):
        window = slice(max(0, index-half), min(len(y), index+half+1))
        if window.stop-window.start < 3:
            continue
        quadratic = np.polynomial.polynomial.polyfit(y[window], psi[window], 2)
        position = -quadratic[1]/(2*quadratic[2])
        estimates.append((float(position),
                          float(np.polynomial.polynomial.polyval(position,
                                                                 quadratic))))
    positions = [p for p, _ in estimates]
    values = [v for _, v in estimates]
    uncertainty = max(ordered[index]['psi_gb']['spread'],
                      by_model.get(ordered[index]['alpha_gb'], 0.),
                      float(np.ptp(values)) if len(values) > 1 else 0.)
    return dict(grid_y=float(y[index]), grid_value=float(psi[index]),
                bracket=[float(y[max(0, index-1)]), float(y[min(len(y)-1, index+1)])],
                fitted_y=float(np.mean(positions)),
                fitted_y_spread=float(np.ptp(positions)) if len(positions) > 1 else 0.,
                fitted_value=float(np.mean(values)),
                value_uncertainty=float(uncertainty),
                suppression=float(np.pi/np.mean(values)),
                suppression_uncertainty=float(
                    np.pi*uncertainty/np.mean(values)**2))


def perturbative_limit():
    """Psi_0(q) from the linear response, against the closed-form oracle.

    No extrapolation and no finite difference enter this: at alpha_GB=0 the
    profile is the exact Myers-Perry closed form, so the response is taken at
    the exact solution. What limits the agreement is the asymptotic tail fit,
    whose spread across the three windows is reported alongside.
    """
    rows = []
    for spin in PERTURBATIVE_SPINS:
        coarse = measured_perturbative_potential(spin, resolution=CONTRAST_RESOLUTION)
        fine = measured_perturbative_potential(spin, resolution=RESOLUTION)
        oracle = float(perturbative_potential(spin))
        rows.append(dict(
            q=spin, oracle=oracle, measured=fine['psi_gb'],
            measured_n48=coarse['psi_gb'],
            resolution_difference=abs(fine['psi_gb']-coarse['psi_gb']),
            absolute_deviation=abs(fine['psi_gb']-oracle),
            relative_deviation=abs(fine['psi_gb']-oracle)/max(abs(oracle), 1e-30),
            significant_digits=float(
                -np.log10(max(abs(fine['psi_gb']-oracle)/max(abs(oracle), 1e-30),
                              1e-17)))))
    return rows


def step_refinement(cache):
    """Central differences in the coupling against the exact linear response.

    Hito 6 measured Psi_GB this way. If the linear response is the same
    quantity, halving the step must quarter the gap between them; if it is not,
    this diverges. It is the gate that lets the Hito 6 table be corrected
    rather than merely contradicted.
    """
    rows = []
    for spin, alpha_gb in REFINEMENT_POINTS:
        key = f'refine|{spin:.12g}|{alpha_gb:.12g}'
        stored = cache.get(key)
        if stored is not None:
            rows.append(stored)
            continue
        base = predictor_ladder(spin, alpha_gb, step=.005,
                                resolution=CONTRAST_RESOLUTION)
        exact = response_observables(base)['psi_gb']
        centre = measure(base)
        ladder = []
        for step in REFINEMENT_STEPS:
            if step >= alpha_gb:
                continue
            neighbours = {}
            for sign in (-1, 1):
                solution = predictor_ladder(spin, alpha_gb+sign*step, step=.005,
                                            resolution=CONTRAST_RESOLUTION)
                neighbours[sign] = measure(solution)
            derivative = {k: (neighbours[1][k]-neighbours[-1][k])/(2*step)
                          for k in ('E', 'S', 'J')}
            value = (derivative['E'] - centre['T_H']*derivative['S']
                     - 2*spin*derivative['J'])
            ladder.append(dict(step=step, psi_gb=float(value),
                               gap=float(value-exact)))
        ratios = [abs(a['gap']/b['gap']) for a, b in zip(ladder, ladder[1:])
                  if b['gap']]
        row = dict(q=spin, alpha_gb=alpha_gb, linear_response=float(exact),
                   ladder=ladder, ratios=ratios,
                   worst_ratio_deviation=float(max(abs(r-4) for r in ratios))
                   if ratios else float('inf'))
        print(f'  refinement q={spin} alpha_gb={alpha_gb}: ratios '
              f'{[round(r, 3) for r in ratios]}', flush=True)
        rows.append(cache.put(key, row))
    return rows


def resolution_contrast(cache):
    """The same walks at a coarser resolution; the difference is the budget."""
    rows = []
    for alpha_gb in (.02, .1, .5):
        fine = extremal_state(approach(alpha_gb, RESOLUTION, cache))
        coarse = extremal_state(approach(alpha_gb, CONTRAST_RESOLUTION, cache))
        rows.append(dict(
            alpha_gb=alpha_gb,
            psi_fine=fine['psi_gb']['value'], psi_coarse=coarse['psi_gb']['value'],
            psi_difference=abs(fine['psi_gb']['value']-coarse['psi_gb']['value']),
            mu_fine=fine['mu'], mu_coarse=coarse['mu'],
            mu_difference=abs(fine['mu']-coarse['mu']),
            y_fine=fine['y'], y_coarse=coarse['y']))
        print(f'  resolution alpha_gb={alpha_gb}: |Psi_ext(N64)-Psi_ext(N48)| = '
              f'{rows[-1]["psi_difference"]:.2e}', flush=True)
    return rows


def near_horizon_comparison(extremals):
    """sigma = S_ext/J against the published algebraic branch at the same y."""
    rows = []
    for record in extremals:
        y = record['y']
        if y <= 0:
            predicted = float(2*np.pi)
            coupling = 0.
        else:
            coupling = coupling_from_invariant(y)
            predicted = extremal(coupling)['S_over_J']
        rows.append(dict(
            alpha_gb=record['alpha_gb'], y=y, sigma=record['sigma'],
            sigma_spread=record['sigma_spread'], near_horizon=predicted,
            alpha_tilde=coupling,
            absolute_difference=abs(record['sigma']-predicted),
            relative_difference=abs(record['sigma']-predicted)/predicted,
            within_extrapolation_spread=bool(
                abs(record['sigma']-predicted) <= record['sigma_spread'])))
    return rows


def mass_curve(extremals):
    """mu(y) and its derivative, from masses and spins alone.

    A global polynomial fit was tried first and was wrong for a reason worth
    recording: mu(y) rises by 30% over the range while its slope falls from
    about 3.14 to about 1.07, so a single polynomial that fits the values to
    1e-3 gets the endpoint derivative wrong by 5%. The derivative of a fit is
    not the fit's accuracy.

    Secants are the right instrument here. Between two measured extremal
    states, (mu_{i+1}-mu_i)/(y_{i+1}-y_i) *is* the mean of mu' over that
    interval, with no model interposed, and its uncertainty follows from the
    two extrapolation spreads. At y=0 a one-sided quadratic through the first
    few points gives the slope, with the spread across how many points are used
    reported as its uncertainty -- and nothing at y=0 is imposed.
    """
    ordered = sorted(extremals, key=lambda r: r['y'])
    y = np.array([r['y'] for r in ordered])
    mu = np.array([r['mu'] for r in ordered])
    spread = np.array([r['mu_spread'] for r in ordered])
    gaps = np.diff(y)
    secants = np.diff(mu)/gaps
    uncertainty = (spread[1:]+spread[:-1])/gaps
    midpoints = (y[1:]+y[:-1])/2

    one_sided = {}
    for count in (3, 4, 5):
        if len(y) <= count:
            continue
        coefficients = np.polynomial.polynomial.polyfit(y[:count], mu[:count],
                                                        min(2, count-1))
        one_sided[count] = float(
            np.polynomial.polynomial.polyval(0., np.polynomial.polynomial.polyder(
                coefficients)))
    slope_zero = one_sided[min(one_sided)]
    return dict(y=y.tolist(), mu=mu.tolist(), mu_spread=spread.tolist(),
                midpoint=midpoints.tolist(), secant=secants.tolist(),
                secant_spread=uncertainty.tolist(),
                one_sided_slopes=one_sided,
                mass_coefficient=float(MASS_COEFFICIENT),
                published_shift=float(PUBLISHED_SHIFT),
                mu_at_zero=float(mu[0]),
                mu_at_zero_deviation=float(abs(mu[0]-MASS_COEFFICIENT)),
                slope_at_zero=float(slope_zero),
                slope_at_zero_spread=float(np.ptp(list(one_sided.values()))),
                slope_at_zero_deviation=float(abs(slope_zero-PUBLISHED_SHIFT)))


def route_comparison(extremals, curve):
    """The thermodynamic route against the mass-only one, as an integral.

    The comparison to make is not "two derivatives agree" but the exact
    statement behind it. The secant of `mu` across an interval *is* the
    integral of `mu'` there -- that is the fundamental theorem, not an
    approximation -- so if `Psi_GB` is `mu'`, then

        Int_{y_i}^{y_{i+1}} Psi_GB dy = mu_{i+1} - mu_i

    with no error term on the right at all. Comparing the trapezoid of `Psi`
    against that secant instead, as a first version did, measures the
    trapezoid's own O(h^2) quadrature error and calls it a disagreement; the
    "difference" it reported tracked the interval width, not the physics.

    So `Psi` is integrated with a local cubic through the four nearest points,
    whose quadrature error is O(h^4) and therefore below the measurement, and
    the trapezoid is kept alongside purely to show how much of the earlier
    number was quadrature. Everything on the left comes from entropies,
    temperatures and the first law; everything on the right comes from masses
    and spins. They share the solver and nothing else.
    """
    ordered = sorted(extremals, key=lambda r: r['y'])
    y = np.array([r['y'] for r in ordered])
    mu = np.array([r['mu'] for r in ordered])
    psi = np.array([r['psi_gb']['value'] for r in ordered])
    # The uncertainty the paper quotes and the figure draws, which is the
    # larger of the degree spread and the fit-model spread -- not the degree
    # spread alone. Comparing the routes against the narrower of the two was
    # the reason this comparison looked tenser than the error budget allowed.
    psi_spread = np.array([r.get('psi_uncertainty', r['psi_gb']['spread'])
                           for r in ordered])
    mu_spread = np.array([r['mu_spread'] for r in ordered])
    rows = []
    for index in range(len(y)-1):
        low = max(0, min(index-1, len(y)-4))
        window = slice(low, low+4)
        coefficients = np.polynomial.polynomial.polyfit(y[window], psi[window], 3)
        primitive = np.polynomial.polynomial.polyint(coefficients)
        integral = float(np.polynomial.polynomial.polyval(y[index+1], primitive)
                         - np.polynomial.polynomial.polyval(y[index], primitive))
        gap = y[index+1]-y[index]
        trapezoid = float((psi[index]+psi[index+1])/2*gap)
        change = float(mu[index+1]-mu[index])
        uncertainty = float(gap*(psi_spread[index]+psi_spread[index+1])/2
                            + mu_spread[index]+mu_spread[index+1])
        rows.append(dict(
            y_low=float(y[index]), y_high=float(y[index+1]),
            y=float((y[index]+y[index+1])/2), width=float(gap),
            covered=bool(abs(integral-change) <= uncertainty),
            integrated_psi=integral, trapezoid_psi=trapezoid,
            mass_change=change,
            difference=float(abs(integral-change)),
            trapezoid_difference=float(abs(trapezoid-change)),
            combined_spread=uncertainty,
            relative_difference=float(abs(integral-change)/max(abs(change), 1e-30)),
            goon_penco=float(integral/gap), mass_only=float(change/gap),
            goon_penco_spread=float((psi_spread[index]+psi_spread[index+1])/2),
            mass_only_spread=float((mu_spread[index]+mu_spread[index+1])/gap)))
    cumulative_psi, cumulative = 0., []
    for row in rows:
        cumulative_psi += row['integrated_psi']
        cumulative.append(dict(y=row['y_high'], integrated=float(cumulative_psi),
                               measured=float(mu[rows.index(row)+1]-mu[0]),
                               difference=float(abs(cumulative_psi
                                                    - (mu[rows.index(row)+1]-mu[0])))))
    return dict(intervals=rows, cumulative=cumulative,
                worst_difference=float(max(r['difference'] for r in rows)),
                worst_relative=float(max(r['relative_difference'] for r in rows)),
                worst_trapezoid_difference=float(max(r['trapezoid_difference']
                                                     for r in rows)),
                worst_cumulative=float(max(r['difference'] for r in cumulative)))


def extremal_anchor(extremals):
    """The alpha_GB = 0 endpoint, where two published numbers have to appear.

    Reported separately from the fit because this point is not extrapolated in
    the coupling at all: it is the Myers-Perry limit of the family, reached by
    the same walk as every other point and compared with the same two constants
    of the literature.
    """
    vacuum = min(extremals, key=lambda r: r['y'])
    return dict(y=vacuum['y'], mu=vacuum['mu'], mu_spread=vacuum['mu_spread'],
                psi=vacuum['psi_gb']['value'],
                psi_spread=vacuum['psi_gb']['spread'],
                mass_coefficient=float(MASS_COEFFICIENT),
                published_shift=float(PUBLISHED_SHIFT),
                mu_deviation=float(abs(vacuum['mu']-MASS_COEFFICIENT)),
                psi_deviation=float(abs(vacuum['psi_gb']['value']
                                        - PUBLISHED_SHIFT)),
                lowest_temperature=vacuum['lowest_temperature'])


def published_bounds(walks, extremals):
    """Our states in the scaled variables of arXiv:2303.12471, and its bounds.

    That paper reports two bounds for spherical-horizon solutions: the static
    one `alpha < 4M/(3 pi)`, i.e. `x < 1`, and that "the solutions with a
    spherical horizon topology still satisfy the Einstein gravity bound on
    angular momentum", i.e. `j <= 1`. Both are quotable statements at finite
    coupling, and both are testable on every state measured here.

    The normalisations are rebuilt from the paper's own stated conditions --
    "chosen such that t_H = a_H = s = 1 in the static limit with alpha=0, while
    the maximal value for Einstein gravity BH solutions is j = 1" -- using this
    project's closed forms, rather than transcribed from the PDF text layer.
    That makes them checkable: `j` must come out exactly 1 on our own
    alpha_GB=0 extremal state, which is a closed-form Myers-Perry point.
    """
    # Static Schwarzschild-Tangherlini at alpha=0, r_h: M = 3 pi r_h^2/8,
    # A = 2 pi^2 r_h^3, S = A/4, T = 1/(2 pi r_h).
    static_mass = 3*np.pi/8
    area_constant = static_mass**1.5/(2*np.pi**2)
    entropy_constant = static_mass**1.5/(np.pi**2/2)
    temperature_constant = 2*np.pi/np.sqrt(static_mass)
    # Extremal Myers-Perry at alpha=0: M = 3 pi a^2/2, J = pi a^3 each.
    spin_constant = (3*np.pi/2)**1.5/(2*np.pi)

    def scaled(mass, spin, entropy, temperature, area, alpha_gb):
        return dict(x=float(3*np.pi*alpha_gb/(4*mass)),
                    j=float(spin_constant*2*spin/mass**1.5),
                    a_H=float(area_constant*area/mass**1.5),
                    s=float(entropy_constant*entropy/mass**1.5),
                    t_H=float(temperature_constant*temperature*np.sqrt(mass)))

    rows = []
    for record in sorted(extremals, key=lambda r: r['y']):
        rows.append(dict(alpha_gb=record['alpha_gb'], y=record['y'],
                         **scaled(record['E']['value'], record['J']['value'],
                                  record['S']['value'], 0., 0.,
                                  record['alpha_gb'])))
    every = [scaled(s['E'], s['J'], s['S'], s['T_H'], s['A_H'], s['alpha_gb'])
             for states in walks.values() for s in states]
    finite = [r for r in rows if r['alpha_gb'] > 0]
    return dict(extremal=rows,
                vacuum_extremal_j=rows[0]['j'] if rows else None,
                worst_j=float(max(r['j'] for r in every)),
                worst_x=float(max(r['x'] for r in every)),
                # Where the published bound actually sits at finite coupling:
                # it is saturated only in the Einstein limit, and the extremal
                # family falls strictly below it as soon as the coupling is on.
                extremal_j_at_largest_coupling=float(finite[-1]['j']),
                largest_coupling=float(finite[-1]['alpha_gb']),
                extremal_j_is_monotone=bool(all(
                    b['j'] < a['j'] for a, b in zip(rows, rows[1:]))),
                spin_bound_holds=bool(all(r['j'] <= 1+1e-6 for r in every)),
                static_bound_holds=bool(all(r['x'] < 1 for r in every)))


def sign_locus(walks):
    """Where Psi_GB vanishes along each coupling, in the scaled temperature.

    Hito 6 reported this locus in `q`, which compares states at very different
    distances from extremality as the coupling changes. The first attempt to fix
    that used `tau = T_H J^{1/3}` and was worse: `tau` vanishes at both ends of
    the family -- at extremality because the temperature does, at zero spin
    because the spin does -- so scanning in `tau` found the wrong crossing. The
    measure used here is `t_H` of arXiv:2303.12471, normalised to 1 in the
    static limit and 0 at extremality, which orders the family monotonically and
    is a published convention besides.

    Only couplings whose walk covers the whole spin range are reported: locating
    a crossing between two states half a decade apart in `q` is interpolation,
    not measurement.
    """
    rows = []
    for alpha_gb, states in walks.items():
        ordered = sorted(states, key=lambda s: s['omega_h'])
        covered = (len(ordered) >= 25
                   and min(s['omega_h'] for s in ordered) <= .05)
        crossing = None
        for low, high in zip(ordered, ordered[1:]):
            if low['psi_gb']*high['psi_gb'] < 0:
                weight = low['psi_gb']/(low['psi_gb']-high['psi_gb'])
                blend = lambda key: float(low[key]+weight*(high[key]-low[key]))
                crossing = dict(
                    q=blend('omega_h'), t_scaled=blend('t_scaled'),
                    j=blend('j'), x=blend('x'), y=blend('y'), tau=blend('tau'),
                    gap=float(abs(high['omega_h']-low['omega_h'])),
                    bracket_low=dict(q=low['omega_h'], psi_gb=low['psi_gb'],
                                     t_scaled=low['t_scaled']),
                    bracket_high=dict(q=high['omega_h'], psi_gb=high['psi_gb'],
                                      t_scaled=high['t_scaled']))
                break
        rows.append(dict(alpha_gb=alpha_gb, bracketed=crossing is not None,
                         fully_covered=bool(covered), **(crossing or {})))
    return rows


def run():
    cache = Progress(PROGRESS)
    print('perturbative limit', flush=True)
    perturbative = perturbative_limit()
    worst = max(r['relative_deviation'] for r in perturbative)
    print(f'  worst relative deviation from the oracle: {worst:.2e}', flush=True)

    print('step refinement against the exact response', flush=True)
    refinement = step_refinement(cache)

    print('walks to extremality', flush=True)
    walks = {}
    for alpha_gb in EXTREMAL_COUPLINGS:
        walks[alpha_gb] = approach(alpha_gb, RESOLUTION, cache,
                                   descend=alpha_gb in MAP_COUPLINGS)
    extremals = [extremal_state(states) for states in walks.values()]

    print('resolution contrast', flush=True)
    contrast = resolution_contrast(cache)

    print('exact checks that use no extrapolation', flush=True)
    static = static_limit(cache)
    vacuum = vacuum_masses(cache)
    consistency = consistency_checks(cache)
    bound = mass_bound({f'{k:.12g}': v for k, v in walks.items()})
    tails = tail_diagnostics(walks)
    models = extrapolation_models(walks)

    by_model = {r['alpha_gb']: r['model_spread'] for r in models}
    for record in extremals:
        record['psi_uncertainty'] = float(max(record['psi_gb']['spread'],
                                              by_model.get(record['alpha_gb'], 0.)))
    curve = mass_curve(extremals)
    anchor = extremal_anchor(extremals)
    minimum = minimum_location(extremals, models)
    print(f"  minimum of the shift: {minimum['fitted_value']:.5f} +- "
          f"{minimum['value_uncertainty']:.1e} at y = {minimum['fitted_y']:.4f}",
          flush=True)
    print(f"  mu(0) = {anchor['mu']:.9f} vs published {MASS_COEFFICIENT:.9f} "
          f"(off by {anchor['mu_deviation']:.2e})", flush=True)
    print(f"  Psi_ext(0) = {anchor['psi']:.9f} vs published pi "
          f"(off by {anchor['psi_deviation']:.2e})", flush=True)
    print(f"  mu'(0) from masses alone = {curve['slope_at_zero']:.6f} "
          f"(off by {curve['slope_at_zero_deviation']:.2e}, spread "
          f"{curve['slope_at_zero_spread']:.2e})", flush=True)

    comparison = route_comparison(extremals, curve)
    horizon = near_horizon_comparison(extremals)
    print(f"  worst sigma vs near-horizon: "
          f"{max(r['relative_difference'] for r in horizon):.2e}", flush=True)
    locus = sign_locus(walks)
    bounds = published_bounds(walks, extremals)
    for row in locus:
        if row['bracketed'] and row['fully_covered']:
            print(f"  sign change at alpha_gb={row['alpha_gb']:g}: "
                  f"t_H={row['t_scaled']:.5f}, j={row['j']:.5f}, "
                  f"q={row['q']:.5f}", flush=True)
    print(f"  scaled j at alpha_gb=0 extremality: {bounds['vacuum_extremal_j']:.8f} "
          f"(must be 1); worst j over all states {bounds['worst_j']:.6f}", flush=True)

    published = [dict(alpha_tilde=v, **paper_residuals(extremal(v)))
                 for v in (0., .05, .1, .2, .4, .6, .8)]

    evidence = dict(
        perturbative=perturbative, step_refinement=refinement,
        walks={f'{k:.12g}': v for k, v in walks.items()},
        extremals=extremals, resolution_contrast=contrast,
        mass_curve=curve, route_comparison=comparison,
        extremal_anchor=anchor, static_limit=static, vacuum_masses=vacuum,
        consistency=consistency, mass_bound=bound, tail_diagnostics=tails,
        extrapolation_models=models,
        minimum=minimum,
        near_horizon=horizon, published_near_horizon=published,
        sign_locus=locus, published_bounds=bounds,
        perturbative_zero=perturbative_zero(),
        resolution=RESOLUTION, contrast_resolution=CONTRAST_RESOLUTION,
        extremal_couplings=EXTREMAL_COUPLINGS, map_couplings=MAP_COUPLINGS,
        refined_resolution=REFINED_RESOLUTION,
        tensor_gate=TENSOR_GATE, boundary_gate=BOUNDARY_GATE,
        relative_tensor_gate=RELATIVE_TENSOR_GATE)

    from rotating_bh.extremality_validation import checks
    gates = checks(evidence)
    for gate in gates:
        print(f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['name']}", flush=True)

    sources = [Path(__file__),
               ROOT/'src/rotating_bh/extremality.py',
               ROOT/'src/rotating_bh/extremality_validation.py',
               ROOT/'src/rotating_bh/gb_response.py',
               ROOT/'src/rotating_bh/egb_rotating_predictor.py',
               ROOT/'src/rotating_bh/egb_rotating_scale.py',
               ROOT/'src/rotating_bh/near_horizon.py',
               ROOT/'src/rotating_bh/_near_horizon_generated.py',
               ROOT/'src/rotating_bh/egb_rotating_bvp.py',
               ROOT/'src/rotating_bh/egb_rotating_observables.py',
               ROOT/'src/rotating_bh/egb_rotating_validation.py',
               ROOT/'src/rotating_bh/einstein.py',
               ROOT/'environment/requirements-lock.txt']
    sources += sorted((ROOT/'src/rotating_bh').glob('_egb_rotating*_generated.py'))

    data = dict(
        schema=1,
        units='r_H=G5=1; J is each spin; alpha_paper = 4 alpha_gb; '
              'y = alpha_gb/J^(2/3); tau = T_H J^(1/3); mu = E/J^(2/3); sigma = S/J',
        scope='The extremal mass function of the D=5 equal-spin Einstein-Gauss-Bonnet '
              'family at finite coupling -- the E(J) diagram left open in section 4.2 '
              'of arXiv:1010.0860v1 -- together with the conjugate potential Psi_GB '
              'over the family and its extremal limit dM_ext/dalpha|_J. NOT new: the '
              'solutions, their domain of existence, the near-horizon entropy function '
              'and S_ext (1010.0860v1 sec. 4.2), and the linear-order extremal mass '
              'M_ext = (3/2) pi^(1/3) J^(2/3) + pi alpha (arXiv:2009.00015). Those are '
              'used here as external anchors.',
        **evidence, checks=gates,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__),
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    output = ROOT/'results/egb-extremality.json'
    output.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(f'wrote {output.relative_to(ROOT)}', flush=True)
    return 0 if all(g['passed'] for g in gates) else 1


if __name__ == '__main__':
    sys.exit(run())
