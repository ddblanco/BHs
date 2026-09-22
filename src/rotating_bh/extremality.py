"""The extremal limit of the rotating EGB family at finite coupling (Hito 8).

Deliverables 3, 4 and 6 of `plans/2026-09-14-hito-8.md`. Three things happen
here that Hito 6 could not do:

**The potential is exact at every point.** Hito 6 measured
`Psi_GB = dE/dalpha - T dS/dalpha - 2 Omega dJ/dalpha` by central differences
in the coupling, so each value carried an `O(h^2)` truncation error that grew
towards small coupling and high spin -- up to `1.5e-2` absolute at
`q=0.65, alpha_GB=0.02`. `gb_response.response_observables` gets the same
derivatives from one linear solve, with no step at all. The two agree, and the
central differences converge to the linear response at exactly second order
over five halvings (`checks/hito-8.ps1`), which is what licenses replacing one
by the other rather than averaging them.

**Extremality is reachable.** With the two-parameter predictor seeding of
`egb_rotating_bvp`, the continuation walks in the spin as well as the coupling
and reaches `T_H` of order `1e-5` in units where the static value is `O(0.1)`
-- that is, five decades below the scale, at every coupling tried up to 0.5.
The `q = 0.68` wall of Hito 6 and the `q = 0.70` wall of its addendum were both
seeding artefacts; nothing physical lives there.

**The extremal shift is measured twice, independently.** Route one is
thermodynamic: `Psi_GB` extrapolated to `T -> 0`, which by Goon-Penco is
`dM_ext/dalpha_GB|_J`. Route two uses masses and spins only -- no entropy, no
temperature, no Wald formula, no first law. Scaling fixes
`M_ext = J^{2/3} mu(y)` with `y = alpha_GB/J^{2/3}`, so the measured
`(y, E/J^{2/3})` pairs *are* `mu`, and `dM_ext/dalpha|_J = mu'(y)` follows by
differentiating it. The two routes share the solver and nothing else.

The external anchor is `mu'(0) = pi`, published in arXiv:2009.00015 as
`M_ext = (3/2) pi^{1/3} J^{2/3} + pi alpha` to linear order. It is not
reproduced by construction anywhere in this module.
"""
import numpy as np

from .egb_rotating_predictor import solve
from .egb_rotating_observables import measure
from .egb_rotating_scale import scaled_diagnose
from .gb_response import predictor_ladder, response_field, response_observables

__all__ = ['TENSOR_GATE', 'BOUNDARY_GATE', 'RELATIVE_TENSOR_GATE',
           'MASS_COEFFICIENT', 'PUBLISHED_SHIFT', 'state_at', 'invariants',
           'walk', 'walk_to_extremality', 'extrapolate', 'extremal_state',
           'smarr_residual']


# The Hito 4 numbers, kept and recorded for every point.
TENSOR_GATE, BOUNDARY_GATE = 1e-6, 1e-8
# Approaching extremality the *absolute* tensor gate stops being the right
# instrument: |G| and |alpha_GB H| each grow past 100 near the horizon while
# their sum stays at 1e-6, so a solution accurate to one part in 1e9 gets
# rejected for a reason that has nothing to do with its accuracy
# (egb_rotating_validation.tensor_parts). The gate applied here is therefore the
# relative residual at 1e-8 -- which, wherever the curvature scale is at most
# 100, is *stricter* than the absolute 1e-6 it replaces, and that covers every
# point of the moderate-spin grid. Both numbers and the scale are recorded for
# every state, so the substitution can be audited rather than believed.
RELATIVE_TENSOR_GATE = 1e-8
# M_ext = (3/2) pi^{1/3} J^{2/3} + pi alpha_GB + O(alpha^2), arXiv:2009.00015.
MASS_COEFFICIENT = 1.5*np.pi**(1/3)
PUBLISHED_SHIFT = np.pi


def state_at(solution, count=51):
    """Observables, exact coupling derivatives and the Hito 4 gates at a point."""
    checked = scaled_diagnose(solution, solution.alpha_gb, count=count)
    if not (checked['max_relative_tensor_residual'] < RELATIVE_TENSOR_GATE
            and solution.boundary_residual < BOUNDARY_GATE):
        raise RuntimeError(f'unaccepted solution at omega_h={solution.omega_h}, '
                           f'alpha_gb={solution.alpha_gb}: {checked}')
    observed = measure(solution)
    field = response_field(solution)
    derivatives = response_observables(solution, field)
    windows = [response_observables(solution, field, window=w)['psi_gb']
               for w in range(3)]
    return dict(omega_h=float(solution.omega_h), alpha_gb=float(solution.alpha_gb),
                resolution=int(solution.initial_resolution),
                E=observed['E'], J=observed['J'], S=observed['S'],
                T_H=observed['T_H'], A_H=observed['A_H'],
                psi_gb=derivatives['psi_gb'],
                dE=derivatives['dE'], dS=derivatives['dS'], dJ=derivatives['dJ'],
                psi_window_spread=float(np.ptp(windows)),
                mass_tail_spread=observed['mass_tail_spread'],
                spin_tail_spread=observed['spin_tail_spread'],
                mass_b_f_difference=observed['mass_b_f_difference'],
                max_tensor_residual=checked['max_tensor_residual'],
                max_relative_tensor_residual=checked['max_relative_tensor_residual'],
                tensor_scale=checked['max_tensor_scale'],
                absolute_gate_passed=bool(checked['max_tensor_residual'] < TENSOR_GATE),
                boundary_residual=float(solution.boundary_residual),
                smarr=smarr_residual(observed, solution.omega_h,
                                     solution.alpha_gb, derivatives['psi_gb']))


def smarr_residual(observables, omega_h, alpha_gb, psi_gb):
    """2E = 3 T_H S + 6 Omega_H J + 2 alpha_GB Psi_GB, as in Hito 6."""
    left = 2*observables['E']
    right = (3*observables['T_H']*observables['S'] + 6*omega_h*observables['J']
             + 2*alpha_gb*psi_gb)
    return float(abs(left-right)/abs(left))


# The normalisations of arXiv:2303.12471 eqs. (2.7)-(2.8), rebuilt from that
# paper's own stated conditions rather than transcribed: "chosen such that
# t_H = a_H = s = 1 in the static limit with alpha=0, while the maximal value
# for Einstein gravity BH solutions is j = 1". Using this project's closed
# forms -- static Schwarzschild-Tangherlini M = 3 pi r_h^2/8, A = 2 pi^2 r_h^3,
# T = 1/(2 pi r_h); extremal Myers-Perry M = 3 pi a^2/2, J = pi a^3 each -- the
# four constants are fixed with nothing left to choose, and `j = 1` on our own
# alpha_GB=0 extremal state is then a check rather than a definition.
_STATIC_MASS = 3*np.pi/8
_AREA_CONSTANT = _STATIC_MASS**1.5/(2*np.pi**2)
_ENTROPY_CONSTANT = _STATIC_MASS**1.5/(np.pi**2/2)
_TEMPERATURE_CONSTANT = 2*np.pi/np.sqrt(_STATIC_MASS)
_SPIN_CONSTANT = (3*np.pi/2)**1.5/(2*np.pi)


def invariants(state):
    """The scale-invariant labels of a state, in two complementary sets.

    The system has one scaling symmetry, `r_H -> lambda r_H` with
    `alpha_GB -> lambda^2 alpha_GB`, under which `E, alpha ~ length^2`,
    `J, S ~ length^3` and `T ~ 1/length`. Every statement that is about physics
    rather than about a choice of units is a statement about invariants of it.

    The first set is normalised by the **spin**,

        y = alpha_GB/J^{2/3},  tau = T_H J^{1/3},
        mu = E/J^{2/3},        sigma = S/J,

    and is the right one for the extremal mass function, because `M_ext` at
    fixed `J` is exactly `J^{2/3} mu(y)` and `dM_ext/dalpha = mu'(y)`.

    The second is normalised by the **mass**, and is the convention of
    arXiv:2303.12471: `x` for the coupling, `j` for the spin, `t_H` for the
    temperature, `a_H` and `s` for area and entropy. It is used wherever a
    statement has to order the family from static to extremal, because `tau`
    cannot: `tau` vanishes at *both* ends -- at extremality because `T_H` does,
    and at zero spin because `J` does -- so it does not order anything. `t_H`
    runs monotonically from 1 in the static limit to 0 at extremality, and `j`
    from 0 to 1. Reporting the sign-change locus in `t_H` is also what makes it
    directly comparable with the published domain of existence.
    """
    spin, mass = state['J'], state['E']
    return dict(y=float(state['alpha_gb']/spin**(2/3)),
                tau=float(state['T_H']*spin**(1/3)),
                mu=float(mass/spin**(2/3)),
                sigma=float(state['S']/spin),
                x=float(3*np.pi*state['alpha_gb']/(4*mass)),
                j=float(_SPIN_CONSTANT*2*spin/mass**1.5),
                t_scaled=float(_TEMPERATURE_CONSTANT*state['T_H']*np.sqrt(mass)),
                a_scaled=float(_AREA_CONSTANT*state['A_H']/mass**1.5),
                s_scaled=float(_ENTROPY_CONSTANT*state['S']/mass**1.5))


def _advance(current, spin, alpha_gb, step, resolution):
    return solve(spin+step, alpha_gb, resolution=resolution, tol=1e-11,
                 previous=current, seed='predictor', jacobian='analytic')


def walk(base, alpha_gb, *, direction=1, resolutions=(64, 96), spin_step=.01,
         minimum_step=1e-5, floor=None, stop=None, spin_record=.02,
         temperature_record=1.3, count=51, patience=4):
    """Walk the spin from an accepted solution, measuring on a schedule.

    Three behaviours, each answering a way an earlier version failed:

    *Halve, do not stop.* At this distance from extremality a Newton that does
    not converge is a seeding accident far more often than a boundary, and the
    whole point of the Hito 6 addendum was that a wall which looked physical was
    not. What ends the walk is the step collapsing below `minimum_step`.

    *Roll back on a refused state.* Without that, one state that misses the gate
    leaves the walk standing on an unaccepted profile, every later seed inherits
    it, and the walk dies nine halvings later far short of where it could have
    reached -- which is what the first production run did at the two smallest
    couplings.

    *Raise the resolution before shortening the step.* Approaching extremality
    the independent tensor residual grows sharply -- at `alpha_GB=0.005` it goes
    from 5e-8 at `T=4e-3` to 2e-5 at `T=2e-4` -- and the cure is resolution, not
    a smaller step: the same point at N=96 comes back at 3e-7, and the
    observables themselves move by only 1e-6 between the two. So a refused state
    is retried at the next resolution before the step is touched, and the
    resolution that accepted it is recorded with it.

    Measuring at every accepted step would spend most of the run in `diagnose`;
    measuring on a fixed spin grid would crowd the samples where nothing
    changes. The schedule keeps a state whenever the spin has moved by
    `spin_record` *or* the temperature has changed by `temperature_record`.
    """
    resolutions = tuple(resolutions)
    states, current = [], base
    level = 0
    spin = float(base.omega_h)
    step = spin_step*direction
    last_spin, last_temperature = spin, None
    failures = 0
    while abs(step) > minimum_step:
        try:
            candidate = _advance(current, spin, alpha_gb, step, resolutions[level])
            observed = measure(candidate)
        except Exception:
            step /= 2
            failures += 1
            continue
        temperature = observed['T_H']
        due = (abs(spin+step-last_spin) >= spin_record or last_temperature is None
               or not (1/temperature_record < temperature/last_temperature
                       < temperature_record))
        if due:
            measured, refined = None, candidate
            for attempt in range(level, len(resolutions)):
                try:
                    if attempt != level:
                        refined = solve(spin+step, alpha_gb,
                                        resolution=resolutions[attempt], tol=1e-11,
                                        previous=candidate, jacobian='analytic')
                    measured = state_at(refined, count=count)
                    level = attempt
                    candidate = refined
                    break
                except Exception:
                    continue
            if measured is None:
                step /= 2
                failures += 1
                if failures > patience and not states:
                    break
                continue
            measured['resolution'] = int(resolutions[level])
            states.append(measured)
            last_spin, last_temperature = spin+step, temperature
        current, spin = candidate, spin+step
        failures = 0
        if stop is not None and stop(spin, observed):
            break
        if floor is not None and temperature < floor:
            break
    return states


def walk_to_extremality(alpha_gb, *, start=.60, resolutions=(64, 96),
                        coupling_step=.005, down_to=0., **kwargs):
    """Both directions from one climb: down to zero spin, up to extremality."""
    base = predictor_ladder(start, alpha_gb, step=coupling_step,
                            resolution=resolutions[0])
    anchor = state_at(base)
    descending = walk(base, alpha_gb, direction=-1, resolutions=resolutions,
                      stop=lambda spin, _: spin <= down_to, **kwargs)
    ascending = walk(base, alpha_gb, direction=+1, resolutions=resolutions,
                     floor=1e-6, **kwargs)
    return sorted(descending+[anchor]+ascending, key=lambda s: s['omega_h'])


def extrapolate(temperatures, values, degrees=(1, 2, 3)):
    """Fit in the temperature and read the T=0 intercept, with a spread.

    The uncertainty reported is the spread across polynomial degrees, which is
    an honest sensitivity of the extrapolation to its own model and not a
    statistical error bar. A value whose degrees disagree is a value that has
    not converged, and the caller is expected to say so rather than average.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(temperatures)
    temperatures, values = temperatures[order], values[order]
    intercepts = []
    for degree in degrees:
        if len(temperatures) < degree+1:
            continue
        use = slice(0, max(degree+1, min(len(temperatures), degree+3)))
        coefficients = np.polynomial.polynomial.polyfit(
            temperatures[use], values[use], degree)
        intercepts.append(float(coefficients[0]))
    if not intercepts:
        raise ValueError('not enough points to extrapolate')
    return dict(value=intercepts[-1], spread=float(np.ptp(intercepts)),
                intercepts=intercepts, points=int(len(temperatures)))


def extremal_state(states, keys=('E', 'J', 'S', 'psi_gb'), window=12):
    """Every listed observable extrapolated to T=0 along one approach.

    Only the `window` coldest states enter the fit. The approach spans two or
    three decades in temperature and the warm end is nowhere near the linear
    regime, so including it would let a far-away point set the intercept.
    """
    states = sorted(states, key=lambda s: s['T_H'])[:window]
    temperatures = [s['T_H'] for s in states]
    result = {key: extrapolate(temperatures, [s[key] for s in states])
              for key in keys}
    result['alpha_gb'] = states[0]['alpha_gb']
    result['lowest_temperature'] = float(min(temperatures))
    result['highest_spin'] = float(max(s['omega_h'] for s in states))
    spin = result['J']['value']
    result['y'] = float(result['alpha_gb']/spin**(2/3))
    result['mu'] = float(result['E']['value']/spin**(2/3))
    result['sigma'] = float(result['S']['value']/spin)
    # The spreads propagated to the invariants, so a reader never has to
    # reconstruct them from the raw intercepts.
    result['y_spread'] = float(abs(result['y'])*(2/3)*result['J']['spread']/spin)
    result['mu_spread'] = float(
        result['E']['spread']/spin**(2/3)
        + abs(result['mu'])*(2/3)*result['J']['spread']/spin)
    result['sigma_spread'] = float(
        result['S']['spread']/spin + abs(result['sigma'])*result['J']['spread']/spin)
    result['worst_tensor_residual'] = float(max(s['max_tensor_residual']
                                                for s in states))
    result['worst_relative_tensor_residual'] = float(
        max(s['max_relative_tensor_residual'] for s in states))
    result['states_passing_the_absolute_gate'] = int(
        sum(s['absolute_gate_passed'] for s in states))
    result['states'] = int(len(states))
    result['worst_smarr'] = float(max(s['smarr'] for s in states))
    return result
