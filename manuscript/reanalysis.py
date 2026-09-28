"""Reproducible sensitivity analysis of the recorded near-extremal sequences.

Central values are the production cubic-in-T intercepts from six cold states.
These envelopes quantify model/coordinate/window sensitivity, not confidence
intervals or rigorous bounds on numerical error. No solver data are modified.

`sensitivity` answers "how far does the intercept move when the fit model,
window or coordinate is changed?". `production_fit` and `leave_one_out`
answer the two questions a referee asks next: how well does the production
cubic describe the points it was given, and how much does its intercept move
when one of those points is removed. `smarr_potential` re-derives Psi from
the charges alone. All four are read back into the manuscript; none of them
converts a sensitivity into an error bar.
"""
import numpy as np

WINDOW = 6          # states in the production fit
DEGREE = 3          # cubic in T, the production model


def _cold(states, count=12):
    return sorted(states, key=lambda s: s['T_H'])[:count]


def _design(x, model):
    if model == 'sqrt':
        return np.column_stack([np.ones_like(x), x, np.sqrt(x)])
    if model == 'log':
        return np.column_stack([np.ones_like(x), x, x*x, x*x*np.log(x)])
    return np.column_stack([x**k for k in range(int(model)+1)])


def sensitivity(states, central):
    cold = _cold(states)
    values = np.array([s['psi_gb'] for s in cold])
    intercepts = {}
    for name, key in (('T', 'T_H'), ('tau', 'tau')):
        raw = np.array([s[key] for s in cold])
        for requested in (WINDOW, 12):
            count = min(requested, len(cold))
            # Scaling the fit coordinate improves conditioning and leaves
            # the span of each model, including x^2 log(x), unchanged.
            x = raw[:count]/raw[count-1]
            y = values[:count]
            designs = {
                'linear': _design(x, 1),
                'quadratic': _design(x, 2),
                'cubic': _design(x, 3),
                'sqrt': _design(x, 'sqrt'),
                'log': _design(x, 'log'),
            }
            for model, design in designs.items():
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                intercepts[f'{name}_{model}_{count}'] = float(coefficients[0])
    return dict(points=len(cold), central=float(central), intercepts=intercepts,
                envelope=float(max(abs(v-central) for v in intercepts.values())))


def production_fit(states, field='psi_gb'):
    """The production cubic itself: its coefficients, points and residuals.

    Returned in the scaled coordinate the fit uses, so that the intercept is
    `coefficients[0]` and the residuals are directly comparable with it.
    """
    cold = _cold(states, WINDOW)
    raw = np.array([s['T_H'] for s in cold])
    y = np.array([s[field] for s in cold])
    scale = raw[-1]
    x = raw/scale
    design = _design(x, DEGREE)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design@coefficients
    return dict(temperature=raw.tolist(), values=y.tolist(),
                fitted=fitted.tolist(), residuals=(y-fitted).tolist(),
                coefficients=coefficients.tolist(), scale=float(scale),
                intercept=float(coefficients[0]),
                worst_residual=float(np.max(np.abs(y-fitted))))


def leave_one_out(states, central, field='psi_gb'):
    """Refit the production model with each fitted point removed in turn.

    The referee's question is whether one cold or one warm state carries the
    intercept. The answer is the largest displacement over the six refits.
    """
    cold = _cold(states, WINDOW)
    raw = np.array([s['T_H'] for s in cold])
    y = np.array([s[field] for s in cold])
    x = raw/raw[-1]
    shifts = []
    for dropped in range(len(cold)):
        keep = np.ones(len(cold), dtype=bool)
        keep[dropped] = False
        # Same model, one point fewer: five points still over-determine a
        # cubic, so this isolates the point and does not also change the fit.
        design = _design(x[keep], DEGREE)
        coefficients, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        shifts.append(float(coefficients[0]-central))
    return dict(shifts=shifts, worst=float(max(abs(s) for s in shifts)),
                coldest_dropped=shifts[0], warmest_dropped=shifts[-1])


def matched_window(fine_states, coarse_states, field='psi_gb'):
    """The same intercept from two resolutions over a common temperature range.

    A coarse walk need not reach as cold as the production one, and when it
    stops warmer the two fits extrapolate from different distances. Comparing
    their intercepts then measures the window as well as the resolution. This
    refits both sides using only states at or above the warmer of the two floors
    and returns the difference, which isolates the resolution.

    Returns `None` when either side has fewer than `WINDOW` states left, since
    the production model would then be fitted to a different number of points.
    """
    floor = max(min(s['T_H'] for s in fine_states),
                min(s['T_H'] for s in coarse_states))
    intercepts = []
    for states in (fine_states, coarse_states):
        kept = sorted((s for s in states if s['T_H'] >= floor),
                      key=lambda s: s['T_H'])[:WINDOW]
        if len(kept) < WINDOW:
            return None
        intercepts.append(production_fit(kept, field)['intercept'])
    return dict(floor=float(floor), fine=intercepts[0], coarse=intercepts[1],
                absolute=float(abs(intercepts[0]-intercepts[1])))


def smarr_potential(state, alpha_gb):
    """Psi from the Smarr relation alone: 2M = 3TS + 6 Omega J + 2 alpha Psi.

    This uses only the charges of the single solution. It does not use the
    implicit-response solve, so it tests that solve rather than repeating it.
    Undefined at alpha_gb = 0, where the coupling term drops out.
    """
    return ((2*state['E']-3*state['T_H']*state['S']
             - 6*state['omega_h']*state['J'])/(2*alpha_gb))
