"""Reproducible sensitivity analysis of the recorded near-extremal sequences.

Central values are the production cubic-in-T intercepts from six cold states.
These envelopes quantify model/coordinate/window sensitivity, not confidence
intervals or rigorous bounds on numerical error. No solver data are modified.
"""
import numpy as np


def sensitivity(states, central):
    cold = sorted(states, key=lambda s: s['T_H'])[:12]
    values = np.array([s['psi_gb'] for s in cold])
    intercepts = {}
    for name, key in (('T', 'T_H'), ('tau', 'tau')):
        raw = np.array([s[key] for s in cold])
        for requested in (6, 12):
            count = min(requested, len(cold))
            # Scaling the fit coordinate improves conditioning and leaves
            # the span of each model, including x^2 log(x), unchanged.
            x = raw[:count]/raw[count-1]
            y = values[:count]
            designs = {
                'linear': np.column_stack([x**k for k in range(2)]),
                'quadratic': np.column_stack([x**k for k in range(3)]),
                'cubic': np.column_stack([x**k for k in range(4)]),
                'sqrt': np.column_stack([np.ones_like(x), x, np.sqrt(x)]),
                'log': np.column_stack([np.ones_like(x), x, x*x, x*x*np.log(x)]),
            }
            for model, design in designs.items():
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                intercepts[f'{name}_{model}_{count}'] = float(coefficients[0])
    return dict(points=len(cold), central=float(central), intercepts=intercepts,
                envelope=float(max(abs(v-central) for v in intercepts.values())))
