"""The reserved tt/b field equation for the static EGB ansatz (b=f, h=r^2, w=0).

Derived in experiments/derive_static_egb.py directly from the tensor (not a
reduced action); see docs/static-egb.md. `field_residual` is proportional to
G_{tt}+alpha_GB H_{tt} (factor 2r^3/(3b), always positive outside the
horizon) and vanishes only on shell. It exists so tests can cross-check the
generated ODE (`_static_egb_generated.rhs`) against the independent,
generic-profile tensor evaluator in `einstein.py`, exactly as Hito 2 did for
the vacuum system.
"""
import numpy as np


def field_residual(r, f, fp, alpha_gb):
    """Proportional to G_tt+alpha_GB H_tt; zero iff f solves the static ODE."""
    r, f, fp = (np.asarray(v, dtype=float) for v in (r, f, fp))
    return 4*alpha_gb*(f-1)*fp-r*(r*fp+2*f-2)
