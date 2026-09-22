"""Conjugate momentum of cyclic w, used only as an additional diagnostic.

This is the reduced-action first integral, without an ADM normalization.
It is not used by either solver. Symbolic identities against the original
Lagrangian and the generated raw horizon equation live in the tests.
"""


def first_integral(x, values, first, alpha_gb):
    """Return dL_eff/dw' in the compact variables, with x=1-1/r."""
    B, F, H, W = values
    V = first[3]
    z = 1-x
    A = 1+z**4*H
    C = 1+4*alpha_gb*z**2*(1-(1-z**2)*F-3*z**4*H)
    return (F/B*A)**.5*A*C*(z*V-4*W)
