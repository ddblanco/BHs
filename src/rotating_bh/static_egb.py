"""Closed-form static (w=0, h=r^2) D=5 EGB black hole, GR-continuous branch.

f=b solves r^2 q+2 alpha_GB q^2=r_H^2+2 alpha_GB, q=1-f (arXiv:1010.0860v1
eq. (2.20), the root with a smooth alpha_GB->0 limit). This module only
evaluates that closed form; it is not derived here. The independent tensor
derivation and the check that this closed form solves the derived field
equations are in experiments/derive_static_egb.py and docs/static-egb.md.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class StaticEGB:
    r_h: float
    alpha_gb: float
    G5: float = 1.0

    def __post_init__(self):
        if not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_))
                   and np.isfinite(v) for v in (self.r_h, self.alpha_gb, self.G5)):
            raise ValueError('parameters must be finite real scalars')
        if self.r_h <= 0 or self.G5 <= 0:
            raise ValueError('r_h and G5 must be positive')
        if self.alpha_gb < 0:
            raise ValueError('this branch requires alpha_GB >= 0 for a real, smooth GR limit')

    def q(self, r):
        """1-f(r): the GR-continuous root of r^2 q+2 alpha_GB q^2=r_H^2+2 alpha_GB.

        Written as 2(r_H^2+2 alpha_GB)/(r^2*(1+sqrt(1+eps))), the rationalized
        form of the quadratic formula (multiply the naive r^2/(4a)*(sqrt(1+eps)-1)
        by its conjugate). This avoids cancelling two nearly equal numbers for
        small alpha_GB: eps=8*alpha_GB*(r_H^2+2*alpha_GB)/r^4 -> 0 makes
        sqrt(1+eps)-1 lose precision in float64 (AstraCheck, 2026-09-09), while
        the rationalized form stays accurate and needs no separate alpha_GB=0
        branch (it reduces to r_H^2/r^2 exactly when eps=0).
        """
        r = np.asarray(r, dtype=float)
        if np.any(~np.isfinite(r)) or np.any(r < self.r_h):
            raise ValueError('r must be finite and at or outside r_h')
        a = self.alpha_gb
        eps = 8*a*(self.r_h**2+2*a)/r**4
        return 2*(self.r_h**2+2*a)/(r**2*(1+np.sqrt(1+eps)))

    def functions(self, r):
        """Return b,f,h,w; b=f exactly on this branch (rr equation, see docs)."""
        f = 1-self.q(r)
        r = np.asarray(r, dtype=float)
        return {'b': f, 'f': f, 'h': r*r, 'w': np.zeros_like(r)}

    def horizon_data(self):
        """f'(r_H)=b'(r_H) from the derived ODE at q=1 (simple zero, f1>0)."""
        f1 = 2*self.r_h/(self.r_h**2+4*self.alpha_gb)
        return {'f1': f1, 'b1': f1}

    def mass_coefficient(self):
        """b=1+U/r^2+...; U=-(r_H^2+2 alpha_GB) matches the GR U=-r_H^2 at alpha_GB=0."""
        return -(self.r_h**2+2*self.alpha_gb)
