"""Equal-spin D=5 Myers--Perry, gauge g=r²; arXiv:1010.0860 eqs. 2.17--19.

Only the nonextremal exterior is supported. G5 is the five-dimensional
Newton constant; J1 and J2 are separate angular momenta, not their sum.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class MyersPerry:
    r_h: float
    omega_h: float
    G5: float = 1.0

    def __post_init__(self):
        if not all(isinstance(v, Real) and not isinstance(v, (bool,np.bool_))
                   and np.isfinite(v) for v in (self.r_h,self.omega_h,self.G5)):
            raise ValueError('parameters must be finite real scalars')
        if self.r_h <= 0 or self.G5 <= 0:
            raise ValueError('r_h and G5 must be positive')
        if abs(self.r_h*self.omega_h) >= 1/np.sqrt(2):
            raise ValueError('outer nonextremal horizon requires |r_h*omega_h| < 1/sqrt(2)')

    def functions(self, r):
        """Return b,f,h,w for finite scalar/array radii r >= r_h.

        Factorization of f avoids subtraction of nearly equal terms at r_h.
        The radial functions are finite there; the coordinate metric is not.
        """
        r = np.asarray(r, dtype=float)
        if np.any(~np.isfinite(r)) or np.any(r < self.r_h):
            raise ValueError('r must be finite and at or outside r_h')
        z = self.r_h/r
        q2 = (self.r_h*self.omega_h)**2
        squashing = 1+q2*z**4/(1-q2)
        f = (1-z)*(1+z)*(1-q2*z*z/(1-q2))
        return {'b': f/squashing, 'f': f, 'h': r*r*squashing,
                'w': self.omega_h*z**4/(1-q2+q2*z**4)}

    def horizon_data(self):
        """Analytic simple-zero derivatives and horizon squashing."""
        q2 = (self.r_h*self.omega_h)**2
        return {'b1': 2*(1-2*q2)/self.r_h,
                'f1': 2*(1-2*q2)/(self.r_h*(1-q2)),
                'h_H': self.r_h**2/(1-q2)}

    def asymptotic_coefficients(self):
        """b=1+U/r²+..., f=1+U/r²+V/r⁴, w=W/r⁴+... ."""
        denominator = 1-(self.r_h*self.omega_h)**2
        return {'U': -self.r_h**2/denominator,
                'V': self.r_h**6*self.omega_h**2/denominator,
                'W': self.r_h**4*self.omega_h/denominator}

    def thermodynamics(self):
        """Mass, each spin, angular velocity, temperature, area and GR entropy."""
        c, horizon = self.asymptotic_coefficients(), self.horizon_data()
        angular_momentum = np.pi*c['W']/(4*self.G5)
        area = 2*np.pi**2*self.r_h**2*np.sqrt(horizon['h_H'])
        return {'M': -3*np.pi*c['U']/(8*self.G5),
                'J1': angular_momentum, 'J2': angular_momentum,
                'Omega_H': self.omega_h,
                'T_H': np.sqrt(horizon['b1']*horizon['f1'])/(4*np.pi),
                'A_H': area, 'S': area/(4*self.G5)}
