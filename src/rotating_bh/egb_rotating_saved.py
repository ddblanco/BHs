"""Reconstruct explicitly stored spectral polynomials for further checks.

Loading is not physical acceptance: callers must verify artifact hashes and
independent residuals before treating a stored profile as a solution.
"""
import numpy as np
from numpy.polynomial import Chebyshev


class SavedProfile:
    def __init__(self, record):
        coefficients = np.asarray(record['chebyshev_coefficients'],dtype=float)
        if coefficients.shape!=(4,record['N']) or not np.all(np.isfinite(coefficients)):
            raise ValueError('invalid spectral coefficient array')
        self.omega_h = record['q']
        self.alpha_gb = record['alpha_gb']
        self.nodes = (1+np.cos(np.pi*np.arange(record['N'])/(record['N']-1)))/2
        polynomials = [Chebyshev(c,domain=[0,1]) for c in coefficients]
        self._derivatives = [[p.deriv(k) for p in polynomials] for k in range(3)]

    def evaluate(self,x,derivative=0):
        x = np.asarray(x,dtype=float)
        if derivative not in (0,1,2) or np.any(~np.isfinite(x)) or np.any((x<0)|(x>1)):
            raise ValueError('invalid compact coordinate or derivative order')
        return np.asarray([p(x) for p in self._derivatives[derivative]])
