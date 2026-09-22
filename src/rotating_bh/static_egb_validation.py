"""Off-collocation checks of represented static EGB solutions, independent of
the ODE used to obtain them.
"""
from functools import lru_cache

import numpy as np
import sympy as sp

from .einstein import compile_metric, curvature, gauss_bonnet, metric_from_ansatz
from .static_egb import StaticEGB
from ._static_egb_generated import rhs


def physical_jets(solution, r):
    """f=b and its first two r-derivatives, from the represented F(x)."""
    r = np.asarray(r, dtype=float)
    x = 1-1/r
    F = solution.evaluate(x)
    Fx = solution.evaluate(x, 1)
    Fxx = solution.evaluate(x, 2)
    Fr = Fx/r**2
    Frr = Fxx/r**4-2*Fx/r**3
    return np.asarray([F, Fr, Frr])


@lru_cache(maxsize=1)
def _metric_jets():
    t, r, theta, phi1, phi2 = sp.symbols('t r theta phi1 phi2', real=True)
    R = sp.symbols('R', positive=True)
    f0, f1, f2 = sp.symbols('f0 f1 f2')
    f = f0+f1*(r-R)+f2*(r-R)**2/2
    return compile_metric(metric_from_ansatz(r, theta, f, f, r**2, 0),
                           (t, r, theta, phi1, phi2), (R, f0, f1, f2))


def tensor_residual(solution, r, alpha_gb, theta=0.6):
    """Full G+alpha_GB H (all diagonal components) on the represented b=f=F
    metric; only the tt component was ever collocated (rr, theta-theta are
    an independent check of the numerical solution, not of the derivation).
    """
    F, Fr, Frr = physical_jets(solution, np.array([r]))[:, 0]
    arrays = _metric_jets()(0, r, theta, 0, 0, r, F, Fr, Frr)
    result = curvature(*arrays)
    gb = gauss_bonnet(arrays[0], result['inverse'], result['riemann'],
                       result['ricci'], result['scalar'])
    field = result['einstein']+alpha_gb*gb['H']
    scales = np.array([1, 1, r, r, r])
    return float(np.max(np.abs(field/np.outer(scales, scales))))


def check_points(solution, count=201):
    # A fixed interior sample away from both the solver's nodes and x=1.
    lower, upper = 0.005, min(0.995, 1-solution.cutoff-0.005)
    step = (upper-lower)/count
    points = lower+(np.arange(count)+.5)*step
    for i in range(count):
        while np.min(np.abs(points[i]-solution.nodes)) < 1e-12:
            points[i] += step/8
    return points


def diagnose(solution, alpha_gb):
    x = check_points(solution)
    r = 1/(1-x)
    oracle = StaticEGB(r_h=1.0, alpha_gb=alpha_gb).functions(r)['f']
    represented = solution.evaluate(x)
    profile_error = float(np.max(np.abs(represented-oracle)))

    # The represented derivative (Chebyshev polynomial for spectral, finite
    # differences of the dense-output spline for adaptive; see
    # static_egb_bvp.py) is independent of the RHS formula; comparing the two
    # off the collocation/mesh nodes is a real check of the represented curve.
    represented_derivative = solution.evaluate(x, 1)
    expected_derivative = rhs(1-x, represented, alpha_gb)
    ode_residual = float(np.max(np.abs((represented_derivative-expected_derivative)*x*(1-x))))

    tensors = [tensor_residual(solution, radius, alpha_gb, theta)
               for radius in (1.05, 1.3, 2., 5., 20.)
               for theta in (.2, .7, 1.2)
               if solution.cutoff <= 1-1/radius <= 1-solution.cutoff]
    coverage = len(tensors) == 15
    einstein_residual = max(tensors) if tensors else float('inf')

    positive = bool(np.all(np.isfinite(represented)) and np.all(represented > -1e-9)
                    and np.all(represented < 1+1e-6))
    horizon_slope = float(solution.evaluate(0., 1))
    simple_horizon = bool(np.isfinite(horizon_slope) and horizon_slope > 0)

    measurements = dict(
        profile_error=profile_error, ode_residual=ode_residual,
        einstein_residual=einstein_residual, positive_exterior=positive,
        tensor_sample_coverage=coverage, simple_horizon=simple_horizon,
        horizon_slope=horizon_slope,
        horizon_condition=float(abs(solution.evaluate(0.))))
    measurements['accepted'] = bool(
        positive and coverage and simple_horizon
        and np.isfinite(profile_error) and profile_error < 1e-6
        and np.isfinite(ode_residual) and ode_residual < 1e-6
        and np.isfinite(einstein_residual) and einstein_residual < 1e-6
        and measurements['horizon_condition'] < 1e-8)
    return measurements
