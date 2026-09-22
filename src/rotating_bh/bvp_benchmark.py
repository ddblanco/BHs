"""Compare numerical solutions of y'' = -pi**2 sin(pi*x), y(0)=y(1)=0.

The adaptive representation integrates the computed velocity spline and
corrects its endpoint drift linearly. Its second derivative is the derivative
of that spline, never a substitution of the known right-hand side.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.fft import dct
from scipy.integrate import solve_bvp

ADAPTIVE_TOL = 1e-10
ADAPTIVE_MAX_NODES = 10000
CHECK_POINT_COUNT = 1001


@dataclass
class BVPSolution:
    initial_resolution: int
    nodes: np.ndarray
    _evaluate: Callable

    @property
    def effective_resolution(self):
        return len(self.nodes)

    @property
    def check_points(self):
        # Midpoints are independent of the solver meshes. Move coincidences
        # (e.g. x=1/2) into the same checking cell, away from collocation.
        points = (np.arange(CHECK_POINT_COUNT) + 0.5) / CHECK_POINT_COUNT
        for i, point in enumerate(points):
            while np.min(np.abs(point - self.nodes)) <= 1e-12:
                point += 0.125 / CHECK_POINT_COUNT
            points[i] = point
        return points

    def evaluate(self, x, derivative=0):
        """Evaluate this approximation or its actual first/second derivative."""
        if derivative not in (0, 1, 2):
            raise ValueError("derivative must be 0, 1 or 2")
        return self._evaluate(np.asarray(x), derivative)

    @property
    def residual(self):
        x = self.check_points
        return float(np.max(np.abs(self.evaluate(x, derivative=2)
                                   + np.pi**2 * np.sin(np.pi * x))))


def _validate_resolution(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 3:
        raise ValueError("resolution must be an integer >= 3")


def solve_adaptive(n, *, tol=ADAPTIVE_TOL, max_nodes=ADAPTIVE_MAX_NODES):
    """Solve a first-order system starting with n uniformly spaced nodes.

    Returned y is the integral of solve_bvp's velocity cubic, with a linear
    endpoint correction. This yields a C2 piecewise quartic representation;
    the raw cubic representation of y is not used for reported errors.
    """
    _validate_resolution(n)
    nodes = np.linspace(0.0, 1.0, n)
    result = solve_bvp(
        lambda x, u: np.vstack((u[1], -np.pi**2 * np.sin(np.pi*x))),
        lambda left, right: np.array([left[0], right[0]]),
        nodes, np.zeros((2, n)), tol=tol, max_nodes=max_nodes,
    )
    if not result.success:
        raise RuntimeError(f"adaptive solver did not converge: {result.message}")
    integral = result.sol.antiderivative()
    start = integral(0.0)[1]
    drift = integral(1.0)[1] - start

    def evaluate(x, derivative):
        if derivative == 0:
            return integral(x)[1] - start - x*drift
        if derivative == 1:
            return result.sol(x)[1] - drift
        return result.sol(x, 1)[1]

    return BVPSolution(n, result.x, evaluate)


def solve_spectral(n):
    """Use n Chebyshev-Lobatto nodes, a dense D2 and Dirichlet rows."""
    _validate_resolution(n)
    z = np.cos(np.pi*np.arange(n)/(n-1))
    weights = np.ones(n)
    weights[[0, -1]] = 2.0
    weights *= (-1.0)**np.arange(n)
    differences = z[:, None] - z[None, :]
    d = (weights[:, None]/weights[None, :])/(differences + np.eye(n))
    d -= np.diag(np.sum(d, axis=1))
    nodes = (z+1.0)/2.0
    matrix = 4.0*(d @ d)
    rhs = -np.pi**2*np.sin(np.pi*nodes)
    matrix[[0, -1], :] = 0.0
    matrix[0, 0] = matrix[-1, -1] = 1.0
    rhs[[0, -1]] = 0.0
    values = np.linalg.solve(matrix, rhs)
    # Lobatto interpolation coefficients from a discrete cosine transform;
    # avoid a least-squares fit that amplifies roundoff in high derivatives.
    coefficients = dct(values, type=1)/(n-1)
    coefficients[[0, -1]] *= 0.5
    polynomial = Chebyshev(coefficients, domain=[0, 1])
    derivatives = [polynomial, polynomial.deriv(1), polynomial.deriv(2)]
    return BVPSolution(n, nodes, lambda x, derivative: derivatives[derivative](x))


def max_error(solution):
    """Maximum sampled error on 1001 off-collocation points (not a bound)."""
    x = solution.check_points
    return float(np.max(np.abs(solution.evaluate(x) - np.sin(np.pi*x))))
