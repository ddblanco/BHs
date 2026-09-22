"""Exact linear response of the rotating EGB family in the coupling (Hito 8).

Hito 6 reached the perturbative potential `Psi_0(q)` by solving at four small
couplings and extrapolating `alpha_GB -> 0`, which agreed with the oracle of
`reports/originality-precheck.md` to three or four digits -- enough to say the
oracle survived, not enough to call it derived. The extrapolation's own
`O(alpha^2)` error was the dominant uncertainty, and it grew with the spin,
exactly where the oracle is most interesting.

Differentiating the discretised system instead removes the extrapolation
entirely. With `R(u;alpha)=0`,

    J du/dalpha = -dR/dalpha,

both sides exact to machine precision (`egb_rotating_predictor.linear_response`), so
at `alpha_GB=0` -- where `u` is the closed-form Myers-Perry profile and the
residual is already zero -- one linear solve gives `du/dalpha` with no Newton
iteration and no step to refine. Propagating it through the same observable
formulas `egb_rotating_observables.measure` uses gives `dE/dalpha`,
`dS/dalpha`, `dJ/dalpha` and hence

    Psi_0(q) = dE/dalpha - T_H dS/dalpha - 2 Omega_H dJ/dalpha

directly. `S` is the one observable with an *explicit* coupling dependence on
top of the profile's, `S = (A/4)(1+4 alpha_GB (3-H))`, and dropping that term
is the easiest way to get a wrong answer that still looks smooth, so it is
written out separately below.

The formulas here duplicate `measure`'s rather than calling it, because a
derivative is not a measurement; `tests/test_gb_response.py` pins the two
together by checking this against central differences of `measure` itself.
"""
import numpy as np
from scipy.fft import dct

from .egb_rotating_bvp import _cheb
from .egb_rotating_predictor import linear_response, solve
from .egb_rotating_saved import SavedProfile

__all__ = ['TAIL_WINDOWS', 'response_field', 'response_observables',
           'measured_perturbative_potential', 'predictor_ladder']


# The same three asymptotic fit windows measure() uses; the first is its
# primary, the other two its spread diagnostic.
TAIL_WINDOWS = [(.03, .12, 4), (.04, .16, 4), (.03, .12, 3)]


def response_field(solution, resolution=None, jacobian='analytic',
                   parameter='alpha_gb'):
    """du/dparameter on the solver's nodes, as a callable spectral profile.

    Returned as a `SavedProfile`-shaped object so the caller can evaluate the
    response and its x-derivatives anywhere, the same way a solution is used.
    `parameter` selects the coupling or the horizon angular velocity; the second
    is what the first law in the spin direction and the Goon-Penco identity need.
    """
    n = int(resolution or solution.initial_resolution)
    x, _, _ = _cheb(n)
    zz = 1-x
    flat = np.asarray(solution.evaluate(x)).ravel()
    response = linear_response(flat, n, solution.omega_h, float(solution.alpha_gb),
                               zz, jacobian, parameter=parameter).reshape(4, n)
    coefficients = dct(response, type=1, axis=1)/(n-1)
    coefficients[:, [0, -1]] *= .5
    return SavedProfile(dict(q=solution.omega_h, alpha_gb=float(solution.alpha_gb),
                             N=n, chebyshev_coefficients=coefficients.tolist()))


def _tail_coefficients(values_B, values_F, values_W, z, degree):
    """The three leading asymptotic coefficients measure() fits, on any data.

    Linear in the field values, which is exactly why the same call serves both
    the profile (giving U, U_f, W) and its derivative: the polyfit of a
    derivative is the derivative of the polyfit.
    """
    tails = [((1-z*z)*values_B-1)/(z*z), ((1-z*z)*values_F-1)/(z*z), values_W]
    return [np.polynomial.polynomial.polyfit(z*z, v, degree)[0] for v in tails]


def _tail_derivative(dB, dF, dW, z, degree):
    """The same functional applied to the response.

    The -1 of `_tail_coefficients` is a constant and so does not appear here;
    carrying it over would shift dU by a 1/z^2 that the finite-difference test
    catches at once.
    """
    tails = [(1-z*z)*dB/(z*z), (1-z*z)*dF/(z*z), dW]
    return [np.polynomial.polynomial.polyfit(z*z, v, degree)[0] for v in tails]


def response_observables(solution, response=None, window=0, jacobian='analytic',
                        parameter='alpha_gb'):
    """Derivatives of E, S and J along one parameter, and Psi_GB.

    `window` selects one of TAIL_WINDOWS; varying it is the tail-fit component
    of the Hito 8 error budget, not a free parameter to tune.

    The Wald entropy `S = (A/4)(1+4 alpha (3-H))` carries an *explicit*
    dependence on the coupling on top of the profile's, and only on the
    coupling: differentiating along `omega_h` must not pick that term up. Both
    cases are written out below rather than shared, because the difference is
    exactly the kind that survives review by looking smooth.
    """
    if parameter not in ('alpha_gb', 'omega_h'):
        raise ValueError("parameter must be 'alpha_gb' or 'omega_h'")
    if response is None:
        response = response_field(solution, jacobian=jacobian, parameter=parameter)
    lower, upper, degree = TAIL_WINDOWS[window]
    z = np.linspace(lower, upper, 80)
    B, F, H, W = solution.evaluate(1-z)
    dB, dF, dH, dW = response.evaluate(1-z)
    U, U_f, spin = _tail_coefficients(B, F, W, z, degree)
    dU, dU_f, dspin = _tail_derivative(dB, dF, dW, z, degree)

    B_H, F_H, H_H, _ = solution.evaluate(0.)
    dB_H, dF_H, dH_H, _ = response.evaluate(0.)
    if min(B_H, F_H, 1+H_H) <= 0:
        raise ValueError('nonregular horizon')
    alpha = float(solution.alpha_gb)
    area = 2*np.pi**2*np.sqrt(1+H_H)
    d_area = 2*np.pi**2*dH_H/(2*np.sqrt(1+H_H))
    wald = 1+4*alpha*(3-H_H)

    energy = -3*np.pi*U/8
    d_energy = -3*np.pi*dU/8
    spin_charge = np.pi*spin/4
    d_spin = np.pi*dspin/4
    entropy = area/4*wald
    explicit = 4*(3-H_H) if parameter == 'alpha_gb' else 0.
    d_entropy = d_area/4*wald + area/4*(explicit - 4*alpha*dH_H)
    temperature = np.sqrt(B_H*F_H)/(2*np.pi)
    psi = d_energy - temperature*d_entropy - 2*solution.omega_h*d_spin
    return dict(omega_h=float(solution.omega_h), alpha_gb=alpha, window=window,
                parameter=parameter,
                E=float(energy), J=float(spin_charge), S=float(entropy),
                T_H=float(temperature), A_H=float(area),
                dE=float(d_energy), dJ=float(d_spin), dS=float(d_entropy),
                dA=float(d_area), dU=float(dU), dU_f=float(dU_f),
                mass_b_f_difference=float(3*np.pi/8*abs(dU-dU_f)),
                psi_gb=float(psi))


def measured_perturbative_potential(q, resolution=32, jacobian='analytic'):
    """Psi_0(q): the potential at alpha_GB=0, with no extrapolation anywhere.

    The seed is the closed-form Myers-Perry profile, so the solve costs zero
    Newton iterations and the response is taken at the exact vacuum solution.
    """
    from .gb_potential import VacuumSeed

    solution = solve(q, 0., resolution=resolution, tol=1e-11,
                     previous=VacuumSeed(q), jacobian=jacobian)
    return response_observables(solution, jacobian=jacobian)


def predictor_ladder(omega_h, alpha_gb, step=.005, resolution=32,
                     jacobian='analytic', seed='predictor', tol=1e-11,
                     max_iterations=50):
    """Climb from the vacuum to a coupling, seeding each rung by linear response.

    Deliverable 1 of `plans/2026-09-14-hito-8.md`. Continuation is upward only,
    for the reason `gb_potential.reach` records; what changes is the direction
    each rung starts in, not the ladder.
    """
    from .gb_potential import VacuumSeed

    if alpha_gb < 0:
        raise ValueError('alpha_gb must be non-negative')
    rungs = (np.linspace(0., alpha_gb, int(np.ceil(round(alpha_gb/step, 9)))+1)
             if alpha_gb > 0 else np.array([0.]))
    current = solve(omega_h, 0., resolution=resolution, tol=tol,
                    previous=VacuumSeed(omega_h), jacobian=jacobian,
                    max_iterations=max_iterations)
    for value in rungs[1:]:
        current = solve(omega_h, float(value), resolution=resolution, tol=tol,
                        previous=current, jacobian=jacobian, seed=seed,
                        max_iterations=max_iterations)
    return current
