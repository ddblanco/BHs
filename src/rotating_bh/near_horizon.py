"""The extremal branch from its near-horizon geometry (Hito 8).

**This reproduces a published calculation, and is used as an external anchor
rather than offered as a new result.** Section 4.2 of arXiv:1010.0860v1 -- the
benchmark this whole project reconstructs -- already derives the extremal
near-horizon geometry of the equal-spin D=5 EGB family by Sen's entropy
function, in the same ansatz, and gives its algebraic system in closed form
(their eqs. 4.6-4.20). The Hito 8 plan listed a near-horizon route as an
"independent confirmation"; the literature review that same plan required
before the manuscript found it is independent of *this project's solver* but
not new, and that is recorded here rather than in a footnote.

That changes the module's job, and for the better. A route invented here would
confirm the numerics against ourselves. A route published in 2010, rederived
from scratch and agreeing with the spectral solver's `T -> 0` limit, confirms
them against someone else -- at *finite* coupling, which is exactly what Hito
4C could not achieve with its ergosurface comparison.

So `experiments/derive_near_horizon.py` derives the entropy function
independently (its own symbolic curvature, gated against `einstein.py`), and
then checks that the resulting extremal state satisfies eqs. (4.11) and (4.12)
of 1010.0860v1 as published, in the paper's own variables. It does, which
simultaneously reproduces their algebra and pins the coupling normalisation
`alpha_paper = 4 alpha_GB`.

What this module gives and what it does not:

* It gives `S_ext` and `J` on the extremal branch exactly, hence the
  dimensionless curve `S/J` against `y = alpha_GB/J^{2/3}`, with no
  discretisation, no continuation and no tail fit.
* It does **not** give `M_ext`. The near-horizon geometry of an extremal black
  hole determines its entropy as a function of its charges and nothing else;
  the asymptotic mass is global data the scaling limit discards. 1010.0860v1
  says so itself -- "finding local solutions in the vicinity of the horizon
  does not guarantee the existence of global asymptotically flat solutions...
  this would also allow to construct the E(J) diagram for such configurations"
  -- and leaves that diagram open. Constructing it is what `extremality.py`
  does, and it is the part of this work that is not already published.

Units: `v1=1` fixes the scale, so the coupling that enters here is the
dimensionless `alpha_tilde = alpha_GB/v1`. Physical quantities follow from
`v_i -> lambda v_i`, `alpha -> lambda alpha`, under which `J` and `S` both
scale as `lambda^{3/2}` -- so `S/J` and `alpha/J^{2/3}` are scale invariant and
are what gets compared. Variables translate to the paper's as
`v1 = v1_paper`, `v2 = v2_paper`, `v3 = v2_paper v3_paper`, `k = 2 k_paper`,
and `J_paper = 2 J`, because their charge is conjugate to their own `k` and
counts both planes of rotation while ours is each spin.
"""
import numpy as np
from scipy.optimize import brentq

from ._near_horizon_generated import density, entropy_function, scalar, stationarity

__all__ = ['VACUUM_STATE', 'BRANCH_LIMIT', 'extremal', 'extremal_curve',
           'entropy_over_spin', 'coupling_from_invariant', 'paper_residuals']


# The alpha_GB=0 root, solved exactly in the derivation script: k=1, v2=4 v1,
# v3=8 v1. It seeds every continuation here, and it is also the check that the
# normalisation is right -- it reproduces S = 2 pi J, which is extremal
# Myers-Perry (S = 2 pi^2 a^3, J = pi a^3 each) with nothing tuned, and is the
# paper's own eq. (4.13) in its variables (v3_paper = 2, k_paper = 1/2).
VACUUM_STATE = (4., 8., 1.)

_STEP = 1e-20


def _jacobian(state, alpha):
    """d(stationarity)/d(v2,v3,k), by complex step on the generated polynomials."""
    jac = np.empty((3, 3))
    for j in range(3):
        trial = [complex(v) for v in state]
        trial[j] += _STEP*1j
        jac[:, j] = [complex(e).imag/_STEP for e in stationarity(*trial, alpha)]
    return jac


def _newton(state, alpha, tolerance=1e-13, max_iterations=60):
    """Damped Newton on the three polynomials, with a *relative* stopping test.

    The stationarity conditions carry monomials of order 1e2-1e3 at these
    states, so an absolute threshold of 1e-14 asks for more digits than the
    polynomials have; the scale below is the one Newton itself sees, and the
    reported residual is scaled by it so a caller can judge it.
    """
    state = np.asarray(state, dtype=float)
    for _ in range(max_iterations):
        residual = np.asarray(stationarity(*state, alpha), dtype=float)
        jac = _jacobian(state, alpha)
        scale = max(1., float(np.max(np.abs(jac))*np.max(np.abs(state))))
        norm = float(np.max(np.abs(residual))/scale)
        if norm < tolerance:
            return state, norm
        step = np.linalg.solve(jac, -residual)
        for power in range(30):
            trial = state+step*2.**(-power)
            if np.all(trial > 0) and np.max(np.abs(
                    np.asarray(stationarity(*trial, alpha), dtype=float)))/scale < norm:
                state = trial
                break
        else:
            return state, norm
    raise RuntimeError('near-horizon Newton did not converge')


def _charges(state, alpha):
    """J = df/dk and S = 2 pi (k J - f), at v1=1, by complex step in k."""
    v2, v3, k = (float(v) for v in state)
    spin = complex(entropy_function(1., v2, v3, complex(k, _STEP), alpha)).imag/_STEP
    value = float(entropy_function(1., v2, v3, k, alpha))
    return spin, 2*np.pi*(k*spin-value)


def paper_residuals(state):
    """Eqs. (4.11) and (4.12) of 1010.0860v1, evaluated on one of our states.

    Transcribed from the paper in the paper's own variables, with
    `alpha_paper = 4 alpha_GB`, `v3_paper = v3/v2` and `k_paper = k/2`. The
    three stationarity residuals come back scaled by the largest monomial the
    polynomials carry; the spin comes back as the ratio of the paper's eq.
    (4.12) to twice ours -- twice, because their charge counts both planes of
    rotation while ours is each spin.

    A failure here means either that our derivation disagrees with the
    published one or that the transcription is wrong. Both are worth knowing,
    and neither can be papered over by adjusting the comparison.
    """
    v1, v2 = 1., float(state['v2'])
    v3, k = float(state['v3'])/v2, float(state['k'])/2
    alpha = 4*float(state['alpha_tilde'])
    stationary = np.array([
        -16*v1**2 + 4*v1**2*v3 + k**2*v2**2*v3 + alpha*k**2*v2*v3*(4-3*v3),
        -16*v1**2 + 12*v1*v2 + 4*v1**2*v3 - 5*k**2*v2**2*v3
        + alpha*(-4*v1*(v3-4) + 3*k**2*v2*v3*(3*v3-4)),
        -16*v1**2 + 4*v1*v2 + 12*v1**2*v3 - 3*k**2*v2**2*v3
        + alpha*(-4*v1*(3*v3-4) + 3*k**2*v2*v3*(5*v3-4))])
    scale = max(1., float(np.max([v1, v2, v3, k]))**4)
    spin = np.pi*k*(v2*v3)**1.5*(v2 + alpha*(4-3*v3))/(8*v1)
    return dict(stationarity=float(np.max(np.abs(stationary))/scale),
                spin_ratio=float(spin/(2*state['J'])),
                alpha_paper=alpha)


def extremal(alpha_tilde, rungs=None):
    """The extremal near-horizon state at dimensionless coupling alpha/v1.

    Continuation from the vacuum root in steps of at most 0.02, because the
    polynomial system has several branches and Newton from a distant seed can
    cross to one of them; stepping keeps it on the branch that starts at
    Myers-Perry. `residual` is reported so a caller can refuse a state the
    same way the spectral solver's gates refuse a profile.
    """
    alpha_tilde = float(alpha_tilde)
    if alpha_tilde < 0:
        raise ValueError('alpha_tilde must be non-negative')
    if rungs is None:
        # The branch steepens as alpha_tilde -> 1 (v2, v3 collapse towards zero),
        # so a uniform ladder that is fine at 0.5 walks off the branch at 0.98.
        # Refining in proportion to the distance left keeps the seed close.
        rungs, value = [0.], 0.
        while value < alpha_tilde:
            value = min(alpha_tilde, value+max(1e-4, min(.02, .05*(1-value))))
            rungs.append(value)
    state, residual = np.asarray(VACUUM_STATE, dtype=float), 0.
    for value in rungs:
        state, residual = _newton(state, float(value))
    spin, entropy = _charges(state, alpha_tilde)
    v2, v3, k = (float(v) for v in state)
    return dict(alpha_tilde=alpha_tilde, v1=1., v2=v2, v3=v3, k=k,
                J=float(spin), S=float(entropy),
                S_over_J=float(entropy/spin),
                invariant=float(alpha_tilde/spin**(2/3)),
                residual=float(residual),
                ricci_scalar=float(scalar(1., v2, v3, k)),
                gauss_bonnet_density=float(density(1., v2, v3, k)))


def entropy_over_spin(alpha_tilde):
    return extremal(alpha_tilde)['S_over_J']


# The branch runs over alpha_tilde in [0,1): v2 and v3 shrink to zero as
# alpha_tilde -> 1 and the invariant y runs monotonically from 0 to infinity,
# so that open interval is the whole branch and the right bracket to search.
BRANCH_LIMIT = 0.985


def coupling_from_invariant(invariant, bracket=(0., BRANCH_LIMIT)):
    """Invert y = alpha_tilde/J(alpha_tilde)^{2/3} for alpha_tilde.

    The numerics measure `y` (from a physical coupling and a measured spin);
    this module is parameterised by `alpha_tilde`. Comparing them needs the
    inversion, and `y` is monotone on the branch over the range used here --
    checked by the caller, not assumed.
    """
    target = float(invariant)
    if target == 0.:
        return 0.
    low, high = bracket
    return float(brentq(lambda a: extremal(a)['invariant']-target, low, high,
                        xtol=1e-13, rtol=1e-14))


def extremal_curve(couplings):
    return [extremal(float(value)) for value in couplings]
