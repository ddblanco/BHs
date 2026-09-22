"""Exact checks that do not depend on the extremal extrapolation (Hito 8, rev.).

The first version of the Hito 8 measurement had exact anchors only at
`alpha_GB = 0`. A referee pointed out that the static limit of this family is
closed-form at *finite* coupling too, and is therefore the one exact test at
`alpha != 0` that was missing. That observation is correct and this module
implements it, together with two further tests that cost nothing once the
linear response is available in both parameters.

**The static limit.** At `q=0` the solution is Boulware-Deser
\\cite{Boulware:1985wk}: the Wheeler polynomial in `D=5` is `X r^2 + 2 alpha X^2
= mu` with `X = 1-f`, so `mu = r_H^2 + 2 alpha` on the horizon and

    M = (3 pi/8)(r_H^2 + 2 alpha),
    T = r_H / (2 pi (r_H^2 + 4 alpha)),
    S = (A/4)(1 + 12 alpha/r_H^2),   A = 2 pi^2 r_H^3,

the entropy being the Jacobson-Myers form with `Rtilde = 6/r_H^2` on a round
`S^3`. Holding `S` fixed and differentiating `M` gives the conjugate potential
in closed form,

    Psi_static = (3 pi/4) (4 alpha - 3 r_H^2)/(r_H^2 + 4 alpha),

which satisfies the Smarr relation identically at `J=0` and reduces to
`-9 pi/4` at `alpha=0`. Both statements are verified symbolically in
`tests/test_consistency.py`.

**The first law in the spin direction.** With the linear response available for
`omega_h` as well as for `alpha_GB`, `dM - T dS - 2 Omega dJ` along the family
at fixed coupling is an exact combination of measured quantities, and it tests
the extraction of all five of them against each other. Nothing in the extremal
measurement uses it, so it is genuinely independent.

**Goon-Penco, used rather than cited.** The relation
`dM_ext/dalpha = -lim_{T->0} T dS/dalpha|_{M,J}` is an identity given the first
law, so evaluating `-T dS/dalpha|_{M,J}` at finite temperature and comparing it
with `Psi` tests the data rather than the relation. Holding `M` and `J` fixed
while varying `alpha` needs two directions in the family, and the scaling
`r_H -> lambda r_H`, `alpha -> lambda^2 alpha` supplies the second one, so this
too costs no new solve.
"""
import numpy as np

from .egb_rotating_bvp import _cheb, _residual
from .egb_rotating_observables import measure
from .egb_rotating_predictor import analytic_jacobian
from .gb_response import response_field, response_observables

__all__ = ['static_closed_form', 'myers_perry_mass', 'first_law_residual',
           'goon_penco_residual', 'jacobian_conditioning']


def static_closed_form(alpha_gb, r_h=1.):
    """The q=0 member of the family, in closed form, at any coupling."""
    alpha, r = float(alpha_gb), float(r_h)
    area = 2*np.pi**2*r**3
    mass = 3*np.pi/8*(r**2 + 2*alpha)
    return dict(alpha_gb=alpha, r_h=r,
                E=float(mass),
                T_H=float(r/(2*np.pi*(r**2 + 4*alpha))),
                S=float(area/4*(1 + 12*alpha/r**2)),
                A_H=float(area),
                psi_gb=float(3*np.pi/4*(4*alpha - 3*r**2)/(r**2 + 4*alpha)),
                x=float(3*np.pi*alpha/(4*mass)),
                t_scaled=float(r*np.sqrt(r**2 + 2*alpha)/(r**2 + 4*alpha)))


def myers_perry_mass(q, r_h=1.):
    """M for equal-spin Myers-Perry at alpha=0, in the gauge of the ansatz.

    In that gauge `r_H^2 = r_+^2 + a^2`, so `q = r_H Omega_H = a/r_H` and
    `M = 3 pi r_H^2 / (8(1-q^2))`. Used to check the mass extraction along the
    whole spin range, not merely at the static end.
    """
    return float(3*np.pi*float(r_h)**2/(8*(1-float(q)**2)))


def first_law_residual(solution, window=0):
    """dM - T dS - 2 Omega dJ along the family at fixed coupling.

    Returned both absolute and relative to the largest term, so that a point
    where every derivative is small is not flattered by the comparison.
    """
    spin = response_observables(solution, window=window, parameter='omega_h')
    observed = measure(solution)
    terms = (spin['dE'], observed['T_H']*spin['dS'],
             2*solution.omega_h*spin['dJ'])
    residual = terms[0] - terms[1] - terms[2]
    scale = max(abs(value) for value in terms)
    return dict(omega_h=float(solution.omega_h),
                alpha_gb=float(solution.alpha_gb),
                dE=float(terms[0]), T_dS=float(terms[1]), Omega_dJ=float(terms[2]),
                absolute=float(abs(residual)),
                relative=float(abs(residual)/scale) if scale else float('inf'))


def goon_penco_residual(solution, window=0):
    """-T dS/dalpha at fixed M and J, against Psi measured directly.

    Two directions are needed to hold both charges fixed while the coupling
    moves. Writing physical quantities in terms of the `r_H=1` family,
    `M = r^2 m`, `J = r^3 j`, `S = r^3 s`, `alpha = a r^2`, and demanding
    `dM = dJ = 0` fixes `dr` and `dq` per unit `da`; the entropy derivative and
    the coupling increment follow. Degenerate at `q=0`, where `j` and its spin
    derivative both vanish, so callers should stay away from the static end.
    """
    coupling = response_observables(solution, window=window, parameter='alpha_gb')
    spin = response_observables(solution, window=window, parameter='omega_h')
    observed = measure(solution)
    m, j, s = observed['E'], observed['J'], observed['S']
    matrix = np.array([[2*m, spin['dE']], [3*j, spin['dJ']]])
    right = -np.array([coupling['dE'], coupling['dJ']])
    conditioning = float(np.linalg.cond(matrix))
    try:
        d_radius, d_spin = np.linalg.solve(matrix, right)
    except np.linalg.LinAlgError:
        return dict(omega_h=float(solution.omega_h),
                    alpha_gb=float(solution.alpha_gb), singular=True)
    d_alpha = 2*float(solution.alpha_gb)*d_radius + 1.
    d_entropy = 3*s*d_radius + spin['dS']*d_spin + coupling['dS']
    identity = -observed['T_H']*d_entropy/d_alpha
    scale = max(abs(identity), abs(coupling['psi_gb']))
    return dict(omega_h=float(solution.omega_h),
                alpha_gb=float(solution.alpha_gb), singular=False,
                psi_gb=float(coupling['psi_gb']),
                minus_T_dS_dalpha=float(identity),
                entropy_derivative=float(d_entropy/d_alpha),
                absolute=float(abs(identity-coupling['psi_gb'])),
                relative=float(abs(identity-coupling['psi_gb'])/scale),
                system_conditioning=conditioning)


def jacobian_conditioning(solution):
    """Condition number of dR/du at a solution, and the residual it produces.

    Reported because the linear response solves a system with this matrix, and
    approaching extremality it is where the accuracy would be lost first if it
    were lost anywhere.
    """
    n = int(solution.initial_resolution)
    x, _, _ = _cheb(n)
    zz = 1-x
    flat = np.asarray(solution.evaluate(x)).ravel()
    jacobian = analytic_jacobian(flat, n, solution.omega_h,
                                 float(solution.alpha_gb), zz)
    residual = _residual(flat, n, solution.omega_h, float(solution.alpha_gb), zz)
    return dict(omega_h=float(solution.omega_h),
                alpha_gb=float(solution.alpha_gb), resolution=n,
                condition_number=float(np.linalg.cond(jacobian)),
                residual=float(np.max(np.abs(residual))))
