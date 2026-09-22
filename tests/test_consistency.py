"""The referee's exact checks, verified where they can be: symbolically.

The Hito 8 measurement first shipped with exact anchors only at `alpha_gb = 0`.
The static limit of this family is closed-form at *finite* coupling too, and is
therefore the one exact test at `alpha != 0` that was missing. `consistency.py`
implements it; this file proves the closed form it implements, rather than
trusting the algebra that produced it.

Three things are checked here and nowhere else:

  1. `Psi_static` really is `(dM/dalpha)|_S` for the Boulware-Deser solution,
     derived symbolically from `M(r_H, alpha)` and `S(r_H, alpha)` instead of
     being typed in;
  2. that expression satisfies the Smarr relation `2M = 3TS + 2 alpha Psi`
     identically at `J = 0`, and reduces to `-9 pi/4` as `alpha -> 0`, which is
     the `u = 0` endpoint the manuscript uses to fix eq. (14);
  3. the Myers-Perry mass used to bound the mass extraction is the one implied
     by the gauge of the ansatz, `r_H^2 = r_+^2 + a^2`.

The numerical side -- that the solver reproduces all of this -- is tested by
`test_extremality_artifacts.py` against the recorded measurement, not here.
"""
import numpy as np
import pytest
import sympy as sp

from rotating_bh.consistency import myers_perry_mass, static_closed_form


@pytest.fixture(scope='module')
def symbols():
    r, a = sp.symbols('r_H alpha', positive=True)
    mass = 3*sp.pi/8*(r**2 + 2*a)
    temperature = r/(2*sp.pi*(r**2 + 4*a))
    # Jacobson-Myers on a round S^3: A/4 * (1 + 12 alpha / r_H^2).
    entropy = sp.simplify(2*sp.pi**2*r**3/4*(1 + 12*a/r**2))
    return r, a, mass, temperature, entropy


def test_the_static_potential_is_the_derivative_at_fixed_entropy(symbols):
    """Psi = dM/dalpha at fixed S, by implicit differentiation, not by hand.

    Inverting S(r_H, alpha) explicitly means solving a cubic; the constraint is
    the same one without the branches:  dM/dalpha|_S = M_a - M_r S_a / S_r.
    """
    r, a, mass, _, entropy = symbols
    psi = sp.simplify(sp.diff(mass, a)
                      - sp.diff(mass, r)*sp.diff(entropy, a)/sp.diff(entropy, r))
    assert sp.simplify(psi - 3*sp.pi/4*(4*a - 3*r**2)/(r**2 + 4*a)) == 0


def test_the_static_potential_satisfies_smarr_identically(symbols):
    """2M = 3TS + 2 alpha Psi at J = 0, for every r_H and alpha, not numerically."""
    r, a, mass, temperature, entropy = symbols
    psi = 3*sp.pi/4*(4*a - 3*r**2)/(r**2 + 4*a)
    assert sp.simplify(2*mass - 3*temperature*entropy - 2*a*psi) == 0


def test_the_static_potential_reduces_to_the_perturbative_endpoint(symbols):
    """alpha -> 0 gives -9 pi/4, the u = 0 endpoint of the manuscript's eq. (14)."""
    r, a, *_ = symbols
    psi = 3*sp.pi/4*(4*a - 3*r**2)/(r**2 + 4*a)
    assert sp.simplify(sp.limit(psi, a, 0) + 9*sp.pi/4) == 0


def test_the_static_potential_changes_sign_where_the_manuscript_says(symbols):
    """Psi_static = 0 at alpha/r_H^2 = 3/4, that is at x = 3/5."""
    r, a, mass, *_ = symbols
    psi = 3*sp.pi/4*(4*a - 3*r**2)/(r**2 + 4*a)
    zeros = sp.solve(sp.Eq(psi, 0), a)
    assert zeros == [3*r**2/4]
    x = sp.simplify((3*sp.pi*a/(4*mass)).subs(a, 3*r**2/4))
    assert sp.nsimplify(x) == sp.Rational(3, 5)


@pytest.mark.parametrize('alpha_gb', [0., .05, .2, .5])
def test_the_closed_form_module_agrees_with_the_symbolic_expressions(symbols, alpha_gb):
    """consistency.static_closed_form is the same object, evaluated."""
    r, a, mass, temperature, entropy = symbols
    psi = 3*sp.pi/4*(4*a - 3*r**2)/(r**2 + 4*a)
    exact = static_closed_form(alpha_gb, r_h=1.)
    for value, expression in ((exact['E'], mass), (exact['T_H'], temperature),
                              (exact['S'], entropy), (exact['psi_gb'], psi)):
        expected = float(expression.subs({r: 1, a: alpha_gb}))
        assert value == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_the_myers_perry_mass_is_the_one_the_ansatz_gauge_implies():
    """M = 3 pi r_H^2 / (8(1-q^2)) with q = a/r_H and r_H^2 = r_+^2 + a^2."""
    a, rp = sp.symbols('a r_plus', positive=True)
    r_h = sp.sqrt(rp**2 + a**2)
    q = a/r_h
    # mu = (r_+^2 + a^2)^2 / r_+^2 for equal-spin Myers-Perry, and M = 3 pi mu/8.
    mass = 3*sp.pi/8*(rp**2 + a**2)**2/rp**2
    gauge = 3*sp.pi*r_h**2/(8*(1 - q**2))
    assert sp.simplify(mass - gauge) == 0
    # and the module evaluates it
    for spin in (.05, .3, .634, .7):
        assert myers_perry_mass(spin) == pytest.approx(
            float(gauge.subs({a: spin, rp: np.sqrt(1 - spin**2)})), rel=1e-13)


def test_the_static_limit_is_extremal_nowhere():
    """T > 0 at every coupling: the static member is never the extremal one."""
    for alpha_gb in (0., .05, .1, .2, .3, .4, .5):
        assert static_closed_form(alpha_gb)['T_H'] > 0.
