"""Symbolic checks of the cyclic angular equation, without regeneration."""
import numpy as np
import sympy as sp


def test_angular_integral_is_the_action_conjugate_momentum():
    from experiments.derive_egb_rotating import lagrangians
    from rotating_bh.egb_rotating_angular import first_integral
    data = lagrangians()
    r, b, f, g, h, w = [data[k] for k in ['r', 'b', 'f', 'g', 'h', 'w']]
    alpha = sp.symbols('alpha', real=True)
    L = data['L_E']+alpha*data['L_GB']
    assert sp.diff(L, w) == 0
    momentum = sp.diff(L, sp.diff(w, r))
    expected = h*sp.sqrt(f*h/b)*(g+alpha*(16-12*h/g-f*sp.diff(g, r)**2/g))*sp.diff(w, r)
    assert sp.simplify(momentum-expected) == 0
    z, B, F, H = sp.symbols('z B F H', positive=True)
    W, V = sp.symbols('W V', real=True)
    compact = expected.subs({g: r**2, sp.diff(g, r): 2*r}, simultaneous=True)
    compact = compact.subs({r: 1/z, b: (1-z*z)*B, f: (1-z*z)*F,
                           h: (1+z**4*H)/z**2, sp.diff(w, r): z**5*(z*V-4*W)},
                          simultaneous=True)
    actual = sp.nsimplify(first_integral(1-z, [B, F, H, W], [0, 0, 0, V], alpha))
    assert sp.simplify(compact-actual) == 0


def test_raw_horizon_ew_is_the_limit_of_the_interior_angular_equation():
    from rotating_bh.egb_rotating_angular import first_integral
    from rotating_bh._egb_rotating_horizon_generated import horizon
    z, B, F, H = sp.symbols('z B F H', positive=True)
    W, P, Fx, Q, V, Wxx, alpha = sp.symbols('W P Fx Q V Wxx alpha', real=True)
    # Use exact SymPy rationals rather than floating square roots.
    A = 1+z**4*H
    C = 1+4*alpha*z**2*(1-(1-z**2)*F-3*z**4*H)
    J = sp.sqrt(F/B)*A**sp.Rational(3, 2)*C*(z*V-4*W)
    actual = sp.nsimplify(first_integral(1-z, [B, F, H, W], [P, Fx, Q, V], alpha))
    assert sp.simplify(J-actual) == 0
    dJ = -sp.diff(J, z)+sum(sp.diff(J, v)*dv for v, dv in
                            [(B, P), (F, Fx), (H, Q), (W, V), (V, Wxx)])
    normalized = sp.simplify(dJ.subs(z, 1)/sp.sqrt(F/B)/sp.sqrt(1+H))
    Ew = horizon(B, F, H, W, P, Q, V, Fx, Wxx, alpha)[3]
    assert sp.simplify(Ew+8*B*F*normalized) == 0


def test_myers_perry_has_constant_angular_integral():
    from rotating_bh.egb_rotating_angular import first_integral
    q = .3
    a = q*q/(1-q*q)
    x = np.linspace(0., 1., 73)
    z = 1-x
    B, F = (1-a*z*z)/(1+a*z**4), 1-a*z*z
    H, W = np.full_like(x, a), q/(1-q*q+q*q*z**4)
    V = 4*q**3*z**3/(1-q*q+q*q*z**4)**2
    np.testing.assert_allclose(first_integral(x, [B, F, H, W], [0, 0, 0, V], 0.),
                               -4*q/(1-q*q), atol=2e-14, rtol=2e-14)
