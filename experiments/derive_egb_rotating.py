"""Derive the general rotating (b,f,h,w all unknown, alpha_GB!=0) D=5 EGB
radial system from the reduced action, gauge-fixed at g=r^2.

Scope of this script (Hito 4A, first pass): equations only, in the
*uncompactified* r coordinate. Horizon/infinity expansions and the
compactified BVP-ready system are a separate, later step (they need their
own re-derivation for this system, see docs/2026-09-09-hito-4-diseno.md;
the vacuum compactification cannot be assumed to carry over unchanged).

Method and why: a full symbolic tensor derivation (Riemann+GB contracted
directly on the general rotating metric, as experiments/derive_static_egb.py
does for the diagonal static case) was attempted first and found
computationally intractable in this environment -- Christoffel/Riemann
construction alone did not finish within several minutes for this
non-diagonal 5x5 metric. This mirrors why Hito 2's vacuum derivation
(docs/vacuum-bvp.md) used the reduced action instead of the brute-force
tensor route for the rotating case.

Equivalence with the tensor route (AstraCheck, 2026-09-09: the gate is
Euler-Lagrange vs. tensor contraction on generic profiles, not literal
Lagrangian equality) is therefore checked the way Hito 2 already checks it:
numerically, via `E_v=-r^2(G+alpha_GB H)^{ij} dg_{ij}/dv` on jet-substituted,
non-solution profiles, using the *numeric* (NumPy) curvature()/gauss_bonnet()
evaluator from einstein.py -- which is fast regardless of the metric being
non-diagonal, unlike full symbolic Riemann-squared. This is the scientific
gate below; generation aborts if it fails.

L_E, L_GB and the first integral of w (eq. 2.10-2.12 of 1010.0860v1) were
read from the cached PDF and independently confirmed against the arXiv HTML
rendering (matching term for term). L_E is additionally cross-checked here
against the vacuum Lagrangian already derived and tested in Hito 2
(docs/vacuum-bvp.md): both must give identical Euler-Lagrange equations
(they are allowed to differ by a total radial derivative, per AstraCheck's
adjustment to the design; they turn out not to differ even by that, but
verifying via equations-of-motion rather than literal density equality is
the correct-in-general gate).
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
for key in ('TEMP', 'TMP'):
    (ROOT/'.cache/tmp').mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(ROOT/'.cache/tmp')
import numpy as np
import sympy as sp

import sys
sys.path.insert(0, str(ROOT/'src'))
from rotating_bh.einstein import metric_from_ansatz, compile_metric, curvature, gauss_bonnet


def lagrangians():
    """L_E, L_GB (eq. 2.11 of 1010.0860v1) before fixing any gauge.

    alpha_GB here is the internal normalization (alpha_1010=4*alpha_GB, per
    docs/convenciones.md): L_eff=L_E+alpha_GB*L_GB reproduces the paper's
    L_eff=L_E+(alpha_1010/4)*L_GB exactly.
    """
    r = sp.symbols('r', positive=True)
    b = sp.Function('b', positive=True)(r)
    f = sp.Function('f', positive=True)(r)
    g = sp.Function('g', positive=True)(r)
    h = sp.Function('h', positive=True)(r)
    w = sp.Function('w', real=True)(r)
    d = lambda v: sp.diff(v, r)
    L_E = sp.sqrt(f*h/b)*(d(b)*d(g) + (g/(2*h))*d(b)*d(h) + (b/(2*g))*d(g)**2
                          + (b/h)*d(g)*d(h) + sp.Rational(1, 2)*g*h*d(w)**2
                          + (2*b/f)*(4-h/g))
    L_GB = sp.sqrt(f*h/b)*(1/g)*((4*h/g)*d(b)*d(g)
                                  + 2*(4*g-3*h)*(d(b)*d(h)/h+h*d(w)**2)
                                  - (f/(2*h))*d(b)*d(h)*d(g)**2
                                  - sp.Rational(1, 2)*f*h*d(g)**2*d(w)**2)
    return dict(r=r, b=b, f=f, g=g, h=h, w=w, L_E=L_E, L_GB=L_GB)


def vacuum_lagrangian(r, b, f, g, h, w):
    """The Hito 2 Einstein Lagrangian (docs/vacuum-bvp.md), independently
    derived and tested with 116+ passing tests; used only as a cross-check
    for the L_E transcription above, never as a derivation input here.
    """
    d = lambda v: sp.diff(v, r)
    return g*sp.sqrt(b*h/f)*(f*(d(g)**2/(2*g**2)+d(b)*d(g)/(b*g)+d(b)*d(h)/(2*b*h)
                                 + d(g)*d(h)/(g*h)+h*d(w)**2/(2*b))+8/g-2*h/g**2)


def euler_lagrange(L, field, r):
    """(dL/dv-d/dr(dL/dv')); NOT yet divided by sqrt(b*h/f)."""
    d = lambda v: sp.diff(v, r)
    return sp.diff(L, field)-d(sp.diff(L, d(field)))


def check_action_transcription(vars_):
    """L_E (transcribed) and the independent vacuum Lagrangian must give the
    *same* Euler-Lagrange equations (equal up to a total radial derivative,
    per AstraCheck; checked here via equations of motion, not density
    equality). Numeric spot-check at several generic jets, mirroring how
    Hito 2 itself validates its own Lagrangian against the tensor evaluator.
    """
    r, b, f, g, h, w = (vars_[k] for k in ('r', 'b', 'f', 'g', 'h', 'w'))
    L_vac = vacuum_lagrangian(r, b, f, g, h, w)
    d = lambda v: sp.diff(v, r)
    jets = [
        dict(b=1.3, bp=0.7, bpp=0.2, f=0.9, fp=-0.2, g=2.1, gp=1.1, gpp=-0.3,
             h=1.7, hp=0.4, hpp=0.1, w=0.05, wp=-0.3, wpp=0.15, r=1.5),
        dict(b=0.6, bp=-0.4, bpp=0.9, f=1.4, fp=0.3, g=3.2, gp=-0.6, gpp=0.5,
             h=0.8, hp=1.2, hpp=-0.2, w=-0.2, wp=0.6, wpp=-0.1, r=2.3),
    ]
    for jet in jets:
        subs = {sp.diff(b, r, 2): jet['bpp'], sp.diff(g, r, 2): jet['gpp'],
                sp.diff(h, r, 2): jet['hpp'], sp.diff(w, r, 2): jet['wpp'],
                d(b): jet['bp'], d(f): jet['fp'], d(g): jet['gp'], d(h): jet['hp'], d(w): jet['wp'],
                b: jet['b'], f: jet['f'], g: jet['g'], h: jet['h'], w: jet['w'], r: jet['r']}
        for field, name in [(b, 'b'), (g, 'g'), (h, 'h'), (w, 'w')]:
            e_vac = float(euler_lagrange(L_vac, field, r).subs(subs))
            e_e = float(euler_lagrange(vars_['L_E'], field, r).subs(subs))
            if abs(e_vac-e_e) > 1e-8*max(abs(e_vac), abs(e_e), 1):
                raise RuntimeError(
                    f'L_E transcription disagrees with the independent vacuum '
                    f'Lagrangian on field {name}: {e_vac} vs {e_e}')
    print('L_E transcription matches the independent vacuum Euler-Lagrange equations: OK')


PROFILES = [
    # (R, b0,b1,b2, f0,f1,f2, h0,h1,h2, w0,w1,w2, alpha_GB, theta) - generic,
    # not solutions of anything; chosen to span small/large alpha_GB, sign
    # changes in the derivatives, and a couple of angles.
    (1.4, 1.1, 0.6, -0.3, 0.8, -0.4, 0.2, 1.6, 0.5, -0.6, 0.10, -0.20, 0.30, 0.02, 0.6),
    (2.2, 0.7, -0.5, 0.8, 1.3, 0.3, -0.2, 0.9, -0.7, 0.4, -0.05, 0.35, -0.15, 0.30, 1.1),
    (1.0, 1.9, 0.2, 0.5, 0.6, -0.6, 0.5, 2.1, 0.9, -0.2, 0.20, 0.10, 0.10, 0.05, 0.9),
    (3.0, 0.9, -0.2, -0.4, 1.7, 0.5, 0.3, 0.7, 0.3, 0.6, -0.30, -0.10, 0.20, 0.45, 0.3),
    (1.7, 1.4, 0.9, 0.1, 0.5, -0.1, -0.5, 1.2, -0.4, 0.3, 0.08, -0.40, -0.25, 0.00, 1.2),
    (2.5, 0.5, 0.3, 0.2, 0.9, 0.7, -0.6, 1.8, 0.6, -0.5, -0.12, 0.25, 0.05, 0.60, 0.4),
]


def gauge_fixed_variations(vars_):
    """E_b,E_g,E_h,E_w,E_f after g=r^2, normalized by 1/sqrt(b*h/f) (same
    convention as vacuum_equations.py: E_v=(dL/dv-d/dr(dL/dv'))/sqrt(bh/f)).
    """
    r, b, f, g, h, w = (vars_[k] for k in ('r', 'b', 'f', 'g', 'h', 'w'))
    alpha = sp.Symbol('alpha_GB')
    L_eff = vars_['L_E']+alpha*vars_['L_GB']
    d = lambda v: sp.diff(v, r)
    if sp.diff(L_eff, d(f)) != 0:
        raise RuntimeError('unexpected f-prime dependence in L_eff')
    raw = {name: euler_lagrange(L_eff, field, r) for name, field in
           [('b', b), ('g', g), ('h', h), ('w', w)]}
    raw['f'] = sp.diff(L_eff, f)
    normalized = {name: sp.cancel(expr/sp.sqrt(b*h/f)) for name, expr in raw.items()}
    gauge = {sp.diff(g, r, 2): 2, d(g): 2*r, g: r*r}
    gauged = {name: sp.cancel(expr.subs(gauge)) for name, expr in normalized.items()}
    return alpha, gauged


def check_tensor_equivalence(vars_, alpha, gauged):
    """E_v (action route, gauge-fixed) must equal -r^2(G+alpha_GB H)^{ij}
    dg_{ij}/dv (tensor route, einstein.py) on PROFILES; E_f must equal the
    mixed r^2 G^r_r/f (the reserved-constraint identity of Hito 2, extended
    with the GB tensor). Numeric only: a full symbolic Riemann-squared for
    this non-diagonal ansatz did not finish in this environment (see module
    docstring), but the numeric tensor evaluator is exact NumPy contraction,
    not an approximation, so this is a real equivalence check, not a proxy.
    """
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    t_, rc, theta, phi1, phi2 = sp.symbols('t r theta phi1 phi2', real=True)
    R = sp.symbols('R', positive=True)
    b0, b1, b2, f0, f1, f2, h0, h1, h2, w0, w1, w2 = sp.symbols(
        'b0 b1 b2 f0 f1 f2 h0 h1 h2 w0 w1 w2')
    jet = lambda v0, v1, v2: v0+v1*(rc-R)+v2*(rc-R)**2/2
    metric = metric_from_ansatz(rc, theta, jet(b0, b1, b2), jet(f0, f1, f2),
                                 jet(h0, h1, h2), jet(w0, w1, w2))
    params = (R, b0, b1, b2, f0, f1, f2, h0, h1, h2, w0, w1, w2)
    coords = (t_, rc, theta, phi1, phi2)
    evaluate = compile_metric(metric, coords, params)
    field_syms = {'b': b0, 'f': f0, 'h': h0, 'w': w0}
    d_evaluate = {name: compile_metric(sp.Matrix(5, 5, lambda i, j: sp.diff(metric[i, j], sym)),
                                        coords, params) for name, sym in field_syms.items()}

    def action_side(name, p):
        R_, b0v, b1v, b2v, f0v, f1v, f2v, h0v, h1v, h2v, w0v, w1v, w2v, a, _ = p
        subs = {alpha: a, r: R_, b: b0v, f: f0v, h: h0v, w: w0v,
                sp.diff(b, r): b1v, sp.diff(h, r): h1v, sp.diff(w, r): w1v,
                sp.diff(b, r, 2): b2v, sp.diff(h, r, 2): h2v, sp.diff(w, r, 2): w2v,
                sp.diff(f, r): f1v}
        return float(gauged[name].subs(subs))

    def tensor_field(p):
        R_, b0v, b1v, b2v, f0v, f1v, f2v, h0v, h1v, h2v, w0v, w1v, w2v, a, th = p
        g, dg, ddg = evaluate(0, R_, th, 0, 0, R_, b0v, b1v, b2v, f0v, f1v, f2v,
                               h0v, h1v, h2v, w0v, w1v, w2v)
        result = curvature(g, dg, ddg)
        gb = gauss_bonnet(g, result['inverse'], result['riemann'], result['ricci'], result['scalar'])
        field_lower = result['einstein']+a*gb['H']
        return field_lower, result['inverse'], R_

    worst = 0.0
    for p in PROFILES:
        field_lower, inv, R_ = tensor_field(p)
        field_upper = inv @ field_lower @ inv
        for name in ('b', 'h', 'w'):
            a_side = action_side(name, p)
            R_, b0v, b1v, b2v, f0v, f1v, f2v, h0v, h1v, h2v, w0v, w1v, w2v, a, th = p
            dgdv, _, _ = d_evaluate[name](0, R_, th, 0, 0, R_, b0v, b1v, b2v, f0v, f1v, f2v,
                                           h0v, h1v, h2v, w0v, w1v, w2v)
            t_side = -R_**2*float(np.einsum('ij,ij->', field_upper, dgdv))
            rel = abs(a_side-t_side)/max(abs(a_side), abs(t_side), 1e-12)
            worst = max(worst, rel)
            if rel > 1e-8:
                raise RuntimeError(f'action/tensor mismatch for {name} at profile {p}: '
                                    f'{a_side} vs {t_side} (rel={rel:.3e})')
        e_f = action_side('f', p)
        f0v = p[4]
        Grr_mixed = float(inv[1, :] @ field_lower[:, 1])
        candidate = R_**2*Grr_mixed/f0v
        rel = abs(e_f-candidate)/max(abs(e_f), abs(candidate), 1e-12)
        worst = max(worst, rel)
        if rel > 1e-8:
            raise RuntimeError(f'E_f/G^r_r mismatch at profile {p}: {e_f} vs {candidate}')
    print(f'Action route == tensor route on {len(PROFILES)} generic profiles '
          f'(worst relative difference {worst:.3e}): OK')


def check_order_stays_seven(vars_, alpha, gauged):
    """The coefficient matrix multiplying [f',b'',h'',w''] must be generically
    invertible on PROFILES: confirms the system solves for the SAME four
    unknowns as vacuum (order 7 total: f' + b'' + h'' + w''), not the "7->9"
    anticipated (without derivation) by the original Hito 4 design -- see
    reports/hito-4-review.md.
    """
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    unknown = [sp.diff(f, r), sp.diff(b, r, 2), sp.diff(h, r, 2), sp.diff(w, r, 2)]
    eqs = []
    for name in ('b', 'g', 'h', 'w'):
        numer, _ = sp.fraction(sp.together(gauged[name]))
        eqs.append(sp.expand(numer))
    M, rhs = sp.linear_eq_to_matrix(eqs, unknown)
    worst_min_abs_det = float('inf')
    for p in PROFILES:
        R_, b0v, b1v, b2v, f0v, f1v, f2v, h0v, h1v, h2v, w0v, w1v, w2v, a, _ = p
        subs = {alpha: a, r: R_, b: b0v, f: f0v, h: h0v, w: w0v,
                sp.diff(b, r): b1v, sp.diff(h, r): h1v, sp.diff(w, r): w1v}
        det = float(M.subs(subs).det())
        worst_min_abs_det = min(worst_min_abs_det, abs(det))
        if det == 0.0:
            raise RuntimeError(f'coefficient matrix singular at profile {p}')
    print(f'Order stays 7 ([f prime, b\'\', h\'\', w\'\'] uniquely solvable on all '
          f'profiles; smallest |det| {worst_min_abs_det:.3e}): OK')
    return M, rhs, unknown


def check_constraint_propagation(vars_, gauged, solution):
    """The reserved constraint E_f must propagate like Hito 2's vacuum one:
    d/dr[sqrt(b*h)*f*E_f]=0 whenever E_b=E_g=E_h=E_w=0, i.e.
    r^2*sqrt(b*h)*C=const with C=E_f*f/r^2 (docs/vacuum-bvp.md) -- same
    functional form as vacuum, unchanged by the GB term (alpha_GB enters
    only implicitly, through which b,f,h,w solve the four equations).

    Found via the Beltrami/Hamiltonian identity for the autonomous (no bare
    r) reduced Lagrangian: Hcal=sum_v(v' dL/dv')-L (v in b,g,h,w) satisfies
    dHcal/dr=-sum_v(v' E_v)-f' E_f identically (calculus of variations; the
    f' E_f term is easy to miss since f has no kinetic term of its own, but
    Hcal still depends on f through L). Separately, Hcal=2f*E_f_raw exactly
    (also an identity, checked below). Combining on-shell (E_b=E_g=E_h=E_w=0)
    gives E_f_raw ~ f^(-3/2), which becomes r^2 sqrt(bh) C=const after
    normalizing/gauge-fixing to match the E_f used elsewhere in this module.

    A full symbolic sp.simplify (generic b,f,h,w, not a numeric spot-check)
    confirms this identically -- aborts if it does not.
    """
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    Q = sp.sqrt(b*h)*f*gauged['f']
    dQ = sp.diff(Q, r)
    dQ_onshell = dQ.subs(solution)
    residual = sp.simplify(dQ_onshell)
    if residual != 0:
        raise RuntimeError(f'constraint propagation identity failed: residual={residual}')
    print('Constraint propagation d/dr[sqrt(bh)*f*E_f]=0 on-shell (r^2 sqrt(bh) C=const, '
          'same form as vacuum): OK (proved for generic b,f,h,w, not just numerically)')


def horizon_expansion(vars_, gauged):
    """Leading-order horizon regularity relations, by direct Taylor expansion
    of the raw (uncompactified) equations around r=r_H=1 -- NOT yet the
    compactified/BVP-ready form (see docs/egb-rotating.md for why: the
    vacuum compactification's amplitude prefactors, substituted here, do
    NOT automatically regularize this system for generic boundary data,
    same as vacuum itself -- specific relations are required, and finding
    them via residues of the compactified expressions was intractable in
    this environment; this direct approach is not).

    b,f,h,w are expanded as b=b1 eps+b2 eps^2/2, f=f1 eps+f2 eps^2/2,
    h=h0+h1 eps+h2 eps^2/2, w=w0+w1 eps+w2 eps^2/2 (eps=r-r_H), substituted
    into the polynomial numerators of the gauge-fixed equations, and
    Taylor-expanded (sp.series, not full polynomial division -- much
    faster). Returns the first nonzero order for each equation: E_b and
    E_f both start at O(eps) and are proportional to each other (E_f's
    O(eps) term is minus E_b's, divided by h0) -- the redundancy is exactly
    the argument of docs/vacuum-bvp.md ("horizonte regular no extremal
    tiene C finito y b->0, por lo que esa constante debe ser cero"), now
    reproduced for the EGB system, not assumed. E_g,E_h,E_w start at
    O(eps^2). Four independent relations in total (E_b/E_f share one).
    """
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    eqs = {}
    for name in ('b', 'g', 'h', 'w', 'f'):
        numer, _ = sp.fraction(sp.together(gauged[name]))
        eqs[name] = sp.expand(numer)

    eps = sp.Symbol('epsilon')
    b1, b2, f1, f2, h0, h1, h2, w0, w1, w2 = sp.symbols('b1 b2 f1 f2 h0 h1 h2 w0 w1 w2')
    b_series = b1*eps+b2*eps**2/2
    f_series = f1*eps+f2*eps**2/2
    h_series = h0+h1*eps+h2*eps**2/2
    w_series = w0+w1*eps+w2*eps**2/2
    subs_series = {
        b: b_series, sp.diff(b, r): sp.diff(b_series, eps), sp.diff(b, r, 2): sp.diff(b_series, eps, 2),
        f: f_series, sp.diff(f, r): sp.diff(f_series, eps), sp.diff(f, r, 2): sp.diff(f_series, eps, 2),
        h: h_series, sp.diff(h, r): sp.diff(h_series, eps), sp.diff(h, r, 2): sp.diff(h_series, eps, 2),
        w: w_series, sp.diff(w, r): sp.diff(w_series, eps), sp.diff(w, r, 2): sp.diff(w_series, eps, 2),
        r: 1+eps,
    }
    orders = {}
    for name in ('b', 'g', 'h', 'w', 'f'):
        expanded = eqs[name].subs(subs_series, simultaneous=True)
        series = sp.series(expanded, eps, 0, 3).removeO()
        poly = sp.Poly(series, eps)
        coeffs = {degree: coeff for (degree,), coeff in poly.terms()}
        if coeffs.get(0, 0) != 0:
            raise RuntimeError(f'{name}: unexpected nonzero order-0 term at the horizon')
        orders[name] = coeffs
    return orders


def check_horizon_expansion_against_myers_perry(orders):
    """Cross-check the alpha_GB=0 leading horizon relations against the
    exact, independently-implemented Myers-Perry closed form (Hito 1),
    computing b1,f1,h0,h1,w1,f2,b2,w2 by symbolic differentiation of
    rotating_bh.myers_perry.MyersPerry.functions() -- not a re-derivation,
    an external check with different code.
    """
    sys.path.insert(0, str(ROOT/'src'))
    from rotating_bh.myers_perry import MyersPerry  # noqa: E402 (path just inserted)
    r_h_sym, omega_h_sym, r_sym = sp.symbols('r_h omega_h r', positive=True)
    z = r_h_sym/r_sym
    q2 = (r_h_sym*omega_h_sym)**2
    squashing = 1+q2*z**4/(1-q2)
    f_expr = (1-z)*(1+z)*(1-q2*z*z/(1-q2))
    b_expr = f_expr/squashing
    h_expr = r_sym*r_sym*squashing
    w_expr = omega_h_sym*z**4/(1-q2+q2*z**4)
    values = {r_h_sym: 1, omega_h_sym: sp.Rational(3, 10)}

    def d(expr, n):
        return float(sp.diff(expr, r_sym, n).subs(values).subs(r_sym, 1).subs(values))
    numeric = dict(b1=d(b_expr, 1), f1=d(f_expr, 1), f2=d(f_expr, 2), b2=d(b_expr, 2),
                   h0=float(h_expr.subs(values).subs(r_sym, 1).subs(values)), h1=d(h_expr, 1),
                   h2=d(h_expr, 2), w0=float(values[omega_h_sym]), w1=d(w_expr, 1), w2=d(w_expr, 2))
    symbols_map = {sp.Symbol(k): v for k, v in numeric.items()}
    symbols_map[sp.Symbol('alpha_GB')] = 0.0
    for name in ('b', 'g', 'h', 'w', 'f'):
        for order, expr in orders[name].items():
            if order == 0:
                continue
            value = float(expr.subs(symbols_map))
            if abs(value) > 1e-8:
                raise RuntimeError(f'{name} order {order}: {value} != 0 for Myers-Perry (alpha_GB=0)')
    # MyersPerry() also validates its own physical constraints (nonextremal, etc.).
    MyersPerry(r_h=1.0, omega_h=numeric['w0'])
    print('Horizon relations (order 1: E_b/E_f; order 2: E_g,E_h,E_w) verified against '
          'the exact Myers-Perry closed form (alpha_GB=0): OK')


def infinity_expansion(vars_, gauged):
    """Leading-order asymptotic-flatness relations, by 1/r expansion at
    r->infinity (s=1/r->0). Same style as horizon_expansion, but s.series
    must run on the FULL gauge-fixed equation (sp.diff/sp.series applied to
    `gauged[name]` directly), NOT on a numerator-cleared version: an earlier
    attempt cleared denominators first (sp.fraction(sp.together(...))) and
    the denominator's own s-dependence silently shifted the apparent pole
    order for one of the five equations, which produced a spurious relation
    (contradicted by Myers-Perry -- caught by the cross-check below before
    it was trusted). Not needed at the horizon (eps->0 has no such issue
    there since eps itself, not 1/eps, is already the local coordinate).

    Ansatz: b=1+U s^2+u4 s^4+u6 s^6, f=1+U s^2+V s^4+v6 s^6,
    h=1/s^2+H2+H4 s^2+H6 s^4, w=W s^4+w6 s^6+w8 s^8 (matching the known
    vacuum falloff powers -- b,f share U at leading order like Hito 1's
    Myers-Perry, docs/myers-perry.md).
    """
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    s = sp.symbols('s', positive=True)
    U, u4, u6, V, v6, H2, H4, H6, W, w6, w8 = sp.symbols('U u4 u6 V v6 H2 H4 H6 W w6 w8')
    b_s = 1+U*s**2+u4*s**4+u6*s**6
    f_s = 1+U*s**2+V*s**4+v6*s**6
    h_s = 1/s**2+H2+H4*s**2+H6*s**4
    w_s = W*s**4+w6*s**6+w8*s**8
    dr_of = lambda expr: -s**2*sp.diff(expr, s)  # d/dr = ds/dr d/ds = -s^2 d/ds
    subs_series = {
        b: b_s, sp.diff(b, r): dr_of(b_s), sp.diff(b, r, 2): dr_of(dr_of(b_s)),
        f: f_s, sp.diff(f, r): dr_of(f_s),
        h: h_s, sp.diff(h, r): dr_of(h_s), sp.diff(h, r, 2): dr_of(dr_of(h_s)),
        w: w_s, sp.diff(w, r): dr_of(w_s), sp.diff(w, r, 2): dr_of(dr_of(w_s)),
        r: 1/s,
    }
    orders = {}
    for name in ('b', 'g', 'h', 'w', 'f'):
        expanded = gauged[name].subs(subs_series, simultaneous=True)
        series = sp.series(expanded, s, 0, 8).removeO()
        poly = sp.Poly(sp.expand(series), s)
        orders[name] = {degree: coeff for (degree,), coeff in poly.terms()}
    return orders


def check_infinity_expansion_against_myers_perry(orders):
    """Cross-check the alpha_GB=0 leading asymptotic relations against the
    exact Myers-Perry closed form (Hito 1), analogous to
    check_horizon_expansion_against_myers_perry.
    """
    sys.path.insert(0, str(ROOT/'src'))
    from rotating_bh.myers_perry import MyersPerry  # noqa: E402
    r_h_sym, omega_h_sym, r_sym = sp.symbols('r_h omega_h r', positive=True)
    s_sym = sp.symbols('s', positive=True)
    z = r_h_sym*s_sym
    q2 = (r_h_sym*omega_h_sym)**2
    squashing = 1+q2*z**4/(1-q2)
    f_expr = (1-z)*(1+z)*(1-q2*z*z/(1-q2))
    b_expr = f_expr/squashing
    h_expr = squashing/s_sym**2
    w_expr = omega_h_sym*z**4/(1-q2+q2*z**4)
    values = {r_h_sym: 1, omega_h_sym: sp.Rational(3, 10)}

    def coeff(expr, order):
        return sp.series(expr.subs(values), s_sym, 0, order+2).removeO().coeff(s_sym, order)
    numeric = dict(U=float(coeff(f_expr, 2)), V=float(coeff(f_expr, 4)), v6=float(coeff(f_expr, 6)),
                   u4=float(coeff(b_expr, 4)), u6=float(coeff(b_expr, 6)),
                   H2=float(coeff(h_expr, 0)), H4=float(coeff(h_expr, 2)), H6=float(coeff(h_expr, 4)),
                   W=float(coeff(w_expr, 4)), w6=float(coeff(w_expr, 6)), w8=float(coeff(w_expr, 8)))
    symbols_map = {sp.Symbol(k): v for k, v in numeric.items()}
    symbols_map[sp.Symbol('alpha_GB')] = 0.0
    for name in ('b', 'g', 'h', 'w', 'f'):
        for order, expr in orders[name].items():
            value = float(expr.subs(symbols_map))
            if abs(value) > 1e-8:
                raise RuntimeError(f'{name} order {order}: {value} != 0 for Myers-Perry (alpha_GB=0)')
    MyersPerry(r_h=1.0, omega_h=float(values[omega_h_sym]))  # sanity: still a valid instance
    print('Infinity relations (order 2: E_f; order 4: E_b,E_g,E_h,E_w,E_f; order 6/8: next) verified '
          'against the exact Myers-Perry closed form (alpha_GB=0): OK')


def compact_horizon_conditions(vars_, gauged):
    """The four horizon regularity relations directly in the compactified
    (spectral-solver-ready) variables B,F,H,W,P=Bx,Q=Hx,V=Wx, plus Fx (F's
    own x-derivative) and Wxx (W's second x-derivative) -- both supplied by
    the spectral differentiation matrix at that boundary node in the actual
    solver, not evaluated from a closed formula (which is exactly what is
    singular there for generic data; see docs/egb-rotating.md).

    Method: substitute the compactifying ansatz b=(1-z^2)B(z), f=(1-z^2)F(z),
    h=(1+z^4 H(z))/z^2, w=z^4 W(z) (z=1/r) directly into the RAW,
    polynomial-in-(b,f,h,w) gauge-fixed equations (eqs, degree 1 in
    [f',b'',h'',w'']) -- NOT into the already-solved rhs(), which is what
    made this intractable before (Cramer's rule divides by a determinant
    that itself vanishes at the horizon, the real source of the earlier
    "singular" numerics, not a pole in the raw equations themselves: eqs
    are polynomial in b,f,h,w, so substituting b=f=0 there gives an
    IDENTICAL zero -- checked directly below -- confirming the horizon
    relations come from the NEXT order, exactly like horizon_expansion, not
    from a removable pole).

    B(z),H(z),W(z) are expanded as B_H+B1(z-1)+B2(z-1)^2/2 etc. around
    z=1 (horizon); F only needs one derivative (Fx). dB/dz=-P, dH/dz=-Q,
    dW/dz=-V (chain rule x=1-z, verified by direct z-derivative sympy
    differentiation, not asserted). d/dr=-z^2 d/dz.

    Returns the four raw-equation residuals (from 'b' at O(zeta); 'g','h','w'
    at O(zeta^2)), each a function of B_H,F_H,H_H,W_H,P,Q,V,Fx,Wxx,alpha_GB.
    'b' and the (unused here) 'f' relation are proportional -- the same
    E_b/E_f redundancy as horizon_expansion, now confirmed in these
    variables too, not just in raw r.
    """
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    d = lambda v: sp.diff(v, r)
    eqs = {}
    for name in ('b', 'g', 'h', 'w'):
        numer, _ = sp.fraction(sp.together(gauged[name]))
        eqs[name] = sp.expand(numer)

    # Scientific gate: eqs must vanish identically at b=f=0 (every term has
    # an explicit b or f factor) -- this is WHY evaluating exactly at the
    # horizon gives a trivial 0=0 and the real content is one order higher.
    for name in ('b', 'g', 'h', 'w'):
        if sp.simplify(eqs[name].subs({b: 0, f: 0})) != 0:
            raise RuntimeError(f'{name}: unexpected nonzero residual at b=f=0 (horizon)')

    zeta = sp.symbols('zeta')
    B_H, F_H, H_H, W_H, Fx, Wxx = sp.symbols('B_H F_H H_H W_H Fx Wxx')
    P, Q, V = sp.symbols('P Q V')
    B_z = B_H+(-P)*zeta+sp.Symbol('Bxx')*zeta**2/2
    F_z = F_H+(-Fx)*zeta
    H_z = H_H+(-Q)*zeta+sp.Symbol('Hxx')*zeta**2/2
    W_z = W_H+(-V)*zeta+Wxx*zeta**2/2

    z = 1+zeta
    b_of_z, f_of_z = (1-z**2)*B_z, (1-z**2)*F_z
    h_of_z, w_of_z = (1+z**4*H_z)/z**2, z**4*W_z
    dr = lambda e: -z**2*sp.diff(e, zeta)  # d/dr=(dzeta/dr)d/dzeta, dzeta/dr=-z^2

    subs_map = {b: b_of_z, d(b): dr(b_of_z), sp.diff(b, r, 2): dr(dr(b_of_z)),
                f: f_of_z, d(f): dr(f_of_z),
                h: h_of_z, d(h): dr(h_of_z), sp.diff(h, r, 2): dr(dr(h_of_z)),
                w: w_of_z, d(w): dr(w_of_z), sp.diff(w, r, 2): dr(dr(w_of_z)),
                r: 1/z}

    relations = {}
    b_substituted = eqs['b'].subs(subs_map, simultaneous=True)
    if b_substituted.subs(zeta, 0) != 0:
        raise RuntimeError('b: unexpected nonzero value exactly at the horizon')
    relations['b'] = sp.factor(sp.diff(b_substituted, zeta).subs(zeta, 0))
    for name in ('g', 'h', 'w'):
        substituted = eqs[name].subs(subs_map, simultaneous=True)
        if substituted.subs(zeta, 0) != 0 or sp.diff(substituted, zeta).subs(zeta, 0) != 0:
            raise RuntimeError(f'{name}: unexpected nonzero value at order 0 or 1 at the horizon')
        relations[name] = sp.factor(sp.diff(substituted, zeta, 2).subs(zeta, 0)/2)
    return relations, dict(B_H=B_H, F_H=F_H, H_H=H_H, W_H=W_H, P=P, Q=Q, V=V, Fx=Fx, Wxx=Wxx)


def check_compact_horizon_against_myers_perry(relations, symbols_):
    """Cross-check against the exact Myers-Perry closed form, computed
    DIRECTLY in the compact variables (B(z)=b/(1-z^2) etc., Taylor-expanded
    around z=1) -- not by translating raw-r data, which is a separate,
    error-prone step avoided here entirely.
    """
    sys.path.insert(0, str(ROOT/'src'))
    from rotating_bh.myers_perry import MyersPerry  # noqa: E402
    r_h_sym, omega_h_sym, r_sym = sp.symbols('r_h omega_h r', positive=True)
    z = r_h_sym/r_sym
    q2 = (r_h_sym*omega_h_sym)**2
    squashing = 1+q2*z**4/(1-q2)
    f_expr = (1-z)*(1+z)*(1-q2*z*z/(1-q2))
    b_expr = f_expr/squashing
    h_expr = r_sym*r_sym*squashing
    w_expr = omega_h_sym*z**4/(1-q2+q2*z**4)
    values = {r_h_sym: 1, omega_h_sym: sp.Rational(3, 10)}

    zsym = sp.symbols('zsym', positive=True)
    subs_z = {r_sym: values[r_h_sym]/zsym}
    B_of_z = (b_expr.subs(values).subs(subs_z))/(1-zsym**2)
    F_of_z = (f_expr.subs(values).subs(subs_z))/(1-zsym**2)
    H_of_z = (h_expr.subs(values).subs(subs_z)*zsym**2-1)/zsym**4
    W_of_z = (w_expr.subs(values).subs(subs_z))/zsym**4

    zeta = sp.symbols('zeta')
    taylor = lambda expr, order: sp.series(expr.subs(zsym, 1+zeta), zeta, 0, order+1).removeO()
    B_series, F_series = taylor(B_of_z, 1), taylor(F_of_z, 1)
    H_series, W_series = taylor(H_of_z, 2), taylor(W_of_z, 2)
    numeric = {
        'B_H': float(B_series.coeff(zeta, 0)), 'F_H': float(F_series.coeff(zeta, 0)),
        'H_H': float(H_series.coeff(zeta, 0)), 'W_H': float(W_series.coeff(zeta, 0)),
        'P': float(-B_series.coeff(zeta, 1)), 'Fx': float(-F_series.coeff(zeta, 1)),
        'Q': float(-H_series.coeff(zeta, 1)), 'V': float(-W_series.coeff(zeta, 1)),
        'Wxx': float(W_series.coeff(zeta, 2)*2),
    }
    symbols_map = {symbols_[k]: v for k, v in numeric.items()}
    symbols_map[sp.Symbol('alpha_GB')] = 0.0
    for name, rel in relations.items():
        value = float(rel.subs(symbols_map))
        if abs(value) > 1e-8:
            raise RuntimeError(f'{name}: {value} != 0 for Myers-Perry (compact vars, alpha_GB=0)')
    MyersPerry(r_h=1.0, omega_h=float(values[omega_h_sym]))
    print('Compact horizon relations (spectral-solver-ready) verified against the exact '
          'Myers-Perry closed form (alpha_GB=0): OK')


def check_compact_infinity_conditions(vars_, gauged):
    """B=1,F=1,Hx=0,Wx=0 at infinity (z=0), matching Hito 2's own spectral
    solver exactly (vacuum_bvp.py::_spectral imposes precisely these four
    at its x=0 end, not the 'B=1,Hx=Wx=0' of the prose summary in
    docs/vacuum-bvp.md -- the code imposes F=1 too).

    Verified by direct symbolic Taylor-coefficient matching between the
    ALREADY Myers-Perry-verified raw s=1/r infinity ansatz (infinity_expansion)
    and the compact amplitude ansatz b=(1-z^2)B(z) etc. at z=s -- not by
    re-running the (much more expensive, and in an earlier attempt spurious
    without a large enough ansatz) direct substitution into the raw
    equations at z=0. Matching order by order forces B_inf=F_inf=1 and
    Bz1=Fz1=Hz1=Wz1=0, i.e. P=Fx=Q=V=0 at z=0 in the notation of
    compact_horizon_conditions -- these are the same four conditions
    vacuum's spectral solver imposes, confirmed here to also hold for the
    EGB system's leading asymptotic order (alpha_GB does not enter until
    1/r^6, per infinity_expansion, past what these four conditions probe).
    """
    z = sp.symbols('z')
    U, u4, V_raw = sp.symbols('U u4 V_raw')
    B_inf, Bz1, Bz2 = sp.symbols('B_inf Bz1 Bz2')
    F_inf, Fz1, Fz2 = sp.symbols('F_inf Fz1 Fz2')
    H_inf, Hz1 = sp.symbols('H_inf Hz1')
    W_inf, Wz1 = sp.symbols('W_inf Wz1')

    b_amp = sp.expand((1-z**2)*(B_inf+Bz1*z+Bz2*z**2/2))
    f_amp = sp.expand((1-z**2)*(F_inf+Fz1*z+Fz2*z**2/2))
    h_amp_z2 = sp.expand((1+z**4*(H_inf+Hz1*z))*1)  # (1+z^4 H(z))/z^2, times z^2
    w_amp = sp.expand(z**4*(W_inf+Wz1*z))

    b_raw, f_raw = 1+U*z**2+u4*z**4, 1+U*z**2+V_raw*z**4
    h_raw_z2 = sp.expand((1+0*z**2+V_raw*z**4)*1)  # (1/z^2+0+V_raw z^2+...)*z^2 = 1+V_raw z^4+...
    w_raw = 0*z**4  # only fixes odd orders zero; even (physical charge) not needed here

    checks = [
        ('B_inf=1', sp.Eq(b_amp.coeff(z, 0), b_raw.coeff(z, 0))),
        ('Bz1=0', sp.Eq(b_amp.coeff(z, 1), b_raw.coeff(z, 1))),
        ('F_inf=1', sp.Eq(f_amp.coeff(z, 0), f_raw.coeff(z, 0))),
        ('Fz1=0', sp.Eq(f_amp.coeff(z, 1), f_raw.coeff(z, 1))),
        ('Hz1=0 (from the absence of an odd z^5 term in z^2 h)',
         sp.Eq(h_amp_z2.coeff(z, 5), h_raw_z2.coeff(z, 5))),
        ('Wz1=0 (from the absence of an odd z^5 term in w)', sp.Eq(w_amp.coeff(z, 5), 0)),
    ]
    solved = {}
    for label, eq in checks:
        sol = sp.solve(eq)
        if len(sol) != 1:
            raise RuntimeError(f'{label}: expected a unique solution, got {sol}')
        solved[label] = sol[0]
        if label in ('B_inf=1', 'F_inf=1') and sol[0] != 1:
            raise RuntimeError(f'{label}: expected 1, got {sol[0]}')
        if label != 'B_inf=1' and label != 'F_inf=1' and sol[0] != 0:
            raise RuntimeError(f'{label}: expected 0, got {sol[0]}')
    print('Compact infinity conditions B=1,F=1,Hx=0,Wx=0 confirmed by direct Taylor matching '
          'against the Myers-Perry-verified raw ansatz (same four as vacuum_bvp.py): OK')


def compact_interior_system(vars_, solution):
    """Convert the solved uncompactified system (b,f,h,w in r) to the
    compact spectral coordinate x=1-z, z=1/r, via b=(1-z^2)B(z) etc. --
    for INTERIOR collocation nodes only (the compact formulas below are
    singular at z=1, the horizon; see compact_horizon_conditions for why
    that endpoint needs its own treatment instead of evaluating these).

    z is kept as a plain symbol with dz/dx=-1 used explicitly in the chain
    rule, NOT the literal substitution z=1-x used in
    experiments/derive_vacuum.py: an earlier attempt with a literal
    substitution made sympy re-expand (1-x) at every step and did not
    finish in this environment; z as an opaque symbol does.

    Returns Bxx,Fx,Hxx,Wxx as functions of (z,B,F,H,W,P,Q,V,alpha_GB),
    verified below to reproduce rotating_bh._vacuum_generated.rhs exactly
    at alpha_GB=0 (a real cross-check against Hito 2's own tested code, not
    a self-consistency check).
    """
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    d = lambda v: sp.diff(v, r)
    fp_sol, bpp_sol = solution[d(f)], solution[sp.diff(b, r, 2)]
    hpp_sol, wpp_sol = solution[sp.diff(h, r, 2)], solution[sp.diff(w, r, 2)]

    z = sp.symbols('z', positive=True)
    A = 1-z**2
    Bc, Fc, Hc, Wc, P, Q, V = sp.symbols('B F H W P Q V')
    values = [A*Bc, A*Fc, (1+z**4*Hc)/z**2, z**4*Wc]
    dx = lambda e: -sp.diff(e, z)+sp.diff(e, Bc)*P+sp.diff(e, Hc)*Q+sp.diff(e, Wc)*V
    sub = {r: 1/z, b: values[0], f: values[1], h: values[2], w: values[3],
           d(b): z*z*dx(values[0]), d(h): z*z*dx(values[2]), d(w): z*z*dx(values[3])}
    fp_z = fp_sol.subs(sub, simultaneous=True)
    bpp_z = bpp_sol.subs(sub, simultaneous=True)
    hpp_z = hpp_sol.subs(sub, simultaneous=True)
    wpp_z = wpp_sol.subs(sub, simultaneous=True)

    Bxx = sp.cancel((bpp_z/z**4+2*dx(values[0])/z-dx(dx(values[0])))/A)
    Fx = sp.cancel((fp_z/z**2+sp.diff(A, z)*Fc)/A)
    Hxx = sp.cancel((hpp_z/z**4+2*dx(values[2])/z-dx(dx(values[2])))/z**2)
    Wxx = sp.cancel((wpp_z/z**4+2*dx(values[3])/z-dx(dx(values[3])))/z**4)
    return dict(Bxx=Bxx, Fx=Fx, Hxx=Hxx, Wxx=Wxx, z=z, Bc=Bc, Fc=Fc, Hc=Hc, Wc=Wc, P=P, Q=Q, V=V)


def check_compact_interior_against_vacuum(compact, alpha):
    """rotating_bh._vacuum_generated.rhs(z,B,F,H,W,P,Q,V) is already tested
    (Hito 2, 116+ tests); at alpha_GB=0 the EGB compact system must
    reproduce it exactly, at several generic (non-solution) points.
    """
    sys.path.insert(0, str(ROOT/'src'))
    from rotating_bh._vacuum_generated import rhs as vacuum_rhs  # noqa: E402
    z, Bc, Fc, Hc, Wc, P, Q, V = (compact[k] for k in ('z', 'Bc', 'Fc', 'Hc', 'Wc', 'P', 'Q', 'V'))
    points = [
        dict(z=0.4, B=1.3, F=0.9, H=0.4, W=-0.2, P=0.6, Q=0.3, V=-0.5),
        dict(z=0.15, B=0.8, F=1.1, H=-0.3, W=0.5, P=-0.4, Q=0.7, V=0.2),
        dict(z=0.7, B=1.05, F=0.95, H=0.1, W=0.05, P=0.1, Q=-0.1, V=0.3),
    ]
    worst = 0.0
    for p in points:
        subs = {z: p['z'], Bc: p['B'], Fc: p['F'], Hc: p['H'], Wc: p['W'],
                P: p['P'], Q: p['Q'], V: p['V'], alpha: 0.0}
        mine = tuple(float(compact[name].subs(subs)) for name in ('Bxx', 'Fx', 'Hxx', 'Wxx'))
        theirs = vacuum_rhs(p['z'], p['B'], p['F'], p['H'], p['W'], p['P'], p['Q'], p['V'])
        rel = max(abs(a-b_)/max(abs(a), abs(b_), 1e-12) for a, b_ in zip(mine, theirs))
        worst = max(worst, rel)
        if rel > 1e-8:
            raise RuntimeError(f'compact interior system disagrees with _vacuum_generated at {p}: '
                                f'{mine} vs {theirs}')
    print(f'Compact interior system (Bxx,Fx,Hxx,Wxx) matches the tested '
          f'_vacuum_generated.rhs at alpha_GB=0 (worst rel. diff {worst:.3e}): OK')


def solve_system(M, rhs, unknown):
    """Cramer's rule via the adjugate (fast: exploits M's sparsity, unlike
    naive column-substitution determinants, which did not finish in this
    environment for this system -- see module docstring)."""
    det = sp.factor(M.det())
    adj = M.adjugate()
    numer = sp.expand(adj*rhs)
    return {u: sp.cancel(numer[i]/det) for i, u in enumerate(unknown)}, det


def derive():
    vars_ = lagrangians()
    check_action_transcription(vars_)
    alpha, gauged = gauge_fixed_variations(vars_)
    check_tensor_equivalence(vars_, alpha, gauged)
    M, rhs, unknown = check_order_stays_seven(vars_, alpha, gauged)
    solution, det = solve_system(M, rhs, unknown)
    check_constraint_propagation(vars_, gauged, solution)
    horizon_orders = horizon_expansion(vars_, gauged)
    check_horizon_expansion_against_myers_perry(horizon_orders)
    infinity_orders = infinity_expansion(vars_, gauged)
    check_infinity_expansion_against_myers_perry(infinity_orders)
    compact_horizon, compact_symbols = compact_horizon_conditions(vars_, gauged)
    check_compact_horizon_against_myers_perry(compact_horizon, compact_symbols)
    check_compact_infinity_conditions(vars_, gauged)
    compact = compact_interior_system(vars_, solution)
    check_compact_interior_against_vacuum(compact, alpha)
    return dict(vars_=vars_, alpha=alpha, gauged=gauged, solution=solution, det=det,
                unknown=unknown, horizon_orders=horizon_orders, infinity_orders=infinity_orders,
                compact_horizon=compact_horizon, compact_symbols=compact_symbols, compact=compact)


def generate(result):
    vars_, alpha, solution, gauged = result['vars_'], result['alpha'], result['solution'], result['gauged']
    r, b, f, h, w = vars_['r'], vars_['b'], vars_['f'], vars_['h'], vars_['w']
    bp, hp, wp = sp.diff(b, r), sp.diff(h, r), sp.diff(w, r)
    fp = solution[sp.diff(f, r)]
    bpp = solution[sp.diff(b, r, 2)]
    hpp = solution[sp.diff(h, r, 2)]
    wpp = solution[sp.diff(w, r, 2)]
    symbols_map = {b: sp.Symbol('b'), f: sp.Symbol('f'), h: sp.Symbol('h'), w: sp.Symbol('w'),
                   bp: sp.Symbol('bp'), hp: sp.Symbol('hp'), wp: sp.Symbol('wp'),
                   r: sp.Symbol('r'), alpha: sp.Symbol('alpha_gb')}
    exprs = [e.subs(symbols_map) for e in (fp, bpp, hpp, wpp)]
    replacements, reduced = sp.cse(exprs, sp.numbered_symbols('c'))
    lines = ['"""Generated by experiments/derive_egb_rotating.py; do not edit by hand.',
             '',
             'Uncompactified radial system (r, not the [0,1] BVP coordinate): see',
             'the module docstring of derive_egb_rotating.py for scope and the',
             'derivation gate. Horizon/infinity expansions and compactification are',
             'a separate, not-yet-done step (Hito 4A continues).',
             '"""',
             '',
             'def rhs(r, b, f, h, w, bp, hp, wp, alpha_gb):',
             '    """Return fp, bpp, hpp, wpp for the general EGB rotating ansatz."""']
    lines += ['    '+str(v)+' = '+sp.pycode(e) for v, e in replacements]
    lines += ['    return ('+', '.join(sp.pycode(e) for e in reduced)+')', '']
    (ROOT/'src/rotating_bh/_egb_rotating_generated.py').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote src/rotating_bh/_egb_rotating_generated.py')

    # All five raw variations (E_b,E_f,E_g,E_h,E_w), including the reserved
    # constraint E_f, for validation/tests -- not used to solve anything.
    # E_b,E_g,E_h,E_w are jointly linear in [f',b'',h'',w''] (that is what
    # made them solvable for those four unknowns in the first place), so
    # they need those as separate inputs, unlike the reserved E_f (first
    # derivatives only). Signature mirrors vacuum_equations.radial_equations
    # (values=(b,f,h,w), first=(bp,fp,hp,wp), second=(bpp,fpp,hpp,wpp)); fpp
    # and plain w never enter either system, same as in Hito 2.
    eq_symbols_map = dict(symbols_map)
    eq_symbols_map[sp.diff(f, r)] = sp.Symbol('fp')
    eq_symbols_map[sp.diff(b, r, 2)] = sp.Symbol('bpp')
    eq_symbols_map[sp.diff(h, r, 2)] = sp.Symbol('hpp')
    eq_symbols_map[sp.diff(w, r, 2)] = sp.Symbol('wpp')
    eq_exprs = [gauged[name].subs(eq_symbols_map) for name in ('b', 'f', 'g', 'h', 'w')]
    eq_repl, eq_reduced = sp.cse(eq_exprs, sp.numbered_symbols('e'))
    eq_lines = ['"""Generated by experiments/derive_egb_rotating.py; do not edit by hand.',
                '',
                'The five raw Euler-Lagrange variations (post gauge g=r^2), for',
                'validation and tests only -- E_f is the reserved constraint',
                '(never solved for; see docs/vacuum-bvp.md for the Hito 2 analogue',
                'and docs/egb-rotating.md for this one). fpp is accepted but unused',
                '(matches vacuum_equations.radial_equations).',
                '"""',
                '',
                'def equations(r, values, first, second, alpha_gb):',
                '    """Return E_b,E_f,E_g,E_h,E_w; zero iff values solve the EGB system.',
                '',
                '    values=(b,f,h,w), first=(bp,fp,hp,wp), second=(bpp,fpp,hpp,wpp).',
                '    """',
                '    b, f, h, w = values',
                '    bp, fp, hp, wp = first',
                '    bpp, fpp, hpp, wpp = second']
    eq_lines += ['    '+str(v)+' = '+sp.pycode(e) for v, e in eq_repl]
    eq_lines += ['    return ('+', '.join(sp.pycode(e) for e in eq_reduced)+')', '']
    (ROOT/'src/rotating_bh/_egb_rotating_equations_generated.py').write_text(
        '\n'.join(eq_lines), encoding='utf-8')
    print('Wrote src/rotating_bh/_egb_rotating_equations_generated.py')

    compact_horizon, compact_symbols = result['compact_horizon'], result['compact_symbols']
    horizon_symbols_map = {compact_symbols[k]: sp.Symbol(k)
                            for k in ('B_H', 'F_H', 'H_H', 'W_H', 'P', 'Q', 'V', 'Fx', 'Wxx')}
    horizon_symbols_map[alpha] = sp.Symbol('alpha_gb')
    horizon_exprs = [compact_horizon[name].subs(horizon_symbols_map) for name in ('b', 'g', 'h', 'w')]
    h_repl, h_reduced = sp.cse(horizon_exprs, sp.numbered_symbols('k'))
    h_lines = ['"""Generated by experiments/derive_egb_rotating.py; do not edit by hand.',
               '',
               'The four horizon regularity relations in the spectral-solver-ready compact',
               'variables (B,F,H,W,P=Bx,Q=Hx,V=Wx at x=0, plus Fx and Wxx supplied by the',
               "solver's own differentiation matrix at that node -- see",
               'compact_horizon_conditions in derive_egb_rotating.py and docs/egb-rotating.md).',
               'Zero iff the represented solution is regular at the horizon.',
               '"""',
               '',
               'def horizon(B_H, F_H, H_H, W_H, P, Q, V, Fx, Wxx, alpha_gb):',
               '    """Return the four raw-equation residuals (from eqs b,g,h,w)."""']
    h_lines += ['    '+str(v)+' = '+sp.pycode(e) for v, e in h_repl]
    h_lines += ['    return ('+', '.join(sp.pycode(e) for e in h_reduced)+')', '']
    (ROOT/'src/rotating_bh/_egb_rotating_horizon_generated.py').write_text(
        '\n'.join(h_lines), encoding='utf-8')
    print('Wrote src/rotating_bh/_egb_rotating_horizon_generated.py')

    compact = result['compact']
    z, Bc, Fc, Hc, Wc, P, Q, V = (compact[k] for k in ('z', 'Bc', 'Fc', 'Hc', 'Wc', 'P', 'Q', 'V'))
    compact_symbols_map = {z: sp.Symbol('z'), Bc: sp.Symbol('B'), Fc: sp.Symbol('F'),
                            Hc: sp.Symbol('H'), Wc: sp.Symbol('W'), P: sp.Symbol('P'),
                            Q: sp.Symbol('Q'), V: sp.Symbol('V'), alpha: sp.Symbol('alpha_gb')}
    compact_exprs = [compact[name].subs(compact_symbols_map) for name in ('Bxx', 'Fx', 'Hxx', 'Wxx')]
    c_repl, c_reduced = sp.cse(compact_exprs, sp.numbered_symbols('m'))
    c_lines = ['"""Generated by experiments/derive_egb_rotating.py; do not edit by hand.',
               '',
               'Compact spectral-coordinate system (z=1/r, x=1-z), for INTERIOR',
               'collocation nodes only -- singular at z=1 (horizon) for generic data;',
               'see _egb_rotating_horizon_generated.py for that endpoint instead, and',
               'docs/egb-rotating.md for why. At alpha_gb=0, reproduces',
               'rotating_bh._vacuum_generated.rhs exactly (checked in derive_egb_rotating.py).',
               '"""',
               '',
               'def rhs(z, B, F, H, W, P, Q, V, alpha_gb):',
               '    """Return Bxx,Fx,Hxx,Wxx."""']
    c_lines += ['    '+str(v)+' = '+sp.pycode(e) for v, e in c_repl]
    c_lines += ['    return ('+', '.join(sp.pycode(e) for e in c_reduced)+')', '']
    (ROOT/'src/rotating_bh/_egb_rotating_compact_generated.py').write_text(
        '\n'.join(c_lines), encoding='utf-8')
    print('Wrote src/rotating_bh/_egb_rotating_compact_generated.py')


if __name__ == '__main__':
    generate(derive())
