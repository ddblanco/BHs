"""Derive the static (w=0, h=r^2) D=5 EGB field equations from the tensor.

Run from project/. Mirrors, symbolically and index-for-index, the exact
algorithm already validated numerically in `rotating_bh.einstein.curvature`
and `rotating_bh.einstein.gauss_bonnet` (constant-curvature and Minkowski
regression tests). No reduced action, no reference solution and no
Myers-Perry/EGB closed form enter this derivation; the closed form is only
substituted at the end, as a check, not as an input.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
for key in ('TEMP', 'TMP'):
    (ROOT/'.cache/tmp').mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(ROOT/'.cache/tmp')
import sympy as sp

import sys
sys.path.insert(0, str(ROOT/'src'))
from rotating_bh.einstein import metric_from_ansatz


def symbolic_curvature(metric, coordinates):
    """Sympy mirror of rotating_bh.einstein.curvature, index for index."""
    n = len(coordinates)
    g = sp.Matrix(metric)
    inverse = g.inv()

    def dg_(k, i, j):
        return sp.diff(g[i, j], coordinates[k])

    connection_lower = [[[dg_(b, a, c)+dg_(c, a, b)-dg_(a, b, c)
                           for c in range(n)] for b in range(n)] for a in range(n)]
    gamma = [[[sp.Rational(1, 2)*sum(inverse[i, a]*connection_lower[a][b][c] for a in range(n))
               for c in range(n)] for b in range(n)] for i in range(n)]
    dgamma = [[[[sp.diff(gamma[i][b][c], coordinates[k])
                 for c in range(n)] for b in range(n)] for i in range(n)] for k in range(n)]

    def riemann(rho, sigma, mu, nu):
        return (dgamma[mu][rho][nu][sigma]-dgamma[nu][rho][mu][sigma]
                + sum(gamma[rho][mu][l]*gamma[l][nu][sigma] for l in range(n))
                - sum(gamma[rho][nu][l]*gamma[l][mu][sigma] for l in range(n)))

    riemann_mixed = [[[[sp.expand(riemann(r, s, m, nn)) for nn in range(n)]
                       for m in range(n)] for s in range(n)] for r in range(n)]
    ricci = sp.Matrix(n, n, lambda s, nn: sum(riemann_mixed[r][s][r][nn] for r in range(n)))
    ricci = sp.simplify(ricci)
    scalar = sp.simplify(sum(inverse[i, j]*ricci[i, j] for i in range(n) for j in range(n)))
    einstein = sp.simplify(ricci-sp.Rational(1, 2)*scalar*g)
    return dict(g=g, inverse=inverse, riemann=riemann_mixed, ricci=ricci,
                scalar=scalar, einstein=einstein)


def symbolic_gauss_bonnet(curv):
    n = curv['g'].shape[0]
    g, inverse, riemann, ricci, scalar = (curv[k] for k in
                                           ('g', 'inverse', 'riemann', 'ricci', 'scalar'))
    riemann_lower = [[[[sum(g[r, a]*riemann[a][s][m][nn] for a in range(n))
                        for nn in range(n)] for m in range(n)] for s in range(n)] for r in range(n)]

    def raise_all(rl):
        out = [[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for s in range(n):
                for m in range(n):
                    for nn in range(n):
                        out[r][s][m][nn] = sum(
                            inverse[r, a]*inverse[s, b]*inverse[m, c]*inverse[nn, d]*rl[a][b][c][d]
                            for a in range(n) for b in range(n) for c in range(n) for d in range(n))
        return out
    riemann_upper = raise_all(riemann_lower)
    riemann_squared = sp.expand(sum(riemann_lower[a][b][c][d]*riemann_upper[a][b][c][d]
                                     for a in range(n) for b in range(n)
                                     for c in range(n) for d in range(n)))
    ricci_mixed = sp.Matrix(n, n, lambda a, c: sum(inverse[a, b]*ricci[b, c] for b in range(n)))
    ricci_squared = sp.expand(sum(ricci_mixed[a, b]*ricci_mixed[b, a]
                                   for a in range(n) for b in range(n)))
    lagrangian = sp.simplify(scalar**2-4*ricci_squared+riemann_squared)

    ricci_upper = sp.Matrix(n, n, lambda r, s: sum(
        inverse[r, a]*inverse[s, b]*ricci[a, b] for a in range(n) for b in range(n)))
    riemann_nu_up = [[[[sum(g[nn, a]*riemann_upper[a][r][s][t] for a in range(n))
                        for t in range(n)] for s in range(n)] for r in range(n)] for nn in range(n)]
    a_term = sp.Matrix(n, n, lambda m, nn: sum(
        riemann_lower[m][r][s][t]*riemann_nu_up[nn][r][s][t]
        for r in range(n) for s in range(n) for t in range(n)))
    b_term = sp.Matrix(n, n, lambda m, nn: sum(
        riemann_lower[m][r][nn][s]*ricci_upper[r, s] for r in range(n) for s in range(n)))
    c_term = sp.Matrix(n, n, lambda m, nn: sum(
        ricci[m, a]*inverse[a, b]*ricci[b, nn] for a in range(n) for b in range(n)))
    h_tensor = sp.simplify(2*(a_term-2*b_term-2*c_term+scalar*ricci)-sp.Rational(1, 2)*g*lagrangian)
    return dict(H=h_tensor, lagrangian=lagrangian)


def derive():
    r, theta = sp.symbols('r theta', positive=True)
    t, phi1, phi2 = sp.symbols('t phi1 phi2', real=True)
    b, f = (sp.Function(name)(r) for name in ('b', 'f'))
    metric = metric_from_ansatz(r, theta, b, f, r**2, 0)
    coordinates = (t, r, theta, phi1, phi2)
    print('metric (should be diagonal):')
    sp.pprint(metric)

    curv = symbolic_curvature(metric, coordinates)
    print('Ricci scalar:')
    sp.pprint(curv['scalar'])
    gb = symbolic_gauss_bonnet(curv)

    alpha = sp.Symbol('alpha_GB')
    field = sp.simplify(curv['einstein']+alpha*gb['H'])
    print('Field equation diagonal entries (tt, rr, thetatheta, phi1phi1, phi2phi2):')
    for i in range(5):
        print(f'  [{i},{i}] =')
        sp.pprint(sp.simplify(field[i, i]))

    return dict(coordinates=coordinates, b=b, f=f, r=r, theta=theta,
                alpha=alpha, field=field, curv=curv, gb=gb)


def verify_and_generate(out):
    """Scientific gate: derive f', confirm b=f and Bianchi consistency of the
    angular equation, then generate the compact ODE. Raises if any check fails
    instead of adjusting signs to force a match.
    """
    b, f, r, alpha = out['b'], out['f'], out['r'], out['alpha']
    field = out['field']

    # tt=0, divided by the shared factor 3*b/(2r^3): gives f' with no b at all.
    tt = sp.simplify(field[0, 0]*2*r**3/(3*b))
    fp = sp.Symbol('Fp')
    solved = sp.solve(sp.Eq(tt.subs(sp.Derivative(f, r), fp), 0), fp)
    if len(solved) != 1:
        raise RuntimeError(f'expected a unique f-prime root, got {solved}')
    fprime_expr = sp.simplify(solved[0])
    print('f-prime (r):', fprime_expr)

    # rr=0 must force b'/b=f'/f (i.e. b=C f, fixed to C=1 by both -> 1 at infinity).
    rr = sp.simplify(field[1, 1]*2*r**3*b*f/3)
    bp = sp.Symbol('Bp')
    rr_at_b_eq_f = rr.subs({b: f, sp.Derivative(b, r): fp, sp.Derivative(f, r): fp})
    rr_residual = sp.simplify(rr_at_b_eq_f.subs(fp, fprime_expr))
    if rr_residual != 0:
        raise RuntimeError(f'rr equation not solved by b=f with the derived f-prime: {rr_residual}')

    # theta-theta must vanish identically (Bianchi), not merely at f=1.
    fpp_expr = sp.simplify(sp.diff(fprime_expr, r).subs(sp.Derivative(f, r), fprime_expr))
    theta_residual = field[2, 2].subs({b: f})
    theta_residual = theta_residual.subs(sp.Derivative(b, (r, 2)), fpp_expr)
    theta_residual = theta_residual.subs(sp.Derivative(f, (r, 2)), fpp_expr)
    theta_residual = theta_residual.subs(sp.Derivative(b, r), fprime_expr)
    theta_residual = theta_residual.subs(sp.Derivative(f, r), fprime_expr)
    theta_residual = sp.simplify(theta_residual)
    if theta_residual != 0:
        raise RuntimeError(f'theta-theta equation not implied by tt+rr+b=f: {theta_residual}')
    print('Bianchi/theta-theta consistency: OK (residual identically 0)')

    # Independent oracle check: the closed form of 1010.0860v1 eq. (2.20),
    # written with r_H=1, must solve the same f-prime ODE exactly.
    rh = sp.Symbol('r_H', positive=True)
    q = sp.Symbol('q')
    quadratic = sp.Eq(r**2*q+2*alpha*q**2, rh**2+2*alpha)
    q_of_r = sp.solve(quadratic, q)
    q_of_r = [sol for sol in q_of_r if sp.simplify(sp.limit(sol, alpha, 0)-rh**2/r**2) == 0]
    if len(q_of_r) != 1:
        raise RuntimeError(f'expected a unique GR-continuous root, got {q_of_r}')
    q_of_r = q_of_r[0]
    f_closed = 1-q_of_r
    lhs = sp.diff(f_closed, r)
    rhs_closed = fprime_expr.subs(f, f_closed)
    oracle_residual = sp.simplify(lhs-rhs_closed)
    if oracle_residual != 0:
        raise RuntimeError(f'closed-form oracle does not solve the derived ODE: {oracle_residual}')
    print('Closed-form oracle (eq. 2.20) satisfies the derived ODE: OK')

    # Compact coordinate x=1-r_H/r, generated function takes z=1-x=r_H/r=1/r
    # (r_H=1 units), matching the convention already used for vacuum_bvp.
    z = sp.Symbol('z', positive=True)
    alpha_hat = sp.Symbol('alpha_hat')
    F = sp.Symbol('F')
    fx_of_r = fprime_expr*sp.diff(1/z, z)  # dr/dz at r=1/z, then dF/dx=-dF/dz
    fx = sp.simplify(-fx_of_r.subs({r: 1/z, f: F, alpha: alpha_hat}))
    print('Fx(z,F,alpha_hat) =', fx)

    lines = ['"""Generated by experiments/derive_static_egb.py; do not edit by hand."""',
             '', 'def rhs(z, F, alpha_hat):',
             '    """Return Fx=dF/dx for the static EGB ansatz; z=1-x=r_H/r."""',
             '    return '+sp.pycode(fx), '']
    (ROOT/'src/rotating_bh/_static_egb_generated.py').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote src/rotating_bh/_static_egb_generated.py')


if __name__ == '__main__':
    verify_and_generate(derive())
