"""Derive the extremal near-horizon entropy function for D=5 equal-spin EGB.

Deliverable 5 of `plans/2026-09-14-hito-8.md`: an independent, ODE-free route
to the extremal branch. For equal angular momenta the extremal near-horizon
geometry is homogeneous -- its isometry group is SL(2,R) x SU(2) x U(1) and
the only invariants are three constants and one rotation parameter -- so Sen's
entropy function reduces the whole problem to *algebra*. Nothing here touches
the spectral solver, the Hito 4A derivation, or the generated radial system;
that is the point, since this is meant to confirm them from outside.

The ansatz, in coordinates (t,r,theta,phi,psi) with the SU(2) invariant forms
written out (sigma_1^2+sigma_2^2 = dtheta^2+sin^2theta dphi^2, sigma_3 =
dpsi+cos theta dphi):

    ds^2 = v1(-r^2 dt^2 + dr^2/r^2)
           + (v2/4)(dtheta^2 + sin^2 theta dphi^2)
           + (v3/4)(dpsi + cos theta dphi + k r dt)^2

R and L_GB come out constant on it, as homogeneity requires, and

    f(v,k) = Int dtheta dphi dpsi sqrt(-g) (R + alpha_GB L_GB)/(16 pi)
           = (pi/8) v1 v2 sqrt(v3) (R + alpha_GB L_GB)

using Vol(theta,phi,psi) = 16 pi^2 for the sin theta measure. The extremal
solution is `df/dv_i = 0`; the angular momentum conjugate to the U(1) is
`J = df/dk`, and the Wald entropy is the Legendre transform
`S = 2 pi (k J - f)`.

**Nothing in this file is calibrated against the answer.** At alpha_GB=0 the
system solves exactly to `k=1, v2=4 v1, v3=8 v1`, giving `S = 2 pi J` and
`J = 2 sqrt(2) pi v1^{3/2}` -- and `S = 2 pi J` is precisely extremal
Myers-Perry (`S=2 pi^2 a^3`, `J = pi a^3` for each spin, at `r_+=a`,
`m=4a^2`). That agreement fixes no constant: the 1/(16 pi G) of the action and
the 2 pi of the Legendre transform were both put in beforehand.

Two gates decide whether this file is written:

1. `R` and `L_GB` must agree with the *independent* numeric evaluator of
   `einstein.py` -- the same one that gates every accepted solution of Hitos
   4-6 -- on the near-horizon metric at generic parameters and angles.
2. The extremal solution must satisfy the full field equations
   `G_{mu nu} + alpha_GB H_{mu nu} = 0` component by component, again through
   `einstein.py`. Extremising a reduced functional is not the same as solving
   the equations; this is what turns one into the other.
3. It must satisfy eqs. (4.11) and (4.12) of arXiv:1010.0860v1 **as
   published**. This calculation is not new -- section 4.2 of that paper does
   it, in this same ansatz -- and the honest use of that fact is to treat the
   published algebra as an external check at finite coupling rather than to
   present the rederivation as a result. The check also pins the coupling
   normalisation: the equations close with `alpha_paper = 4 alpha_GB`, which is
   what `docs/convenciones.md` records from the action.
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from rotating_bh.einstein import compile_metric, curvature, gauss_bonnet  # noqa: E402

TARGET = ROOT/'src'/'rotating_bh'/'_near_horizon_generated.py'

COORDINATES = sp.symbols('t r theta phi psi', real=True)
V1, V2, V3, K = sp.symbols('v1 v2 v3 k', positive=True)


def near_horizon_metric(v1=V1, v2=V2, v3=V3, k=K):
    """The homogeneous extremal near-horizon metric, as a 5x5 sympy Matrix."""
    t, r, theta, phi, psi = COORDINATES
    fibre = sp.Matrix([[k*r, 0, 0, sp.cos(theta), 1]])
    metric = sp.zeros(5)
    metric[0, 0] = -v1*r**2
    metric[1, 1] = v1/r**2
    metric[2, 2] = v2/4
    metric[3, 3] = v2/4*sp.sin(theta)**2
    return metric + (v3/4)*(fibre.T*fibre)


def curvature_scalars():
    """Symbolic R and L_GB on that metric, by explicit Levi-Civita algebra."""
    metric = sp.simplify(near_horizon_metric())
    inverse = sp.simplify(metric.inv())
    n = 5

    def d(expression, index):
        return sp.diff(expression, COORDINATES[index])

    dg = [[[d(metric[i, j], c) for j in range(n)] for i in range(n)] for c in range(n)]
    gamma = [[[sp.simplify(sum(inverse[a, b]*(dg[i][b][j]+dg[j][b][i]-dg[b][i][j])
                               for b in range(n))/2)
               for j in range(n)] for i in range(n)] for a in range(n)]
    riemann = [[[[sp.simplify(d(gamma[a][j][c], i)-d(gamma[a][i][c], j)
                              + sum(gamma[a][i][b]*gamma[b][j][c]
                                    - gamma[a][j][b]*gamma[b][i][c] for b in range(n)))
                  for j in range(n)] for i in range(n)] for c in range(n)] for a in range(n)]
    ricci = sp.Matrix(n, n, lambda c, j: sp.simplify(sum(riemann[a][c][a][j]
                                                         for a in range(n))))
    scalar = sp.simplify(sum(inverse[i, j]*ricci[i, j] for i in range(n) for j in range(n)))
    lower = [[[[sp.simplify(sum(metric[a, b]*riemann[b][c][i][j] for b in range(n)))
                for j in range(n)] for i in range(n)] for c in range(n)] for a in range(n)]
    upper = [[[[sp.simplify(sum(inverse[a, p]*inverse[c, q]*inverse[i, u]*inverse[j, w]
                                * lower[p][q][u][w]
                                for p in range(n) for q in range(n)
                                for u in range(n) for w in range(n)))
                for j in range(n)] for i in range(n)] for c in range(n)] for a in range(n)]
    riemann_squared = sp.simplify(sum(lower[a][c][i][j]*upper[a][c][i][j]
                                      for a in range(n) for c in range(n)
                                      for i in range(n) for j in range(n)))
    mixed = sp.simplify(inverse*ricci)
    ricci_squared = sp.simplify(sum(mixed[i, j]*mixed[j, i]
                                    for i in range(n) for j in range(n)))
    density = sp.simplify(scalar**2-4*ricci_squared+riemann_squared)
    # sin(theta) > 0 on (0, pi), so the |sin| sympy leaves behind is just sin;
    # dividing inside the square root avoids carrying the Abs at all.
    root = sp.sqrt(sp.simplify(-metric.det()/sp.sin(COORDINATES[2])**2))
    return sp.simplify(scalar), density, root


def _numeric_gate(scalar, density, root):
    """Gate 1: the symbolic scalars against einstein.py's numeric evaluator."""
    evaluate = compile_metric(near_horizon_metric(), COORDINATES, (V1, V2, V3, K))
    scalar_fn = sp.lambdify((V1, V2, V3, K), scalar, 'numpy')
    density_fn = sp.lambdify((V1, V2, V3, K), density, 'numpy')
    root_fn = sp.lambdify((V1, V2, V3, K), root, 'numpy')
    rng = np.random.default_rng(20260914)
    worst = 0.
    for _ in range(8):
        v = 0.5+2*rng.random(3)
        k = 0.3+rng.random()
        point = (0., 0.7+rng.random(), 0.4+rng.random(), 0.2, 0.3)
        arrays = evaluate(*point, *v, k)
        result = curvature(*arrays)
        gb = gauss_bonnet(arrays[0], result['inverse'], result['riemann'],
                          result['ricci'], result['scalar'])
        expected = (scalar_fn(*v, k), density_fn(*v, k))
        found = (result['scalar'], gb['lagrangian'])
        determinant = np.sqrt(-np.linalg.det(arrays[0]))/np.sin(point[2])
        worst = max(worst,
                    abs(found[0]-expected[0])/max(abs(expected[0]), 1.),
                    abs(found[1]-expected[1])/max(abs(expected[1]), 1.),
                    abs(determinant-root_fn(*v, k))/max(abs(root_fn(*v, k)), 1.))
    return worst


def entropy_function(alpha):
    """f(v1,v2,v3,k) for a given symbolic or numeric coupling."""
    scalar, density, _ = curvature_scalars()
    return sp.pi/8*V1*V2*sp.sqrt(V3)*(scalar + alpha*density)


def _field_equation_gate(extremal):
    """Gate 2: the full G+alpha H on the extremal near-horizon solution."""
    evaluate = compile_metric(near_horizon_metric(), COORDINATES, (V1, V2, V3, K))
    worst = 0.
    for alpha, v1, v2, v3, k in extremal:
        point = (0., 1.3, 0.9, 0.2, 0.3)
        arrays = evaluate(*point, v1, v2, v3, k)
        result = curvature(*arrays)
        gb = gauss_bonnet(arrays[0], result['inverse'], result['riemann'],
                          result['ricci'], result['scalar'])
        field = result['einstein']+alpha*gb['H']
        scale = np.max(np.abs(result['einstein']))+np.max(np.abs(arrays[0]))
        worst = max(worst, float(np.max(np.abs(field))/scale))
    return worst


def main():
    scalar, density, root = curvature_scalars()
    print('R      =', scalar)
    print('L_GB   =', density)
    print('sqrt-g =', root, '* sin(theta)')
    gate = _numeric_gate(scalar, density, root)
    print(f'gate 1 (symbolic vs einstein.py): {gate:.3e}')
    if not gate < 1e-9:
        raise SystemExit('symbolic curvature disagrees with the numeric evaluator')

    alpha = sp.Symbol('alpha', positive=True)
    f = sp.pi/8*V1*V2*sp.sqrt(V3)*(scalar + alpha*density)
    stationarity = [sp.factor(sp.numer(sp.together(sp.simplify(sp.diff(f, v)))))
                    for v in (V1, V2, V3)]
    vacuum = sp.solve([e.subs(alpha, 0) for e in stationarity], [V2, V3, K], dict=True)
    print('alpha=0 solution:', vacuum)
    if vacuum != [{K: 1, V2: 4*V1, V3: 8*V1}]:
        raise SystemExit('the vacuum near-horizon solution is not the expected one')

    lines = ['"""Generated by experiments/derive_near_horizon.py; do not edit by hand.',
             '',
             'Extremal near-horizon entropy function for D=5 equal-spin EGB, in the',
             'homogeneous ansatz of that script. `scalar` and `density` are R and L_GB,',
             'checked there against einstein.py; `stationarity` are the three numerators',
             'of df/dv_i, whose common root with v1=1 is the extremal near-horizon.',
             '"""',
             '']
    names = {V1: 'v1', V2: 'v2', V3: 'v3', K: 'k', alpha: 'alpha_gb'}
    body = lambda e: sp.pycode(e.subs(names)).replace('math.', 'np.')
    lines += ['import numpy as np', '', '',
              'def scalar(v1, v2, v3, k):',
              '    """Ricci scalar; constant on the near-horizon geometry."""',
              f'    return {body(scalar)}', '', '',
              'def density(v1, v2, v3, k):',
              '    """Gauss-Bonnet density; also constant."""',
              f'    return {body(density)}', '', '',
              'def entropy_function(v1, v2, v3, k, alpha_gb):',
              '    """f = (pi/8) v1 v2 sqrt(v3) (R + alpha_GB L_GB)."""',
              '    return (np.pi/8)*v1*v2*np.sqrt(v3)*(scalar(v1, v2, v3, k)',
              '                                        + alpha_gb*density(v1, v2, v3, k))',
              '', '',
              'def stationarity(v2, v3, k, alpha_gb, v1=1.0):',
              '    """Numerators of df/dv1, df/dv2, df/dv3; zero on the extremal solution."""',
              '    return (']
    for expression in stationarity:
        lines.append(f'        {body(sp.expand(expression/sp.pi))},')
    lines += ['    )', '']
    TARGET.write_text('\n'.join(lines), encoding='utf-8')
    print('wrote', TARGET.relative_to(ROOT))

    # Gate 2 needs the numeric solution, so it runs against the module just
    # written -- the same code every later consumer will use.
    sys.path.insert(0, str(ROOT/'src'))
    from rotating_bh.near_horizon import extremal  # noqa: E402
    sample = []
    for value in (0., .02, .1, .2, .5):
        state = extremal(value)
        sample.append((value, state['v1'], state['v2'], state['v3'], state['k']))
    worst = _field_equation_gate(sample)
    print(f'gate 2 (G+alpha H on the extremal solution): {worst:.3e}')
    if not worst < 1e-10:
        raise SystemExit('the extremal near-horizon does not solve the field equations')

    from rotating_bh.near_horizon import paper_residuals  # noqa: E402
    published = [paper_residuals(extremal(value))
                 for value in (0., .05, .1, .2, .4, .6, .8)]
    stationary = max(r['stationarity'] for r in published)
    spin = max(abs(r['spin_ratio']-1) for r in published)
    print(f'gate 3 (1010.0860v1 eqs. 4.11/4.12 as published): '
          f'{stationary:.3e}, spin ratio off by {spin:.3e}')
    if not (stationary < 1e-12 and spin < 1e-12):
        raise SystemExit('the published near-horizon equations are not reproduced')


if __name__ == '__main__':
    main()
