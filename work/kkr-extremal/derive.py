"""Reduced-action field equations of the equal-spin EGB system with the
squashing function g = f2 left free (no radial gauge fixed).

The project's solver fixes g = r^2. The extremal construction needs other
gauges (the horizon at a double zero), so the Euler-Lagrange equations are
re-derived here for generic (b, f, g, h, w) from the reduced Lagrangian of
arXiv:1010.0860 (eq. 2.11), which `experiments/derive_egb_rotating.py`
transcribed and validated in the g = r^2 gauge.

Gate (aborts on failure): on generic, non-solution jets of all five
functions, every Euler-Lagrange expression must equal
    E_v = -sqrt(b h / f) * g * (G + alpha H)^{mu nu} d g_{mu nu} / d v
computed with the independent NumPy curvature evaluator `einstein.py` on the
full 5x5 metric with g free. This tests the g-dependence of the reduced
action, which the earlier gate (g = r^2) could not.

Metric (our conventions, alpha the coefficient of L_GB inside 1/16 pi):
  ds^2 = -b dt^2 + dr^2/f + g [dth^2 + s^2 c^2 (dphi1-dphi2)^2]
         + h [s^2 dphi1 + c^2 dphi2 - w dt]^2 ,  s = sin th, c = cos th.

Output: equations.pkl  (sympy expressions of E_b, E_g, E_h, E_w, E_f)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments'))
from derive_egb_rotating import lagrangians, euler_lagrange  # noqa: E402
from rotating_bh.einstein import compile_metric, curvature, gauss_bonnet  # noqa: E402


def metric_general(r, theta, b, f, g, h, w):
    s, c = sp.sin(theta)**2, sp.cos(theta)**2
    m = sp.zeros(5)
    m[0, 0] = -b+h*w**2
    m[1, 1] = 1/f
    m[2, 2] = g
    m[0, 3] = m[3, 0] = -h*w*s
    m[0, 4] = m[4, 0] = -h*w*c
    m[3, 3] = h*s+(g-h)*s*c
    m[4, 4] = h*c+(g-h)*s*c
    m[3, 4] = m[4, 3] = -(g-h)*s*c
    return m


def equations():
    v = lagrangians()
    r, b, f, g, h, w = (v[k] for k in ('r', 'b', 'f', 'g', 'h', 'w'))
    alpha = sp.Symbol('alpha')
    L = v['L_E']+alpha*v['L_GB']
    assert sp.diff(L, sp.diff(f, r)) == 0
    E = {name: euler_lagrange(L, field, r) for name, field in
         (('b', b), ('g', g), ('h', h), ('w', w))}
    E['f'] = sp.diff(L, f)
    return dict(r=r, b=b, f=f, g=g, h=h, w=w, alpha=alpha, E=E)


# generic jets (value, first, second derivative) -- not solutions of anything
PROFILES = [
    dict(R=1.4, a=0.02, th=0.6, b=(1.1, .6, -.3), f=(.8, -.4, .2), g=(2.3, 1.7, .4),
         h=(1.6, .5, -.6), w=(.1, -.2, .3)),
    dict(R=2.2, a=0.30, th=1.1, b=(.7, -.5, .8), f=(1.3, .3, -.2), g=(4.1, -.8, 1.3),
         h=(.9, -.7, .4), w=(-.05, .35, -.15)),
    dict(R=1.0, a=0.05, th=0.9, b=(1.9, .2, .5), f=(.6, -.6, .5), g=(.7, .9, -.5),
         h=(2.1, .9, -.2), w=(.2, .1, .1)),
    dict(R=3.0, a=0.45, th=0.3, b=(.9, -.2, -.4), f=(1.7, .5, .3), g=(9.5, 5.1, 2.2),
         h=(.7, .3, .6), w=(-.3, -.1, .2)),
    dict(R=1.7, a=0.60, th=1.2, b=(.5, .3, .2), f=(.9, .7, -.6), g=(1.9, -.3, .8),
         h=(1.8, .6, -.5), w=(-.12, .25, .05)),
]


def check_against_tensor(eq):
    r, alpha = eq['r'], eq['alpha']
    fields = {k: eq[k] for k in 'bfghw'}
    t, rc, th, p1, p2 = sp.symbols('t rc th p1 p2', real=True)
    R0 = sp.Symbol('R0')
    jet_syms = {k: sp.symbols(f'{k}0 {k}1 {k}2') for k in 'bfghw'}
    jet = {k: s0+s1*(rc-R0)+s2*(rc-R0)**2/2 for k, (s0, s1, s2) in jet_syms.items()}
    metric = metric_general(rc, th, *(jet[k] for k in 'bfghw'))
    params = (R0, *[s for k in 'bfghw' for s in jet_syms[k]])
    coords = (t, rc, th, p1, p2)
    evaluate = compile_metric(metric, coords, params)
    dmetric = {k: compile_metric(sp.Matrix(5, 5, lambda i, j: sp.diff(metric[i, j], jet_syms[k][0])),
                                 coords, params) for k in 'bfghw'}
    # substitution for the action side: highest derivatives first
    worst = 0.
    for p in PROFILES:
        values = [p['R']]+[x for k in 'bfghw' for x in p[k]]
        G, dG, ddG = evaluate(0., p['R'], p['th'], 0., 0., *values)
        c = curvature(G, dG, ddG)
        gb = gauss_bonnet(G, c['inverse'], c['riemann'], c['ricci'], c['scalar'])
        upper = c['inverse'] @ (c['einstein']+p['a']*gb['H']) @ c['inverse']
        b0, f0, g0, h0 = p['b'][0], p['f'][0], p['g'][0], p['h'][0]
        subs = {alpha: p['a'], r: p['R']}
        for k, fn in fields.items():
            subs[sp.diff(fn, r, 2)] = p[k][2]
        for k, fn in fields.items():
            subs[sp.diff(fn, r)] = p[k][1]
        for k, fn in fields.items():
            subs[fn] = p[k][0]
        for k in 'bfghw':
            action = float(eq['E'][k].subs(subs))
            dg, _, _ = dmetric[k](0., p['R'], p['th'], 0., 0., *values)
            tensor = -np.sqrt(b0*h0/f0)*g0*float(np.einsum('ij,ij->', upper, dg))
            rel = abs(action-tensor)/max(abs(action), abs(tensor), 1e-12)
            worst = max(worst, rel)
            assert rel < 1e-9, (k, p, action, tensor)
    print(f'reduced action == tensor route for E_b,E_f,E_g,E_h,E_w with g free, '
          f'{len(PROFILES)} generic jets, worst relative difference {worst:.2e}')
    return worst


def main():
    eq = equations()
    worst = check_against_tensor(eq)
    with open(HERE/'equations.pkl', 'wb') as handle:
        pickle.dump(dict(eq=eq, worst=worst), handle)
    print('wrote equations.pkl')


if __name__ == '__main__':
    main()
