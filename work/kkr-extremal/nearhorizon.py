"""Near-horizon geometry of the extremal equal-spin EGB black hole and the
scaling exponents of its U(2)-symmetric perturbations.

Background (AdS2 fibred over a squashed S^3), in the variables of derive.py:
    b = v1 rho^2,  f = rho^2/v1,  g = G0,  h = H0,  w = W0 + w1 rho .
The Euler-Lagrange equations of derive.py reduce to three algebraic relations
in (v1, G0, H0, w1, alpha); the length scale is fixed by G0 = 1, so the
branch is a one-parameter family in alpha/G0.

Perturbations, in the gauge f = rho^2/v1 (kept exact):
    b -> b (1 + eps beta rho^gam),  g -> G0 (1 + eps eta rho^gam),
    h -> H0 (1 + eps chi rho^gam),  w' -> w1 (1 + eps om rho^gam).
At first order every equation is a single power of rho times a linear form in
(beta, eta, chi, om); a mode exists where the 5x4 coefficient matrix loses
rank. These exponents are the non-integer powers a horizon expansion of the
full extremal solution may need.

Checks (asserted in main):
  1. alpha = 0 gives the extremal Myers-Perry throat (v1 = G0/4, H0 = 2 G0,
     w1 = -1/2, S = 2 pi J);
  2. the background satisfies the closed forms of arXiv:2303.12471
     (their eqs. at7, at8, at9) for v1, v2 = G0, v3 = H0/G0, k = -w1, J, S;
  3. J from the conserved w-momentum is independent of rho.

Run: python nearhorizon.py   ->  nearhorizon.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'experiments'))
from derive_egb_rotating import lagrangians  # noqa: E402

RHO = sp.Symbol('rho', positive=True)
V1, G0S, H0S, W1, W0 = sp.symbols('v1 G0 H0 w1 W0', real=True)
EPS, GAM = sp.symbols('epsilon gamma', real=True)
BETA, ETA, CHI, OM = sp.symbols('beta eta chi om', real=True)


def load():
    with open(HERE/'equations.pkl', 'rb') as handle:
        return pickle.load(handle)['eq']


def substitute(eq, expressions, target):
    """Replace b,f,g,h,w (and derivatives) in `target` by functions of rho."""
    r = eq['r']
    subs = {}
    for k in 'bfghw':
        subs[sp.diff(eq[k], r, 2)] = sp.diff(expressions[k], RHO, 2)
    for k in 'bfghw':
        subs[sp.diff(eq[k], r)] = sp.diff(expressions[k], RHO)
    for k in 'bfghw':
        subs[eq[k]] = expressions[k]
    return target.subs(subs, simultaneous=True).subs(r, RHO)


def background():
    return dict(b=V1*RHO**2, f=RHO**2/V1, g=G0S, h=H0S, w=W0+W1*RHO)


def perturbed():
    return dict(b=V1*RHO**2*(1+EPS*BETA*RHO**GAM), f=RHO**2/V1,
                g=G0S*(1+EPS*ETA*RHO**GAM), h=H0S*(1+EPS*CHI*RHO**GAM),
                w=W0+W1*RHO+EPS*W1*OM*RHO**(GAM+1)/(GAM+1))


class NearHorizon:
    def __init__(self):
        eq = self.eq = load()
        alpha = eq['alpha']
        bg = background()
        E = {n: substitute(eq, bg, e) for n, e in eq['E'].items()}
        # each background equation is one power of rho: evaluate at rho = 1
        self.background_eqs = [sp.simplify(E[n].subs(RHO, 1)) for n in ('b', 'g', 'h')]
        assert sp.simplify(E['w']) == 0
        assert sp.simplify(E['f']+V1**2*E['b']) == 0       # E_f = -v1^2 E_b here
        self.residual = sp.lambdify((V1, G0S, H0S, W1, alpha), self.background_eqs)
        # conserved momentum of w and the Wald entropy
        v = lagrangians()
        L = v['L_E']+alpha*v['L_GB']
        p_w = sp.diff(L, sp.diff(v['w'], v['r']))
        relabel = {sp.diff(v[k], v['r']): sp.diff(eq[k], eq['r']) for k in 'bfghw'}
        p_w = p_w.subs(relabel, simultaneous=True)
        p_w = p_w.subs({v[k]: eq[k] for k in 'bfghw'}, simultaneous=True).subs(v['r'], eq['r'])
        p_w = sp.simplify(substitute(eq, bg, p_w))
        assert sp.diff(p_w, RHO) == 0, p_w
        self.J = sp.lambdify((V1, G0S, H0S, W1, alpha), -sp.pi/16*p_w)
        self.S = lambda G0, H0, alpha: np.pi**2/2*G0*np.sqrt(H0)*(1+2*alpha*(8/G0-2*H0/G0**2))
        # first-order perturbation matrix
        pt = perturbed()
        rows = []
        for n in ('b', 'g', 'h', 'w', 'f'):
            e1 = sp.diff(substitute(eq, pt, eq['E'][n]), EPS).subs(EPS, 0)
            rows.append(e1)
        self.rows = rows
        self.first_order = sp.lambdify((V1, G0S, H0S, W1, alpha, GAM, RHO, BETA, ETA, CHI, OM), rows)

    def solve(self, alpha, guess):
        fun = lambda z: self.residual(z[0], 1., z[1], z[2], alpha)
        sol = root(fun, guess, method='hybr', tol=1e-15)
        res = np.max(np.abs(fun(sol.x)))
        assert res < 1e-12, (alpha, res, sol.message)
        v1, H0, w1 = sol.x
        return dict(alpha=alpha, v1=v1, G0=1., H0=H0, w1=w1,
                    J=float(self.J(v1, 1., H0, w1, alpha)),
                    S=float(self.S(1., H0, alpha)), residual=float(res))

    def branch(self, alphas):
        guess, out = np.array([.25, 2., -.5]), []
        for a in alphas:
            s = self.solve(a, guess)
            guess = np.array([s['v1'], s['H0'], s['w1']])
            out.append(s)
        return out

    def matrix(self, s, gam, rho=1.):
        """5x4 coefficient matrix of (beta, eta, chi, om) at exponent gam."""
        args = (s['v1'], s['G0'], s['H0'], s['w1'], s['alpha'], gam, rho)
        cols = []
        for unit in np.eye(4):
            cols.append(np.array(self.first_order(*args, *unit), dtype=float))
        return np.array(cols).T

    def exponents(self, s, window=(-6., 6.), samples=12001):
        """Exponents where the matrix loses rank: minima of the smallest
        singular value (rows normalised), refined, kept when it vanishes."""
        def smin(gam):
            m = self.matrix(s, gam)
            m = m/np.maximum(np.abs(m).max(axis=1, keepdims=True), 1e-300)
            return np.linalg.svd(m, compute_uv=False)[-1]
        grid = np.linspace(*window, samples)
        values = np.array([smin(x) for x in grid])
        found = []
        for i in range(1, len(grid)-1):
            if values[i] <= values[i-1] and values[i] <= values[i+1]:
                a, b = grid[i-1], grid[i+1]
                for _ in range(80):             # golden-section refinement
                    m1, m2 = a+(b-a)*.382, a+(b-a)*.618
                    if smin(m1) < smin(m2):
                        b = m2
                    else:
                        a = m1
                x = (a+b)/2
                if smin(x) < 1e-7:
                    found.append(float(x))
        return found


def main():
    nh = NearHorizon()
    # check 1: Myers-Perry throat
    mp = nh.solve(0., [.3, 1.8, -.4])
    assert abs(mp['v1']-.25) < 1e-12 and abs(mp['H0']-2) < 1e-12 and abs(mp['w1']+.5) < 1e-12, mp
    assert abs(mp['S']-2*np.pi*mp['J']) < 1e-12*mp['S'], mp
    print('alpha=0: extremal Myers-Perry throat, S = 2 pi J: OK', mp)
    # check 2: closed forms of arXiv:2303.12471 along their v3 parametrisation
    worst = 0.
    for v3 in np.linspace(1.98, .05, 60):
        disc = 5*v3**4-34*v3**3+73*v3**2-56*v3+16
        alpha = 1.
        v2 = 4*alpha/(v3-2)*(2*v3**2-7*v3+4-np.sqrt(disc))
        v1 = ((v2-4*alpha*(3*v3-4))*(3*v2+4*alpha*(4-v3))
              / (2*(4-v3)*(3*v2+4*alpha*(8-6*v3))))
        # their at9 'J' is J1+J2 (checked: at alpha=0 it is twice the per-plane J
        # of extremal Myers-Perry, for which S = 2 pi J); ours is per plane
        J = np.pi/4*v2*v3*np.sqrt((4-v3)*(v2+4*alpha*(4-3*v3)))/2
        S = np.pi**2/2*np.sqrt(v2*v3)*(v2+4*alpha*(4-v3))
        k = (4-v3)*np.sqrt(v2*v3)*np.pi*v1/(2*(2*J))
        lam = 1/v2           # rescale to G0 = 1: lengths^2 by lam
        s = nh.solve(alpha*lam, [v1*lam, v3, -k])
        dev = max(abs(s['v1']-v1*lam), abs(s['H0']-v3), abs(s['w1']+k),
                  abs(s['J']-J*lam**1.5)/(J*lam**1.5), abs(s['S']-S*lam**1.5)/(S*lam**1.5))
        worst = max(worst, dev)
    assert worst < 1e-9, worst
    print(f'closed forms of arXiv:2303.12471 (at7-at9) reproduced on 60 points, worst {worst:.1e}: OK')
    # perturbation exponents along the branch
    # dense at small coupling, then out to large alpha/g_H (the branch
    # approaches x -> 1 there); stops where the throat solve fails
    alphas = np.concatenate([[0.], np.geomspace(1e-3, .4, 30), np.geomspace(.45, 200., 40)])
    report = []
    branch = []
    guess = np.array([.25, 2., -.5])
    for a in alphas:
        try:
            s = nh.solve(a, guess)
        except AssertionError:
            print(f'near-horizon solve failed at alpha/G0 = {a:.3f}; scan stops')
            break
        guess = np.array([s['v1'], s['H0'], s['w1']])
        branch.append(s)
    for s in branch:
        y = s['alpha']/s['J']**(2/3)
        ex = nh.exponents(s)
        report.append(dict(s, y=float(y), exponents=ex))
        print(f"alpha/G0={s['alpha']:.4f}  y={y:.4f}  exponents {np.round(ex, 6)}")
    (HERE/'nearhorizon.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
