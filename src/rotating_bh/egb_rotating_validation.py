"""Off-collocation checks of represented rotating EGB solutions, independent
of the equations used to obtain them -- mirrors static_egb_validation.py.

The conversion from the represented compact amplitude jets (B,F,H,W and
their x-derivatives up to 2nd order) to physical (b,f,h,w and r-derivatives
up to 2nd order) needs the SECOND derivative of F and W too (Fxx, on top of
the already-tracked Bxx,Hxx,Wxx), even though neither the reduced action
nor the solved ODE system ever uses f'' or w'' as independent unknowns:
the metric components themselves involve 1/f and h*w^2-type products, whose
own second r-derivative (needed by the curvature tensor) picks up f'' and
w'' through the quotient/product rule. Omitting them here (as an earlier,
uncommitted attempt did) gives a spuriously large tensor residual near the
horizon even for the exact Myers-Perry solution -- caught by cross-checking
against the closed form before trusting this module, not assumed correct.
"""
from functools import lru_cache

import numpy as np
import sympy as sp

from .einstein import compile_metric, curvature, gauss_bonnet, metric_from_ansatz


@lru_cache(maxsize=1)
def _compact_to_physical():
    """Build once: (x0,B,P,Bxx,F,Fx,Fxx,H,Q,Hxx,W,V,Wxx) -> (r,b,bp,bpp,
    f,fp,fpp,h,hp,hpp,w,wp,wpp), via the same b=(1-z^2)B, f=(1-z^2)F,
    h=(1+z^4 H)/z^2, w=z^4 W, z=1-x ansatz as egb_rotating_bvp.py.
    """
    zz_x, Bs, Fs, Hs, Ws = sp.symbols('zz_x Bs Fs Hs Ws')
    Ps, FXs, Qs, Vs = sp.symbols('Ps FXs Qs Vs')
    Bxxs, Fxxs, Hxxs, Wxxs = sp.symbols('Bxxs Fxxs Hxxs Wxxs')
    x0, xi = sp.symbols('x0 xi')
    B_of_x = Bs+Ps*(xi-x0)+Bxxs*(xi-x0)**2/2
    F_of_x = Fs+FXs*(xi-x0)+Fxxs*(xi-x0)**2/2
    H_of_x = Hs+Qs*(xi-x0)+Hxxs*(xi-x0)**2/2
    W_of_x = Ws+Vs*(xi-x0)+Wxxs*(xi-x0)**2/2
    zz_of_x = 1-xi
    A_of_x = 1-zz_of_x**2
    b_of_x, f_of_x = A_of_x*B_of_x, A_of_x*F_of_x
    h_of_x, w_of_x = (1+zz_of_x**4*H_of_x)/zz_of_x**2, zz_of_x**4*W_of_x
    r_of_x = 1/zz_of_x
    dr = lambda e: sp.diff(e, xi)/sp.diff(r_of_x, xi)
    exprs = [r_of_x.subs(xi, x0),
             b_of_x.subs(xi, x0), dr(b_of_x).subs(xi, x0), dr(dr(b_of_x)).subs(xi, x0),
             f_of_x.subs(xi, x0), dr(f_of_x).subs(xi, x0), dr(dr(f_of_x)).subs(xi, x0),
             h_of_x.subs(xi, x0), dr(h_of_x).subs(xi, x0), dr(dr(h_of_x)).subs(xi, x0),
             w_of_x.subs(xi, x0), dr(w_of_x).subs(xi, x0), dr(dr(w_of_x)).subs(xi, x0)]
    return sp.lambdify((x0, Bs, Ps, Bxxs, Fs, FXs, Fxxs, Hs, Qs, Hxxs, Ws, Vs, Wxxs), exprs, 'numpy')


def physical_jets(solution, x):
    """Return r,b,bp,bpp,f,fp,fpp,h,hp,hpp,w,wp,wpp at compact coordinate x."""
    x = np.asarray(x, dtype=float)
    B, F, H, W = solution.evaluate(x)
    P, Fx, Q, V = solution.evaluate(x, 1)
    Bxx, Fxx, Hxx, Wxx = solution.evaluate(x, 2)
    convert = _compact_to_physical()
    return np.asarray(convert(x, B, P, Bxx, F, Fx, Fxx, H, Q, Hxx, W, V, Wxx))


@lru_cache(maxsize=1)
def _metric_jets():
    t, r, theta, phi1, phi2 = sp.symbols('t r theta phi1 phi2', real=True)
    R = sp.symbols('R', positive=True)
    b0, b1, b2, f0, f1, f2, h0, h1, h2, w0, w1, w2 = sp.symbols(
        'b0 b1 b2 f0 f1 f2 h0 h1 h2 w0 w1 w2')
    jet = lambda v0, v1, v2: v0+v1*(r-R)+v2*(r-R)**2/2
    metric = metric_from_ansatz(r, theta, jet(b0, b1, b2), jet(f0, f1, f2),
                                 jet(h0, h1, h2), jet(w0, w1, w2))
    return compile_metric(metric, (t, r, theta, phi1, phi2),
                           (R, b0, b1, b2, f0, f1, f2, h0, h1, h2, w0, w1, w2))


def tensor_residual(solution, x, alpha_gb, theta=0.6):
    """Full G+alpha_GB H (all diagonal + off-diagonal components) on the
    represented solution at compact coordinate x, via the independent
    tensor evaluator -- none of E_b,E_g,E_h,E_w,E_f enter this check.
    """
    jets = physical_jets(solution, np.array([x]))[:, 0]
    r_, b, bp, bpp, f, fp, fpp, h, hp, hpp, w, wp, wpp = jets
    arrays = _metric_jets()(0, r_, theta, 0, 0, r_, b, bp, bpp, f, fp, fpp, h, hp, hpp, w, wp, wpp)
    result = curvature(*arrays)
    gb = gauss_bonnet(arrays[0], result['inverse'], result['riemann'], result['ricci'], result['scalar'])
    field = result['einstein']+alpha_gb*gb['H']
    scales = np.array([1, 1, r_, r_, r_])
    return float(np.max(np.abs(field/np.outer(scales, scales))))


def check_points(solution, count=51):
    """A fixed interior sample away from both the solver's nodes and x=0,1."""
    lower, upper = 0.02, 0.98
    step = (upper-lower)/count
    points = lower+(np.arange(count)+.5)*step
    for i in range(count):
        while np.min(np.abs(points[i]-solution.nodes)) < 1e-9:
            points[i] += step/8
    return points


def diagnose(solution, alpha_gb, count=51):
    """Sampled tensor residual over the domain; no acceptance threshold
    fixed here yet (Hito 4B is a first pass -- see docs/egb-rotating.md for
    what full acceptance, matching Hitos 1-3's pattern, still needs)."""
    points = check_points(solution, count)
    residuals = [tensor_residual(solution, x, alpha_gb) for x in points]
    return dict(max_tensor_residual=float(np.max(residuals)),
                mean_tensor_residual=float(np.mean(residuals)), count=count)
