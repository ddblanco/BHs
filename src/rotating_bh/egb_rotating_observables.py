"""Charges and Wald entropy in r_H=G5=1 units, arXiv:1010.0860v1.

Eqs. (3.6), (3.10): alpha_paper=4*alpha_gb. J denotes EACH spin.
Tail-window spreads and endpoint differences are diagnostics, not rigorous
error bounds. For cutoff solutions the horizon is transported to order two.
"""
import numpy as np
from scipy.optimize import brentq
from .egb_rotating_angular import first_integral


def _horizon(solution):
    cut = getattr(solution, 'cutoff', 0.)
    return (solution.evaluate(cut)-cut*solution.evaluate(cut, 1)
            +cut**2/2*solution.evaluate(cut, 2))


def measure(solution):
    """Measure E,J,T_H,A_H,S with independent tail/angular diagnostics."""
    estimates = []
    for lower, upper, degree in [(.03,.12,4), (.04,.16,4), (.03,.12,3)]:
        z = np.linspace(lower, upper, 80)
        if getattr(solution, 'cutoff', 0.) > lower:
            raise ValueError('cutoff overlaps asymptotic fit window')
        B,F,H,W = solution.evaluate(1-z)
        # U is common to b and f; fit them separately to test that relation.
        tails = [((1-z*z)*v-1)/(z*z) for v in (B,F)] + [W]
        estimates.append([np.polynomial.polynomial.polyfit(z*z,v,degree)[0] for v in tails])
    U, Uf, W = estimates[0]
    B,F,H,_ = _horizon(solution)
    if min(B,F,1+H) <= 0:
        raise ValueError('nonregular extrapolated horizon')
    area = 2*np.pi**2*np.sqrt(1+H)
    current = first_integral(.5, solution.evaluate(.5), solution.evaluate(.5,1), solution.alpha_gb)
    return dict(E=float(-3*np.pi*U/8), J=float(np.pi*W/4),
        T_H=float(np.sqrt(B*F)/(2*np.pi)), A_H=float(area),
        S=float(area/4*(1+4*solution.alpha_gb*(3-H))),
        U=float(U), U_f=float(Uf), W=float(W),
        mass_tail_spread=float(3*np.pi/8*np.ptp(np.array(estimates)[:,0])),
        spin_tail_spread=float(np.pi/4*np.ptp(np.array(estimates)[:,2])),
        tail_fits=[dict(z_min=lo,z_max=hi,degree=degree,U=float(v[0]),U_f=float(v[1]),W=float(v[2]))
                   for (lo,hi,degree),v in zip([(.03,.12,4),(.04,.16,4),(.03,.12,3)],estimates)],
        mass_b_f_difference=float(3*np.pi/8*abs(U-Uf)),
        spin_current_difference=float(abs(np.pi*W/4+np.pi*current/16)))


def ergosurface(solution):
    """Outer stationary-limit root g_tt=-b+h*w²=0, r_H=1."""
    if solution.omega_h == 0:
        return 1.
    cut = getattr(solution, 'cutoff', 0.)
    def gtt(x):
        B,_,H,W = solution.evaluate(x)
        z = 1-x
        return -(1-z*z)*B+(1+z**4*H)*z**6*W**2
    grid = np.linspace(cut, 1-max(cut,1e-6), 1001)
    values = np.array([gtt(x) for x in grid])
    brackets = np.flatnonzero(values[:-1]*values[1:] < 0)
    if not len(brackets):
        raise ValueError('no exterior stationary-limit root bracketed')
    i = brackets[-1]
    root = brentq(gtt, grid[i], grid[i+1], xtol=1e-14)
    return float(1/(1-root))
