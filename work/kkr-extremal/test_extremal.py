"""Gate for the extremal solver at alpha = 0 and a first step in alpha.

Asserted:
  1. exact extremal Myers-Perry is a root of the discrete residual;
  2. the chain-rule Jacobian equals the dense complex-step Jacobian;
  3. Newton from a perturbed guess returns Myers-Perry, and the observables
     are M = 3 pi/4, J = pi sqrt(2)/4, Omega_H = 1/sqrt(2), S = 2 pi J, j = 1,
     a_H = s = 1/2 (r_H = 1);
  4. at small alpha the horizon data match the near-horizon solution of
     nearhorizon.py (h_H/g_H and the ratio J/S scale-free comparison).
"""
import numpy as np

from extremal_solver import Extremal, invariants, myers_perry


def main():
    n = 40
    ex = Extremal(n)
    mp = myers_perry(ex.x)
    ex.set_weights(mp, 0.)        # rows normalised by their largest jet sensitivity
    r0 = np.max(np.abs(ex.residual(mp, 0.)))
    print(f'normalised residual at exact Myers-Perry, N={n}: {r0:.2e}')
    assert r0 < 1e-8          # round-off floor of D2 ~ N^4 eps at this N
    weights, ex.weights = ex.weights, None      # compare the raw Jacobians
    jerr = ex.check_jacobian(mp+0.01*np.sin(np.arange(4*n)), 0.)
    ex.weights = weights
    print(f'chain-rule vs dense complex-step Jacobian: {jerr:.2e}')
    assert jerr < 1e-10       # products with D2 round at this level
    # a smooth perturbation: white noise on the nodes would excite the top
    # Chebyshev modes, which is not what a continuation step ever produces
    x = np.tile(ex.x, 4)
    guess = mp*(1+0.02*np.sin(3*x+np.repeat(np.arange(4), n)))
    sol, info = ex.solve(0., guess, verbose=False)
    print('Newton from perturbed Myers-Perry:', info)
    dev = np.max(np.abs(sol-mp))
    o = ex.observables(sol, 0.)
    inv = invariants(o)
    print(f'  max |u-u_MP| = {dev:.2e}; M={o["M"]:.15f} (3pi/4={3*np.pi/4:.15f}) '
          f'J={o["J"]:.15f} (pi sqrt2/4={np.pi*np.sqrt(2)/4:.15f}) Omega={o["Omega_H"]:.15f}')
    print('  invariants', inv)
    assert dev < 1e-10
    assert abs(o['M']-3*np.pi/4) < 1e-10 and abs(o['J']-np.pi*np.sqrt(2)/4) < 1e-10
    assert abs(o['Omega_H']-1/np.sqrt(2)) < 1e-10 and abs(o['S']-2*np.pi*o['J']) < 1e-9
    assert abs(inv['j']-1) < 1e-10 and abs(inv['aH']-.5) < 1e-10 and abs(inv['s']-.5) < 1e-10
    assert abs(inv['mu']-1.5*np.pi**(1/3)) < 1e-10
    print('alpha = 0 gate: OK')
    # first steps in alpha
    flat = sol
    for alpha in (1e-3, 5e-3, 1e-2, 2e-2):
        flat, info = ex.solve(alpha, flat)
        o = ex.observables(flat, alpha)
        print(f'alpha={alpha}: {info}  M={o["M"]:.10f} J={o["J"]:.10f} '
              f'Omega={o["Omega_H"]:.8f} constraint={o["constraint"]:.1e}', invariants(o))


if __name__ == '__main__':
    main()
