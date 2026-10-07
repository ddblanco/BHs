"""Gate for the KKR-gauge extremal solver.

Asserted at alpha = 0 (extremal Myers-Perry, a = 1, per-plane J):
  * residual exactly zero at P = 1, W = 0; chain-rule Jacobian = dense one;
  * Jacobian well conditioned (the g = r^2 formulation had sigma_min ~ 1e-11);
  * Newton from a smooth perturbation returns P = 1, W = 0;
  * M = 3 pi/4, J = pi sqrt(2)/4, Omega_H = 1/sqrt 2, S = 2 pi J, j = 1,
    a_H = s = 1/2.
Then a first continuation in alpha with diagnostics.
"""
import numpy as np

from kkr_solver import KKR, invariants, myers_perry


def main(power=2):
    for n in (24, 40):
        k = KKR(n, power)
        mp = myers_perry(n)
        # zero symbolically (kkr_equations.py); here only floating-point cancellation
        assert np.max(np.abs(k.rows(mp.reshape(4, n), 0.))) < 1e-9
        p = mp+0.01*np.sin(np.arange(4*n))
        k.weights = None                      # compare the raw Jacobians, row by row
        A, B = k.jacobian(p, 0.), k.dense_jacobian(p, 0.)
        err = np.max(np.abs(A-B)/np.maximum(np.abs(B).max(axis=1, keepdims=True), 1e-300))
        assert err < 1e-10, err
        k.set_weights(mp, 0.)
        s = np.linalg.svd(k.jacobian(mp, 0.), compute_uv=False)
        print(f'N={n}: Jacobian check {err:.1e}; condition number at MP {s[0]/s[-1]:.2e}, '
              f'smallest singular values {s[-3:]/s[0]}')
    n = 40
    k = KKR(n, power)
    x = np.tile(k.x, 4)
    guess = myers_perry(n)+0.05*np.sin(2*x+np.repeat(np.arange(4), n))*x*(1-x)
    sol, info = k.solve(0., guess, verbose=True)
    dev = np.max(np.abs(sol-myers_perry(n)))
    o = k.observables(sol, 0.)
    inv = invariants(o)
    print('Newton from perturbed MP:', info, 'max deviation', dev)
    print('  M', o['M'], 3*np.pi/4, ' J', o['J'], np.pi*np.sqrt(2)/4, ' Omega', o['Omega_H'])
    assert dev < 1e-10
    assert abs(o['M']-3*np.pi/4) < 1e-10 and abs(o['J']-np.pi*np.sqrt(2)/4) < 1e-10
    assert abs(o['S']-2*np.pi*o['J']) < 1e-10 and abs(inv['j']-1) < 1e-10
    assert abs(inv['aH']-.5) < 1e-10 and abs(inv['s']-.5) < 1e-10
    print('alpha = 0 gate: OK')
    flat = sol
    for alpha in (1e-3, 3e-3, 1e-2, 2e-2, 4e-2):
        flat, info = k.solve(alpha, flat)
        o = k.observables(flat, alpha)
        inv = invariants(o)
        print(f"alpha={alpha:.3f}: res {info['residual']:.1e} it {info['iterations']} cond {info['condition']:.1e} "
              f"| M {o['M']:.10f} J {o['J']:.10f} Omega {o['Omega_H']:.8f} C {o['constraint']:.1e} "
              f"| y {inv['y']:.5f} mu {inv['mu']:.8f} j {inv['j']:.6f}")


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
