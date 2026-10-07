import numpy as np, warnings
warnings.simplefilter('ignore')
from extremal_rq import ExtremalRQ, myers_perry_q, invariants
n = 48
e = ExtremalRQ(n)
x = np.tile(e.x, 4)
for amp in (0., 1e-6, 1e-4, 1e-3):
    guess = amp*np.sin(2*x+np.repeat(np.arange(4), n))
    sol, info = e.solve(0., guess, verbose=(amp == 1e-3))
    print(amp, info, 'max|Q|', np.max(np.abs(sol)))

print('--- the stray root: diagnostics and resolution dependence')
guess = 1e-3*np.sin(2*x+np.repeat(np.arange(4), n))
sol, info = e.solve(0., guess)
o = e.observables(sol, 0.)
print('N=48', {k: o[k] for k in ('M', 'J', 'Omega_H', 'g_H', 'h_H', 'constraint', 'J_spread')})
q = sol.reshape(4, n)
for name, c in zip(('Q1', 'Q2', 'Q3', 'Q4'), e.coefficients(sol)):
    print(f'   {name} Chebyshev |c| at k = 0, 8, 16, 32, 47:', np.abs(c[[0, 8, 16, 32, 47]]))
for n2 in (64, 96):
    e2 = ExtremalRQ(n2)
    s2, i2 = e2.solve(0., e.interpolate(sol, n2))
    o2 = e2.observables(s2, 0.)
    print(f'N={n2}', i2, {k: o2[k] for k in ('M', 'J', 'constraint')}, 'max|Q|', np.max(np.abs(s2)))
