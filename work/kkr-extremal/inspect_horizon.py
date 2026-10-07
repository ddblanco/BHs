"""Which jets enter each extremal residual at the horizon node x = 0, and at
what order in x do the second derivatives enter (structure of the
regular-singular point)."""
import pickle
import sympy as sp
from extremal_equations import X, J, HERE

with open(HERE/'extremal_residuals.pkl', 'rb') as handle:
    data = pickle.load(handle)
for name, e in data['residuals'].items():
    poly = sp.Poly(e, X)
    at0 = e.subs(X, 0)
    jets = sorted(str(s) for s in at0.free_symbols)
    print(f'E_{name}: at x=0 depends on {jets}')
    for second in ('Bxx', 'Hxx', 'Wxx', 'Fx', 'Bx', 'Hx', 'Wx'):
        s = J[second]
        coeff = sp.expand(sp.diff(e, s))
        if coeff == 0:
            continue
        low = min(m[0] for m in sp.Poly(coeff, X).monoms())
        print(f'     d/d{second}: lowest power x^{low}')
