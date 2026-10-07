"""Indicial exponents of the extremal residual operator at the horizon x = 0,
linearised about extremal Myers-Perry (alpha = 0) in the gauge of
extremal_equations.py.

Perturbation delta(B,F,H,W) = c x^s. Each residual R_i changes by
sum_j x^(s + q_ij) a_ij(s) c_j + higher powers; for each equation the leading
power is the smallest q_ij, and the indicial matrix keeps those entries.
Its determinant's roots s are the exponents of the regular-singular point;
a doubled root means a log x solution, which polynomial collocation cannot
reject sharply.
"""
import pickle
from pathlib import Path

import sympy as sp

from extremal_equations import ALPHA, J, X

HERE = Path(__file__).resolve().parent
S = sp.Symbol('s')


def main():
    with open(HERE/'extremal_residuals.pkl', 'rb') as handle:
        res = pickle.load(handle)['residuals']
    z = 1-X
    mp = {'B': 1/(1+z**4), 'F': sp.Integer(1), 'H': sp.Integer(1),
          'W': sp.sqrt(2)/(1+z**4)}
    base = {}
    for n in 'BFHW':
        base[J[n]] = mp[n]
        base[J[n+'x']] = sp.diff(mp[n], X)
        if n+'xx' in J:
            base[J[n+'xx']] = sp.diff(mp[n], X, 2)
    t = sp.Symbol('t', positive=True)       # x = t; perturbation x^s
    rows = []
    for name in ('b', 'g', 'h', 'w'):
        R = res[name]
        row = []
        for n in 'BFHW':
            # W enters through w' only, so it balances one power higher
            pert = X**(S+1) if n == 'W' else X**S
            d = sp.diff(R, J[n]).subs(base)*pert
            d += sp.diff(R, J[n+'x']).subs(base)*sp.diff(pert, X)
            if n+'xx' in J:
                d += sp.diff(R, J[n+'xx']).subs(base)*sp.diff(pert, X, 2)
            d = sp.powsimp(sp.expand(d.subs(ALPHA, 0)*X**(-S)), force=True)
            d = sp.simplify(d)
            row.append(d)
        rows.append(row)
    # leading power of x in each entry: rational function num/den in x
    def leading(e):
        if e == 0:
            return sp.oo, 0
        num, den = sp.fraction(sp.together(e))
        pn, pd = sp.Poly(sp.expand(num), X), sp.Poly(sp.expand(den), X)
        ln, ld = min(m[0] for m in pn.monoms()), min(m[0] for m in pd.monoms())
        return ln-ld, sp.simplify(pn.coeff_monomial(X**ln)/pd.coeff_monomial(X**ld))
    M = sp.zeros(4, 4)
    for i, row in enumerate(rows):
        lead = [leading(e) for e in row]
        q = min(p for p, _ in lead)
        for j, (p, c) in enumerate(lead):
            M[i, j] = c if p == q else 0
        print(f'E_{"bghw"[i]}: leading power x^(s{q:+d}); per field', [p for p, _ in lead])
    print('indicial matrix:')
    sp.pprint(sp.simplify(M))
    det = sp.factor(sp.simplify(M.det()))
    print('determinant:', det)
    print('roots:', sp.solve(det, S))


if __name__ == '__main__':
    main()
