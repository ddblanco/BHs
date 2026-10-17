"""Exact weak-coupling coefficients d_n of m(s) = 3/2 + sum d_n s^n from code/pert_order5.json
(d_n = M_n 2^(1-n)/pi), their values, and the comparison of their signs/growth with the
nearest branch point s* of the algebraic extremal throat (strong/gH_branch.py)."""
import json, sympy as s, mpmath as mp
from pathlib import Path
HERE = Path(__file__).resolve().parent
Z2, Z3, Z5 = s.symbols('Z2 Z3 Z5')
J = json.loads((HERE.parent/'code/pert_order5.json').read_text())
d = {1: s.Integer(1)}
for n, M in J['masses'].items():
    n = int(n)
    d[n] = s.expand(s.sympify(M, locals=dict(Z2=Z2, Z3=Z3, Z5=Z5))*s.Integer(2)**(1-n)/s.pi)
zv = {Z2: s.pi**2/6, Z3: s.zeta(3), Z5: s.zeta(5)}
out = {}
for n in sorted(d):
    val = s.N(d[n].subs(zv), 30)
    print(f'd_{n} = {s.expand(d[n])}\n     = {val}')
    out[n] = dict(exact=str(s.expand(d[n])), value=str(val))
# published / earlier checks
assert d[2] == s.Rational(-236, 105)
assert s.simplify(d[3].subs(zv) - (s.Rational(60851752, 33075)-1536*s.zeta(3))) == 0
assert s.simplify(d[4].subs(zv) - (-s.Rational(592541765354296, 1489863375)+s.Rational(227328, 5)*s.zeta(3)+s.Rational(88064, 25)*s.pi**4)) == 0
print('PASS d2, d3, d4 equal the values in extremal-mass-report.tex')
# compare with the throat branch point
mp.mp.dps = 30
sst = mp.mpc('0.0059231771', '0.0548188529')
print('|s*| =', mp.nstr(abs(sst), 10), ' arg s* (deg) =', mp.nstr(mp.degrees(mp.arg(sst)), 8))
for n in sorted(d):
    v = mp.mpf(str(s.N(d[n].subs(zv), 30)))
    print(f'n={n}: sign d_n = {"+" if v > 0 else "-"}, sign Re(s*^-n) = {"+" if mp.re(sst**-n) > 0 else "-"}, |d_n| |s*|^n = {mp.nstr(abs(v)*abs(sst)**n, 5)}, ratio d_n/d_(n-1) = {mp.nstr(v/mp.mpf(str(s.N(d[n-1].subs(zv), 30))), 6) if n > 1 else "-"}')
json.dump(out, open(HERE/'coefficients.json', 'w'), indent=1)
