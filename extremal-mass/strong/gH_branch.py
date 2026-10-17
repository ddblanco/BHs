import sympy as s, numpy as np
al, g, q = s.symbols('alpha g q')
Pt = 16*al**2*q**2*(4-q)-8*al*g*(2*q**2-7*q+4)+g**2*(q-2)
Jt = g**2*q**2*(4-q)*(g+4*al*(4-3*q))-8
D = s.expand(s.Matrix([[s.diff(Pt, g), s.diff(Pt, q)], [s.diff(Jt, g), s.diff(Jt, q)]]).det())
r1 = s.resultant(Pt, Jt, g); r2 = s.resultant(Pt, D, g)
R = s.factor(s.resultant(r1, r2, q))
print('factors (degree in alpha):', [(s.degree(f_, al), k) for f_, k in s.factor_list(R)[1]])
cands = []
for f_, k in s.factor_list(R)[1]:
    if not f_.has(al): continue
    for rt in np.roots([float(c) for c in s.Poly(f_, al).all_coeffs()]):
        cands.append(2*rt)
cands = sorted(set(np.round(cands, 10)), key=abs)
print('candidate s (nearest 12):', [complex(c) for c in cands[:12]])
