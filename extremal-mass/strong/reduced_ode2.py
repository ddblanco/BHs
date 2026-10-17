import sys, pickle, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'code'))
import sympy as s
from reduced_system import r, alpha, f, h, p, q, t, u, v, C2
E = pickle.load(open(HERE/'reduced_eqs.pkl', 'rb'))
eb, ef, eh = E['eb'], E['ef'], E['eh']
U = s.factor(-ef.subs(u, 0)/s.diff(ef, u))
print('U =', U)
Ur = s.diff(U, r)+s.diff(U, f)*p+s.diff(U, h)*q+s.diff(U, q)*t
eh2 = s.numer(s.together(eh.subs({u: U, v: Ur}, simultaneous=True)))
eh2 = s.factor(eh2)
print('eh2 factors:', [ (s.degree(fa, p), s.degree(fa, t), s.count_ops(fa)) for fa, k in s.factor_list(eh2)[1]])
for fa, k in s.factor_list(eh2)[1]:
    if fa.has(p) or fa.has(t):
        main = fa
P = s.Poly(main, p, t); print('eh monomials in (p,t):', P.monoms())
Pb = s.Poly(eb, p, t); print('eb monomials in (p,t):', Pb.monoms())
pickle.dump(dict(eb=eb, eh=main, U=U), open(HERE/'reduced_ode.pkl', 'wb'))
