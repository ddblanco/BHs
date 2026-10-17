import sys, pickle, time
from pathlib import Path
HERE = Path('.').resolve()
sys.path.insert(0, str(HERE.parent/'code'))
import sympy as s
from reduced_system import r, alpha, f, h, p, q, t, C2
E = pickle.load(open('reduced_ode.pkl', 'rb'))
x, rH = s.symbols('x r_H', positive=True)
NO = 4
F = s.symbols('F0:%d' % NO); H = s.symbols('H0:%d' % (NO+1))
fx = x**2*sum(F[k]*x**k for k in range(NO)); hx = sum(H[k]*x**k for k in range(NO+1))
sub = {r: rH+x, f: fx, h: hx, p: s.diff(fx, x), q: s.diff(hx, x), t: s.diff(hx, x, 2), C2: 32}
res = {}
for nm in ('eb', 'eh'):
    t0 = time.time()
    e = s.Poly(E[nm].xreplace(sub), x)
    co = e.all_coeffs()[::-1]
    res[nm] = [s.factor(co[k]) for k in range(4)]
    print(nm, time.time()-t0, flush=True)
    for k in range(4):
        print(f'  {nm}[{k}] free:', sorted(map(str, res[nm][k].free_symbols)), flush=True)
pickle.dump(res, open('series_low.pkl', 'wb'))
print('eb0 =', res['eb'][0])
print('eb1 =', res['eb'][1])
print('eh2 =', res['eh'][2])
