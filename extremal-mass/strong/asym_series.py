"""Large-r expansion of the reduced fixed-J system (C2 = 32):
f = 1 + sum_k a_k z^k, h = r^2 (1 + sum_k b_k z^k), z = 1/r.  Free data appear where
the order-by-order linear system is degenerate."""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve().parent/'code'))
import sympy as s
from reduced_system import r, alpha, f, h, p, q, t, C2
E = pickle.load(open('reduced_ode.pkl', 'rb'))
z = s.Symbol('z', positive=True)
NK = 12
a = s.symbols('a1:%d' % (NK+1)); b = s.symbols('b1:%d' % (NK+1))
fz = 1+sum(a[k-1]*z**k for k in range(1, NK+1))
hz = (1+sum(b[k-1]*z**k for k in range(1, NK+1)))/z**2
# d/dr = -z^2 d/dz
dr = lambda e: -z**2*s.diff(e, z)
sub = {r: 1/z, f: fz, h: hz, p: dr(fz), q: dr(hz), t: dr(dr(hz)), C2: 32}
ser = {}
for nm in ('eb', 'eh'):
    t0 = time.time()
    e = s.together(E[nm].xreplace(sub))
    num, den = s.fraction(e)
    num = s.Poly(s.expand(num), z)
    lo = min(m[0] for m in num.monoms())
    co = {m[0]-lo: c for m, c in zip(num.monoms(), num.coeffs())}
    ser[nm] = [co.get(k, 0) for k in range(NK+1)]
    print(nm, 'lowest power', lo, time.time()-t0, flush=True)
pickle.dump(ser, open('asym_raw.pkl', 'wb'))
sol = {}
for k in range(NK+1):
    eqs = [s.expand(ser[nm][k].subs(sol)) for nm in ('eb', 'eh')]
    unk = [v for v in (a+b) if any(e.has(v) for e in eqs) and v not in sol]
    print('order', k, 'unknowns', unk, 'eqs', [str(e)[:80] for e in eqs], flush=True)
    eqs = [e for e in eqs if e != 0]
    if eqs:
        ss = s.solve(eqs, unk, dict=True)
        print('   ->', ss, flush=True)
        if ss:
            sol.update({kk: s.factor(vv) for kk, vv in ss[0].items()})
            sol = {kk: s.factor(s.sympify(vv).subs(sol)) for kk, vv in sol.items()}
pickle.dump(sol, open('asym_sol.pkl', 'wb'))
for kk in sorted(sol, key=str):
    print(kk, '=', sol[kk])
