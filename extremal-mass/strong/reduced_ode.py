"""Build the reduced fixed-J ODE system (f, h) in the areal gauge g = r^2 and cache it."""
import sys, pickle, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'code'))
import sympy as s
from reduced_system import reduced_equations, r, alpha, f, h, p, q, t, u, v, C2
t0 = time.time()
eb, ef, eh = reduced_equations()
print('built', time.time()-t0, flush=True)
for name, e in [('eb', eb), ('ef', ef), ('eh', eh)]:
    print(name, 'free', sorted(map(str, e.free_symbols)), 'deg u', s.degree(e, u), 'deg v', s.degree(e, v), 'deg p', s.degree(e, p), 'deg t', s.degree(e, t), 'nops', s.count_ops(e), flush=True)
pickle.dump(dict(eb=eb, ef=ef, eh=eh), open(HERE/'reduced_eqs.pkl', 'wb'))
