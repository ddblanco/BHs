"""Compile the reduced fixed-J system to C (long double) and load it with ctypes.
   void rhs(double r, const double *y, double alpha, double *dy)   y = (f, h, h'), dy = (f', h', h'')
   void coef(double r, double f, double h, double hp, double alpha, double *out6)
Built from ode_poly.pkl (same monomial tables as ode_poly.py)."""
import ctypes, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SO = HERE/'_crhs.so'


def generate():
    from ode_poly import TABLES, NAMES
    lines = ['#include <math.h>', 'typedef long double LD;']
    lines.append('static void coefs(LD r, LD f, LD h, LD hp, LD a, LD *o){')
    md = [0, 0, 0, 0, 0]
    for nm in NAMES:
        for m, ka, c in TABLES[nm]:
            for i in range(4):
                md[i] = max(md[i], m[i])
            md[4] = max(md[4], ka)
    for v, n in zip(('r', 'f', 'h', 'hp', 'a'), md):
        lines.append(f'  LD {v}p[{n+1}]; {v}p[0]=1.0L; for(int k=1;k<={n};k++) {v}p[k]={v}p[k-1]*{v};')
    for j, nm in enumerate(NAMES):
        terms = [f'({c}.0L)*ap[{ka}]*rp[{m[0]}]*fp[{m[1]}]*hp_[{m[2]}]*hpp_[{m[3]}]' for m, ka, c in TABLES[nm]]
        lines.append(f'  o[{j}] = ' + ' + '.join(terms) + ';')
    lines.append('}')
    src = '\n'.join(lines)
    # rename power arrays to avoid clashes
    src = src.replace('LD fp[', 'LD fp_[').replace('fp[0]=', 'fp_[0]=').replace('fp[k]=fp[k-1]', 'fp_[k]=fp_[k-1]')
    src = src.replace('LD hp[', 'LD hpp_[').replace('hpp[0]=1.0L; for(int k=1;k<=', 'XX')  # placeholder safety
    # rebuild cleanly instead of string surgery
    lines = ['#include <math.h>', 'typedef long double LD;',
             'static void coefs(LD r, LD f, LD h, LD q, LD a, LD *o){']
    for v, n in zip(('r', 'f', 'h', 'q', 'a'), md):
        lines.append(f'  LD P{v}[{n+1}]; P{v}[0]=1.0L; for(int k=1;k<={n};k++) P{v}[k]=P{v}[k-1]*{v};')
    for j, nm in enumerate(NAMES):
        terms = [f'({c}.0L)*Pa[{ka}]*Pr[{m[0]}]*Pf[{m[1]}]*Ph[{m[2]}]*Pq[{m[3]}]' for m, ka, c in TABLES[nm]]
        lines.append(f'  o[{j}] = ' + '\n    + '.join(terms) + ';')
    lines.append('}')
    lines.append('''void coef(double r, double f, double h, double q, double a, double *out){
  LD o[6]; coefs(r,f,h,q,a,o); for(int i=0;i<6;i++) out[i]=(double)o[i]; }
void rhs(double r, const double *y, double a, double *dy){
  LD o[6]; coefs(r,y[0],y[1],y[2],a,o);
  LD det = o[0]*o[4]-o[3]*o[1];
  dy[0] = (double)((-o[2]*o[4]+o[5]*o[1])/det);
  dy[1] = y[2];
  dy[2] = (double)((-o[0]*o[5]+o[3]*o[2])/det); }
void rhs_many(int n, const double *r, const double *y, double a, double *dy){
  for(int i=0;i<n;i++) rhs(r[i], y+3*i, a, dy+3*i); }''')
    (HERE/'_crhs.c').write_text('\n'.join(lines)+'\n')
    subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-o', str(SO), str(HERE/'_crhs.c')], check=True)


def load():
    if not SO.exists():
        generate()
    lib = ctypes.CDLL(str(SO))
    D = ctypes.c_double; P = ctypes.POINTER(ctypes.c_double)
    lib.rhs.argtypes = [D, P, D, P]; lib.rhs.restype = None
    lib.coef.argtypes = [D, D, D, D, D, P]; lib.coef.restype = None
    return lib


_lib = None


def crhs(r, y, alpha):
    global _lib
    if _lib is None:
        _lib = load()
    y = np.ascontiguousarray(y, dtype=float)
    out = np.empty(3)
    _lib.rhs(float(r), y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), float(alpha),
             out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out


if __name__ == '__main__':
    import time
    generate()
    from shoot import rhs
    for args in [(1.2, np.array([0.1, 0.8, 1.1]), 1.5), (3.0, np.array([0.7, 9.1, 6.0]), 25.)]:
        a, b = crhs(*args), rhs(*args)
        print('C', a, ' py', b, ' rel', np.max(np.abs(a-b)/np.abs(b)))
    t = time.time()
    for i in range(20000):
        crhs(1.2, np.array([0.1, 0.8, 1.1]), 1.5)
    print('C rhs per call (us)', (time.time()-t)/20000*1e6)
