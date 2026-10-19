"""Quad-precision (__float128) outward integrator for the reduced fixed-J system.
   y = (f, h, h'),  f' = F1, h' = h', h'' = F2  (same polynomials as ode_poly.pkl, exact integer coefficients).
Bulirsch-Stoer (modified midpoint, polynomial extrapolation in h^2, n_k = 2,4,6,...,2 KMAX).
API (ctypes, strings for quad values):
   int qint(const char *r0, const char *r1, const char *alpha, const char *y0, const char *y1,
            const char *y2, double tol, char *out, int *nacc)
   out: "r f h hp" of the last accepted point (36 digits); return 0 = reached r1,
   1 = h < 0.05 r^2, 2 = h > 20 r^2, 3 = derivative blow-up, 4 = step failure, 5 = non-finite."""
import ctypes, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
SO = HERE/'_qrhs.so'


def generate():
    from ode_poly import TABLES, NAMES
    md = [0]*5
    for nm in NAMES:
        for m, ka, c in TABLES[nm]:
            for i in range(4):
                md[i] = max(md[i], m[i])
            md[4] = max(md[4], ka)
    L = ['#include <quadmath.h>', '#include <stdio.h>', '#include <stdlib.h>', 'typedef __float128 Q;',
         'static void coefs(Q r, Q f, Q h, Q q, Q a, Q *o){']
    for v, n in zip(('r', 'f', 'h', 'q', 'a'), md):
        L.append(f'  Q P{v}[{n+1}]; P{v}[0]=1; for(int k=1;k<={n};k++) P{v}[k]=P{v}[k-1]*{v};')
    for j, nm in enumerate(NAMES):
        terms = [f'(({c})*Pa[{ka}])*Pr[{m[0]}]*Pf[{m[1]}]*Ph[{m[2]}]*Pq[{m[3]}]' for m, ka, c in TABLES[nm]]
        L.append(f'  o[{j}] = '+'\n    + '.join(terms)+';')
    L.append('}')
    L.append(r'''
static Q ALPHA;
static void rhs(Q r, const Q *y, Q *dy){
  Q o[6]; coefs(r,y[0],y[1],y[2],ALPHA,o);
  Q det = o[0]*o[4]-o[3]*o[1];
  dy[0] = (-o[2]*o[4]+o[5]*o[1])/det;
  dy[1] = y[2];
  dy[2] = (-o[0]*o[5]+o[3]*o[2])/det; }

#define KMAX 14
static void mmid(Q x, const Q *y, Q H, int n, Q *out){
  Q h = H/n, zm[3], zn[3], d[3], t;
  for(int i=0;i<3;i++) zm[i]=y[i];
  rhs(x, y, d);
  for(int i=0;i<3;i++) zn[i]=y[i]+h*d[i];
  for(int m=1;m<n;m++){
    rhs(x+m*h, zn, d);
    for(int i=0;i<3;i++){ t=zm[i]+2*h*d[i]; zm[i]=zn[i]; zn[i]=t; }
  }
  rhs(x+H, zn, d);
  for(int i=0;i<3;i++) out[i]=0.5Q*(zm[i]+zn[i]+h*d[i]);
}

/* one BS step; returns converged column k or -1 */
static int bsstep(Q x, const Q *y, Q H, Q tol, Q *ynew){
  Q T[KMAX][KMAX][3]; int n[KMAX];
  for(int k=0;k<KMAX;k++){
    n[k]=2*(k+1);
    mmid(x,y,H,n[k],T[k][0]);
    for(int j=1;j<=k;j++){
      Q rr=(Q)n[k]/n[k-j]; rr=rr*rr-1;
      for(int i=0;i<3;i++) T[k][j][i]=T[k][j-1][i]+(T[k][j-1][i]-T[k-1][j-1][i])/rr;
    }
    if(k>=2){
      Q err=0;
      for(int i=0;i<3;i++){
        Q sc = fabsq(T[k][k][i]) + fabsq(y[i]) + 1e-30Q;
        Q e = fabsq(T[k][k][i]-T[k][k-1][i])/sc;
        if(!(e==e)) return -1;
        if(e>err) err=e;
      }
      if(err<tol){ for(int i=0;i<3;i++) ynew[i]=T[k][k][i]; return k; }
    }
  }
  return -1;
}

int qint(const char *r0s, const char *r1s, const char *as, const char *y0s, const char *y1s,
         const char *y2s, double tolD, char *out, int *nacc){
  Q x=strtoflt128(r0s,NULL), x1=strtoflt128(r1s,NULL); ALPHA=strtoflt128(as,NULL);
  Q y[3]={strtoflt128(y0s,NULL),strtoflt128(y1s,NULL),strtoflt128(y2s,NULL)}, yn[3], d[3];
  Q tol=(Q)tolD;
  Q H = (x - floorq(x))*0 + 0.02Q*(x1-x);
  /* start with a step comparable to the distance from the horizon scale: caller sets r0 = rH + x0 */
  { Q x0 = strtoflt128(r0s,NULL); (void)x0; }
  H = 1e-3Q*(x1-x);
  int code=0; *nacc=0;
  int fails=0;
  while(x<x1){
    if(x+H>x1) H=x1-x;
    int k=bsstep(x,y,H,tol,yn);
    if(k<0){ H*=0.4Q; if(++fails>200){code=4;break;} continue; }
    fails=0;
    x+=H; for(int i=0;i<3;i++) y[i]=yn[i]; (*nacc)++;
    if(k<=4) H*=1.6Q; else if(k>=9) H*=0.6Q;
    if(!(y[0]==y[0]&&y[1]==y[1]&&y[2]==y[2])){code=5;break;}
    if(y[1]<0.05Q*x*x){code=1;break;}
    if(y[1]>20*x*x){code=2;break;}
    rhs(x,y,d);
    if(fabsq(d[0])*x>1e3Q || fabsq(d[2])*x*x/fabsq(y[1])>1e3Q){code=3;break;}
  }
  char b[4][64];
  quadmath_snprintf(b[0],64,"%.36Qe",x); quadmath_snprintf(b[1],64,"%.36Qe",y[0]);
  quadmath_snprintf(b[2],64,"%.36Qe",y[1]); quadmath_snprintf(b[3],64,"%.36Qe",y[2]);
  sprintf(out,"%s %s %s %s",b[0],b[1],b[2],b[3]);
  return code;
}
''')
    (HERE/'_qrhs.c').write_text('\n'.join(L)+'\n')
    subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-o', str(SO), str(HERE/'_qrhs.c'), '-lquadmath'], check=True)


_lib = None


def qintegrate(r0, r1, alpha, y0, tol=1e-30):
    """r0, r1, alpha, y0: mpmath numbers (or anything str() gives >= 36 digits); returns (code, r, y, nacc)"""
    global _lib
    if _lib is None:
        if not SO.exists():
            generate()
        _lib = ctypes.CDLL(str(SO))
        _lib.qint.restype = ctypes.c_int
    import mpmath as mp
    s = lambda v: mp.nstr(mp.mpf(v), 40, strip_zeros=False).encode() if not isinstance(v, str) else v.encode()
    out = ctypes.create_string_buffer(512); nacc = ctypes.c_int(0)
    code = _lib.qint(s(r0), s(r1), s(alpha), s(y0[0]), s(y0[1]), s(y0[2]), ctypes.c_double(tol), out, ctypes.byref(nacc))
    parts = out.value.decode().split()
    return code, mp.mpf(parts[0]), [mp.mpf(p) for p in parts[1:]], nacc.value


if __name__ == '__main__':
    import time, numpy as np
    import mpmath as mp
    from scipy.integrate import solve_ivp
    from build_crhs import crhs
    generate()
    mp.mp.dps = 40
    # regular test: compare with double DOP853 on a smooth stretch away from the horizon
    al = 1.5; y0 = [0.3, 1.2, 1.4]; r0, r1 = 1.6, 4.0
    t = time.time(); code, r, y, n = qintegrate(r0, r1, al, y0, tol=1e-28); tq = time.time()-t
    sol = solve_ivp(crhs, (r0, r1), y0, args=(al,), method='DOP853', rtol=1e-13, atol=1e-15)
    print('quad code', code, 'r', mp.nstr(r, 20), 'y', [mp.nstr(v, 25) for v in y], 'steps', n, f'{tq:.2f}s')
    print('double  ', sol.y[:, -1], ' rel diff', [float(abs(y[i]/sol.y[i, -1]-1)) for i in range(3)])
    code2, r2, y2, n2 = qintegrate(r0, r1, al, y0, tol=1e-22)
    print('tol 1e-22 vs 1e-28 rel diff', [mp.nstr(abs(y2[i]/y[i]-1), 3) for i in range(3)])
