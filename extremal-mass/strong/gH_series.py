"""Exact Taylor series of the extremal throat data at fixed J0 in s = 2 alpha:
   Pt = 16 al^2 q^2 (4-q) - 8 al g (2q^2-7q+4) + g^2 (q-2) = 0
   Jt = g^2 q^2 (4-q) (g + 4 al (4-3q)) - 8 = 0
Order-by-order linear solve with exact rationals; then ratio/Pade analysis."""
from fractions import Fraction as Fr
import numpy as np, mpmath as mp, json, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

def mul(a, b):
    n = len(a); out = [Fr(0)]*n
    for i, ai in enumerate(a):
        if ai:
            for j in range(n-i):
                out[i+j] += ai*b[j]
    return out
def add(*xs):
    n = len(xs[0]); return [sum(x[i] for x in xs) for i in range(n)]
def sc(c, a): return [c*x for x in a]
def shift(a):  # multiply by alpha
    return [Fr(0)]+a[:-1]

def eqs(g, q, n):
    g, q = g[:n], q[:n]
    one = [Fr(1)]+[Fr(0)]*(n-1)
    q2 = mul(q, q); fq = add(sc(4, one), sc(-1, q))
    Pt = add(sc(16, shift(shift(mul(q2, fq)))), sc(-8, shift(mul(g, add(sc(2, q2), sc(-7, q), sc(4, one))))), mul(mul(g, g), add(q, sc(-2, one))))
    Jt = add(mul(mul(mul(g, g), mul(q2, fq)), add(g, sc(4, shift(add(sc(4, one), sc(-3, q)))))), sc(-8, one))
    return Pt, Jt

n = N+1
g = [Fr(1)]+[Fr(0)]*N; q = [Fr(2)]+[Fr(0)]*N
for k in range(1, n):
    # residual at order k is linear in (g_k, q_k): get by evaluating with g_k,q_k = 0, 1
    base = eqs(g, q, k+1)
    g[k] = Fr(1); e1 = eqs(g, q, k+1); g[k] = Fr(0)
    q[k] = Fr(1); e2 = eqs(g, q, k+1); q[k] = Fr(0)
    A = [[e1[0][k]-base[0][k], e2[0][k]-base[0][k]], [e1[1][k]-base[1][k], e2[1][k]-base[1][k]]]
    b = [-base[0][k], -base[1][k]]
    det = A[0][0]*A[1][1]-A[0][1]*A[1][0]
    g[k] = (b[0]*A[1][1]-A[0][1]*b[1])/det
    q[k] = (A[0][0]*b[1]-A[1][0]*b[0])/det
# convert to s = 2 alpha
gs = [g[k]/2**k for k in range(n)]
print('g_H(s) first terms:', [str(c) for c in gs[:6]])
mp.mp.dps = 50
c = [mp.mpf(x.numerator)/x.denominator for x in gs]
for k in range(N-8, N+1, 2):
    print(k, 'c_k^(-1/k) =', mp.nstr(abs(c[k])**(-mp.mpf(1)/k), 8), ' ratio c_{k-1}/c_k =', mp.nstr(c[k-1]/c[k], 8))
# Pade / roots of denominator to locate the nearest singularities
mp.mp.dps = 120
c = [mp.mpf(x.numerator)/x.denominator for x in gs]
for L in (8, 12, 16):
    try:
        p, qd = mp.pade(c[:2*L+1], L, L)
    except ZeroDivisionError:
        print('L', L, 'singular'); continue
    rts = sorted(mp.polyroots(qd[::-1], maxsteps=800, extraprec=800), key=abs)
    print('L', L, 'Pade poles (nearest):', [mp.nstr(r_, 8) for r_ in rts[:4]])
json.dump(dict(g_H_series_s=[str(x) for x in gs[:21]], q_series_alpha=[str(x) for x in q[:21]]), open('gH_series.json', 'w'), indent=1)
