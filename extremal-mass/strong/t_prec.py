import numpy as np, sys, mpmath as mp
from extremal_shoot import build
from ode_poly import TABLES, NAMES
mp.mp.dps = 50
alpha = float(sys.argv[1])
S, I, gerr = build(alpha, 12)
def coeffs_mp(r, f, h, hp, al):
    out = []
    for nm in NAMES:
        acc = mp.mpf(0); mag = mp.mpf(0)
        for m, ka, c in TABLES[nm]:
            term = c*al**ka*r**m[0]*f**m[1]*h**m[2]*hp**m[3]
            acc += term; mag += abs(term)
        out.append((acc, mag))
    return out
al = mp.mpf(alpha)
for L in (8, 10, 12):
    print('--- series truncated at level', L)
    for x in [mp.mpf('1e-3'), mp.mpf('3e-3'), mp.mpf('1e-2'), mp.mpf('2e-2')]:
        F = [mp.mpf(v) for v in S.F[:L]]; H = [mp.mpf(v) for v in S.H[:L+1]]
        r = mp.mpf(S.rH)+x
        f = x**2*sum(F[k]*x**k for k in range(len(F)))
        fp = sum((k+2)*F[k]*x**(k+1) for k in range(len(F)))
        h = sum(H[k]*x**k for k in range(len(H)))
        hp = sum(k*H[k]*x**(k-1) for k in range(1, len(H)))
        hpp = sum(k*(k-1)*H[k]*x**(k-2) for k in range(2, len(H)))
        (Ab, mAb), (Bb, mBb), (Cb, mCb), (Ah, mAh), (Bh, mBh), (Ch, mCh) = coeffs_mp(r, f, h, hp, al)
        det = Ab*Bh-Ah*Bb
        n1 = -Cb*Bh+Ch*Bb
        Nb = Ab*fp+Bb*hpp+Cb; Nh = Ah*fp+Bh*hpp+Ch
        print(f'x={mp.nstr(x,3)}: fp ode(mp)={mp.nstr(n1/det, 12)} series={mp.nstr(fp, 12)} | Nb rel={mp.nstr(abs(Nb)/(mAb*abs(fp)+mBb*abs(hpp)+mCb), 3)} Nh rel={mp.nstr(abs(Nh)/(mAh*abs(fp)+mBh*abs(hpp)+mCh), 3)} | det/(|Ab Bh|)={mp.nstr(det/abs(Ab*Bh), 3)}')
