"""Control: G_111 at x=3/2 by direct quadrature vs the series of c_constants.py."""
import mpmath as mp
import c_constants as cc
G11 = lambda t: mp.log(1-1/t)**2/2+mp.polylog(2, 1/t)
G111 = lambda x: -mp.quad(lambda t: G11(t)/(t-1), [x, 3, 10, mp.inf])
with mp.workdps(40):
    print('quad G111(3/2)', G111(mp.mpf(3)/2))
    for u in [mp.mpf('1e-6'), mp.mpf('1e-8')]:
        l = mp.log(u)
        print('G111(1+u)-(ell^3/6+z2 ell)', u, G111(1+u)-(l**3/6+mp.zeta(2)*l))
C = {(1,): 0}
E1 = cc.hor_series([(1,)], C)[(1,)]
E11 = cc.step(E1, 1); E11[(0, 0)] = E11.get((0, 0), 0)+mp.zeta(2)
E111 = cc.step(E11, 1)
print('E11(1/2)  ', cc.evalE(E11), ' direct', G11(mp.mpf(3)/2))
print('E111(1/2) without constant', cc.evalE(E111))
print('large terms in E111:', sorted(((k, mp.nstr(v, 5)) for k, v in E111.items() if k[0] < 3)))
