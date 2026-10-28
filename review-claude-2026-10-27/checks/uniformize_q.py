"""New test: re-expand the exact weak-coupling series of m(s) in the algebraic throat variable
q = h_H/g_H (q = 2 at Myers-Perry, q -> 0 at strong coupling).  The map s(q) is exact (algebraic):
    a(q) = (-2q^2+7q-4-sqrt D)/(4q^2(q-4)),  D = 5q^4-34q^3+73q^2-56q+16,
    R = 1+4a(4-3q),  Jbar^2 = q^2(4-q)R/8,  s = 2 a Jbar^(-2/3)
(derived independently in check_throat2.py).  The fold s* of s(q) is a critical point of the map, so in
q the fold disappears; the question is whether y(q) = m - 3s/4 has a Taylor series at q = 2 that is useful
over the physical range.  Inputs: exact d_1..d_6 (series/coefficients.json), numerical branch
(results/egb-extremality.json, work/kkr-extremal/analysis.json, extremal-mass/strong/numerics.tex).
"""
import json, re
from pathlib import Path
import sympy as s, mpmath as mp, numpy as np
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
mp.mp.dps = 40
q, w = s.symbols('q w', positive=True)
D = 5*q**4 - 34*q**3 + 73*q**2 - 56*q + 16
a_q = (-2*q**2 + 7*q - 4 - s.sqrt(D))/(4*q**2*(q - 4))
R_q = 1 + 4*a_q*(4 - 3*q)
Jb2 = q**2*(4 - q)*R_q/8
s_q = 2*a_q*Jb2**s.Rational(-1, 3)
NW = 8
s_w = s.series(s_q.subs(q, 2 - w), w, 0, NW + 1).removeO()
s_w = s.expand(s_w)
print('s(q) around q = 2, w = 2 - q:')
for k in range(0, NW + 1):
    print(f'   w^{k}: {s_w.coeff(w, k)}')
coef = json.loads((ROOT/'extremal-mass/series/coefficients.json').read_text())
d = {int(k): s.Rational(str(mp.mpf(v['value']))) for k, v in coef.items()}   # numerical values, 30 digits
d = {int(k): s.Float(v['value'], 30) for k, v in coef.items()}
m_s = s.Rational(3, 2) + sum(d[n]*s.Symbol('S')**n for n in range(1, 7))
y_s = m_s - s.Rational(3, 4)*s.Symbol('S')
y_w = s.expand(s.series(y_s.subs(s.Symbol('S'), s_w), w, 0, 7).removeO())
e = [y_w.coeff(w, k) for k in range(7)]
print('\ny(q) = m - 3s/4 around q = 2 (coefficients of w^k, w = 2 - q):')
for k, c in enumerate(e): print(f'   w^{k}: {s.N(c, 12)}')
# singularities of the throat data in the q plane
roots = [complex(r_) for r_ in s.Poly(D, q).nroots(n=15)]
print('\nroots of D(q):', [f'{z.real:.6f}{z.imag:+.6f}i' for z in roots], ' |2-q| =', [f'{abs(2-z):.4f}' for z in roots])
# numeric map s -> q on the physical branch
def s_of_q(qq):
    qq = mp.mpmathify(qq); Dv = 5*qq**4 - 34*qq**3 + 73*qq**2 - 56*qq + 16
    aa = (-2*qq**2 + 7*qq - 4 - mp.sqrt(Dv))/(4*qq**2*(qq - 4))
    Rv = 1 + 4*aa*(4 - 3*qq); J2 = qq**2*(4 - qq)*Rv/8
    return 2*aa*J2**(-mp.mpf(1)/3)
def q_of_s(sv):
    # s(q) is monotone decreasing on (0, 2]: bisection
    lo, hi = mp.mpf('1e-6'), mp.mpf(2)
    for _ in range(200):
        mid = (lo + hi)/2
        if s_of_q(mid) > sv: lo = mid
        else: hi = mid
    return (lo + hi)/2
# data
P13, P23 = np.pi**(1/3), np.pi**(2/3)
pts = []
for r_ in json.loads((ROOT/'results/egb-extremality.json').read_text())['extremals']:
    if r_['y'] > 0: pts.append((P23*r_['y'], r_['mu']/P13, 'manuscript'))
for r_ in json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch']:
    pts.append((r_['s'] if 's' in r_ and False else P23*r_['y'], r_['mu']/P13, 'spectral'))
for sv, mv in re.findall(r'^(\d+\.?\d*) & (\d+\.\d+) & ', (ROOT/'extremal-mass/strong/numerics.tex').read_text(), re.M):
    pts.append((float(sv), float(mv), 'strong'))
pts.sort()
# partial sums in w and in s
ef = [float(c) for c in e]; df = [0] + [float(d[n]) for n in range(1, 7)]
print('\n   s        q      y_num      y_w[2]     y_w[4]     y_w[6]    err_w6    | y_s[6]    err_s6   source')
rows = []
for sv, mv, src in pts:
    if sv > 120: continue
    qq = float(q_of_s(sv)); ww = 2 - qq
    yn = mv - 0.75*sv
    yw = [sum(ef[k]*ww**k for k in range(N + 1)) for N in (2, 4, 6)]
    ys6 = 1.5 + sum((df[n] - 0.75*(n == 1))*sv**n for n in range(1, 7))
    rows.append((sv, qq, yn, yw[2], ys6, src))
    if (src == 'manuscript' or (src == 'strong' and sv < 12) or (src == 'spectral' and abs(sv - round(sv)) < 0.0005)):
        print(f'{sv:8.4f} {qq:7.4f} {yn:10.6f} {yw[0]:10.6f} {yw[1]:10.6f} {yw[2]:10.6f} {yw[2]-yn:9.1e} | {ys6:9.4f} {ys6-yn:9.1e}  {src}')
# Pade [3/3] in w from the six coefficients (e_0..e_6) and comparison
mp.mp.dps = 30
pw, qw = mp.pade([mp.mpf(str(c)) for c in e], 3, 3)
def pade_w(ww):
    return mp.polyval(pw[::-1], ww)/mp.polyval(qw[::-1], ww)
print('\nPade [3/3] in w of y(q):')
print('   s        q      y_num      pade      err')
worst = {}
for sv, qq, yn, yw6, ys6, src in rows:
    pv = float(pade_w(2 - qq))
    worst[src] = max(worst.get(src, 0), abs(pv - yn))
    if (src == 'manuscript' or (src == 'strong' and sv < 12) or (src == 'spectral' and abs(sv - round(sv)) < 0.0005)):
        print(f'{sv:8.4f} {qq:7.4f} {yn:10.6f} {pv:10.6f} {pv-yn:9.1e}  {src}')
print('worst |pade - y_num| by source:', worst)
print('poles of the Pade denominator in w:', [mp.nstr(z, 8) for z in mp.polyroots(qw[::-1])])
# singularities of the throat data in the a-plane: discriminant of P in q (report: 64 a^2 (4a+1) Q4(a))
a_ = s.symbols('a')
Q4 = 131072*a_**4 + 28672*a_**3 + 6528*a_**2 - 92*a_ + 5
print('\nroots of the quartic factor of disc_q P in a:', [complex(z) for z in s.Poly(Q4, a_).nroots(n=12)])
print('throat branch point s* = 0.0059231771 + 0.0548188529 i  ->  compare with s(q_c) at the critical points of s(q):')
# critical points of s(q): numerically, ds/dq = 0 near q_c = 2.16110 - 0.94332 i (report)
qc = mp.findroot(lambda z: mp.diff(s_of_q, z), mp.mpc('2.1611', '-0.9433'))
print('   q_c =', mp.nstr(qc, 10), '  s(q_c) =', mp.nstr(s_of_q(qc), 10), '  a(q_c) =', mp.nstr((-2*qc**2+7*qc-4-mp.sqrt(5*qc**4-34*qc**3+73*qc**2-56*qc+16))/(4*qc**2*(qc-4)), 10))
# Pade [3/3] of m(s) itself from d_1..d_6
ps_, qs_ = mp.pade([mp.mpf('1.5')] + [mp.mpf(str(d[n])) for n in range(1, 7)], 3, 3)
print('\nPade [3/3] of m(s) from d_1..d_6: poles at s =', [mp.nstr(z, 8) for z in mp.polyroots(qs_[::-1])])
print('   s        m_num      pade[3/3]   err        source')
for sv, mv, src in pts:
    if src == 'spectral' and abs(sv - round(sv)) > 0.0005: continue
    if sv > 3: break
    pv = float(mp.polyval(ps_[::-1], sv)/mp.polyval(qs_[::-1], sv))
    print(f'{sv:8.4f} {mv:10.6f} {pv:10.6f} {pv-mv:9.1e}  {src}')
json.dump(dict(s_w=[str(s_w.coeff(w, k)) for k in range(NW+1)], y_w=[str(s.N(c, 20)) for c in e]), open(HERE/'uniformize_q.json', 'w'), indent=1)
