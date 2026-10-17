"""The throat squashing q = h_H/g_H as uniformizing variable.
Physical branch: a(q) = (-2q^2+7q-4-sqrt(D))/(4q^2(q-4)), D = 5q^4-34q^3+73q^2-56q+16,
Jbar2 = q^2(4-q)R/8, R = 1+4a(4-3q), g_H = Jbar2^(-1/3), s = 2 a g_H.
Locate the branch point s* of g_H(s) in the q-plane; singularities of s(q)."""
import mpmath as mp
mp.mp.dps = 40
D = lambda q: 5*q**4-34*q**3+73*q**2-56*q+16
def a_of(q): return (-2*q**2+7*q-4-mp.sqrt(D(q)))/(4*q**2*(q-4))
def s_of(q):
    a = a_of(q); R = 1+4*a*(4-3*q); J = q**2*(4-q)*R/8
    return 2*a*J**(-mp.mpf(1)/3)
sstar = mp.mpc('0.0059231771', '0.0548188529')
# continue from q=2 along straight path in s from 0 to s*: track q by Newton
q = mp.mpf(2)
N = 400
for k in range(1, N+1):
    target = sstar*k/N
    for it in range(30):
        f = s_of(q)-target
        dq = f/mp.diff(s_of, q)
        q -= dq
        if abs(dq) < mp.mpf(10)**-30: break
print('q* (image of s*) =', mp.nstr(q, 15), '  ds/dq there =', mp.nstr(mp.diff(s_of, q), 6))
print('roots of D(q):', [mp.nstr(r, 10) for r in mp.polyroots([5, -34, 73, -56, 16])])
# critical points of s(q): solve ds/dq=0 near q*
qc = mp.findroot(lambda z: mp.diff(s_of, z), q)
print('critical point near q*:', mp.nstr(qc, 15), ' s there:', mp.nstr(s_of(qc), 15))
# q -> s map at physical points
for qq in [2, 1.5, 1, 0.5, 0.28184900, 0.2, 0.1, 0.05]:
    print(f'q={qq}: s={mp.nstr(s_of(mp.mpf(qq)), 12)}')
