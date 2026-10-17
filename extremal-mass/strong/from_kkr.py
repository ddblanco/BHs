"""Convert a stored spectral extremal solution (work/kkr-extremal/branch_rq.json, KKR gauge,
production representation P_i = 1 + (1-s)^2 Q_i, s = r/(1+r), horizon at s = 0) to the areal
gauge g = R^2 of the reduced system, in code units J = J0.  Then evaluate the reduced
equations and det = A_b B_h - A_h B_b along it.

KKR ansatz (kkr_equations.py, power 0): u = r^2+1,  g = P2 u,  h = P2 P3 u (1+1/u^2),
g^{ss} = (r^2/u)(1-s)^4/P1,  b = P1 r^4/(u^2+1)."""
import json, math, sys
import numpy as np
from pathlib import Path
CH = np.polynomial.chebyshev
ROOT = Path(__file__).resolve().parents[2]
J0 = math.pi*math.sqrt(2)/4


def load_row(target_s):
    rows = json.loads((ROOT/'work/kkr-extremal/branch_rq.json').read_text())['rows']
    best = None
    for row in rows:
        hi = row['high']
        if hi['alpha'] <= 0: continue
        sv = math.pi**(2/3)*hi['alpha']/hi['J']**(2/3)
        if best is None or abs(sv-target_s) < abs(best[0]-target_s):
            best = (sv, row)
    return best


def fields(row, s):
    hi = row['high']
    c = [np.array(ci) for ci in row['coeffs_high']]
    t = 1-2*s
    Q = [CH.chebval(t, ci) for ci in c]
    dQ = [CH.chebval(t, CH.chebder(ci))*(-2) for ci in c]
    P = [1+(1-s)**2*Q[i] for i in range(3)]
    dP = [-2*(1-s)*Q[i]+(1-s)**2*dQ[i] for i in range(3)]
    r = s/(1-s); drds = 1/(1-s)**2
    u = r**2+1; duds = 2*r*drds
    g = P[1]*u; dg = dP[1]*u+P[1]*duds
    h = P[1]*P[2]*u*(1+1/u**2)
    gss = (r**2/u)*(1-s)**4/P[0]
    R = np.sqrt(g); dRds = dg/(2*R)
    fR = dRds**2*gss
    return R, fR, h, dRds, hi


if __name__ == '__main__':
    from ode_poly import coeffs
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    sv, row = load_row(target)
    hi = row['high']
    L = (J0/hi['J'])**(1/3)
    al = L**2*hi['alpha']
    print(f"row s={sv:.6f}  alpha_K={hi['alpha']:.6f} J_K={hi['J']:.6f} -> code alpha={al:.6f} (s=2alpha={2*al:.6f}), m={2*(hi['M']*L**2)/math.pi:.8f}")
    s = np.linspace(1e-3, 0.97, 4000)
    R, fR, h, dRds, _ = fields(row, s)
    R, h = L*R, L**2*h
    # derivatives wrt R by finite differences on the grid
    hp = np.gradient(h, R); hpp = np.gradient(hp, R); fp = np.gradient(fR, R)
    Ab, Bb, Cb, Ah, Bh, Ch = coeffs(R, fR, h, hp, al)
    det = Ab*Bh-Ah*Bb
    Nb = (Ab*fp+Bb*hpp+Cb)/(np.abs(Ab*fp)+np.abs(Bb*hpp)+np.abs(Cb))
    Nh = (Ah*fp+Bh*hpp+Ch)/(np.abs(Ah*fp)+np.abs(Bh*hpp)+np.abs(Ch))
    print('monotonic R(s):', np.all(np.diff(R) > 0), ' R_H =', R[0], ' (throat sqrt(g_H) for comparison printed by throat.py)')
    for i in list(range(0, 4000, 250))+[3999]:
        print(f'R={R[i]:.5f} f={fR[i]:.5e} h/R^2={h[i]/R[i]**2:.5f} det={det[i]:+.3e} relNb={abs(Nb[i]):.1e} relNh={abs(Nh[i]):.1e}')
    sgn = np.sign(det)
    ch = np.where(np.diff(sgn) != 0)[0]
    print('det sign changes at R =', R[ch])
