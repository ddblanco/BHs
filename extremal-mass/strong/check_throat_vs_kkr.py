"""Independent check: the KKR-gauge numerical branch (work/kkr-extremal/branch_rq.json) stores g_H, h_H.
The exact throat map predicts s from q = h_H/g_H alone: s(q) = 2 a(q) Jbar2(q)^(-1/3)."""
import json, math
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT/'work/kkr-extremal/branch_rq.json').read_text())['rows']
D = lambda q: 5*q**4-34*q**3+73*q**2-56*q+16
def s_of_q(q):
    a = (-2*q**2+7*q-4-math.sqrt(D(q)))/(4*q**2*(q-4))
    R = 1+4*a*(4-3*q); J2 = q**2*(4-q)*R/8
    return 2*a*J2**(-1/3), a
dev = []
for row in rows[::25]+[rows[-1]]:
    hi = row['high']
    al, J, gH, hH = hi['alpha'], hi['J'], hi['g_H'], hi['h_H']
    if al == 0: continue
    s_num = math.pi**(2/3)*al/J**(2/3)
    q = hH/gH
    s_th, a = s_of_q(q)
    dev.append(abs(s_th/s_num-1))
    print(f'alpha={al:.4f} q={q:.8f} s(numerics)={s_num:.8f} s(throat map)={s_th:.8f} rel.dev {s_th/s_num-1:.1e}  a_th={a:.6f} alpha/g_H={al/gH:.6f}')
print('max rel dev', max(dev))
