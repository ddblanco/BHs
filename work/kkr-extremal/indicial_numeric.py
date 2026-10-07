"""Indicial exponents of the extremal residual operator at x = 0, numerically,
about extremal Myers-Perry (alpha = 0) or any computed background.

Perturbation dB = cB x^s, dF = cF x^s, dH = cH x^s, dW = cW x^(s+1).
The linear response of each residual at small x is ~ x^(s+q_i) M_ij(s) c_j;
M(s) is read off at two small x (complex step in c_j), its determinant scanned
in s. Roots with s > -1/2 are regular directions at the horizon, roots below
are the singular modes excluded by regularity.
"""
import numpy as np

from _extremal_generated import residual


def mp_jets(x):
    z = 1-x
    B = 1/(1+z**4); W = np.sqrt(2)/(1+z**4)
    dB = 4*z**3/(1+z**4)**2                       # d/dx of 1/(1+z^4), z = 1-x
    ddB = -12*z**2/(1+z**4)**2+32*z**6/(1+z**4)**3
    return dict(B=B, Bx=dB, Bxx=ddB, F=1., Fx=0., H=1., Hx=0., Hxx=0.,
                W=W, Wx=np.sqrt(2)*dB, Wxx=np.sqrt(2)*ddB)


ORDER = ('B', 'Bx', 'Bxx', 'F', 'Fx', 'H', 'Hx', 'Hxx', 'W', 'Wx', 'Wxx')


def response(x, s, alpha=0.):
    base = mp_jets(x)
    M = np.zeros((4, 4), dtype=float)
    for j, field in enumerate('BFHW'):
        p = s+1 if field == 'W' else s
        jets = {k: complex(v) for k, v in base.items()}
        h = 1e-30
        jets[field] += 1j*h*x**p
        jets[field+'x'] += 1j*h*p*x**(p-1)
        if field+'xx' in jets:
            jets[field+'xx'] += 1j*h*p*(p-1)*x**(p-2)
        R = residual(x, *[jets[k] for k in ORDER], alpha)
        M[:, j] = np.array([r.imag for r in R])/h
    return M


def leading(s):
    """Row-wise leading behaviour: fit M(x) ~ x^(s+q) A for each row."""
    x1, x2 = 1e-7, 1e-8
    M1, M2 = response(x1, s), response(x2, s)
    A = np.zeros((4, 4))
    for i in range(4):
        n1, n2 = np.max(np.abs(M1[i])), np.max(np.abs(M2[i]))
        power = np.log(n1/n2)/np.log(x1/x2)
        A[i] = M2[i]/x2**power
    return A


def main():
    grid = np.linspace(-4.2, 3.2, 1481)
    dets = []
    for s in grid:
        A = leading(s)
        A = A/np.abs(A).max(axis=1, keepdims=True)
        dets.append(np.linalg.svd(A, compute_uv=False)[-1])
    dets = np.array(dets)
    print('local minima of sigma_min(M(s)) (indicial roots at alpha = 0):')
    for i in range(1, len(grid)-1):
        if dets[i] < dets[i-1] and dets[i] < dets[i+1] and dets[i] < 1e-2:
            print(f'   s = {grid[i]:+.3f}   sigma_min = {dets[i]:.1e}')


if __name__ == '__main__':
    main()
