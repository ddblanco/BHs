"""Independent closed-form and intrinsic-geometry checks for Hito 4C."""
import numpy as np
import pytest
import sympy as sp
from rotating_bh.myers_perry import MyersPerry
from rotating_bh.egb_rotating_observables import measure, ergosurface


class ExactProfile:
    def __init__(self, q, alpha=0):
        self.omega_h, self.alpha_gb = q, alpha
        x = sp.symbols('x')
        z = 1-x
        if alpha:
            assert q == 0
            # Rationalized static amplitude, regular also at the horizon.
            root = sp.sqrt(1+8*alpha*(1+2*alpha)*z**4)
            B = 2/(1+root+4*alpha*z*z)
            expr = [B, B, sp.Integer(0), sp.Integer(0)]
        else:
            a = q*q/(1-q*q)
            expr = [(1-a*z*z)/(1+a*z**4), 1-a*z*z,
                    sp.Float(a), q/(1-q*q+q*q*z**4)]
        self.fns = [[sp.lambdify(x, sp.diff(e, x, k), 'numpy') for e in expr]
                    for k in range(3)]

    def evaluate(self, x, derivative=0):
        return np.asarray([np.broadcast_to(f(x), np.shape(x)) for f in self.fns[derivative]])


@pytest.mark.parametrize('q', [0., .3, -.3, .5])
def test_myers_perry_charges_horizon_and_ergo(q):
    solution = ExactProfile(q)
    data = measure(solution)
    expected = MyersPerry(1, q).thermodynamics()
    for name, oracle in [('E','M'), ('J','J1'), ('T_H','T_H'), ('S','S'), ('A_H','A_H')]:
        assert data[name] == pytest.approx(expected[oracle], rel=2e-7, abs=1e-10)
    assert ergosurface(solution) == pytest.approx(1/np.sqrt(1-q*q), abs=1e-11)


@pytest.mark.parametrize('alpha', [.05, .1, .25])
def test_static_charges_and_wald_entropy(alpha):
    data = measure(ExactProfile(0, alpha))
    assert data['E'] == pytest.approx(3*np.pi*(1+2*alpha)/8, rel=2e-7)
    assert data['T_H'] == pytest.approx(1/(2*np.pi*(1+4*alpha)), rel=1e-12)
    assert data['S'] == pytest.approx(np.pi**2/2*(1+12*alpha), rel=1e-12)
    assert data['J'] == 0


def test_wald_horizon_curvature_from_intrinsic_metric():
    theta = sp.symbols('u', positive=True)
    a, h = sp.symbols('a h', positive=True)
    # u=sin²(theta) turns the exact curvature calculation into rational algebra.
    s, c = theta, 1-theta
    metric = sp.Matrix([[a/(4*s*c),0,0], [0,h*s+(a-h)*s*c,-(a-h)*s*c],
                        [0,-(a-h)*s*c,h*c+(a-h)*s*c]])
    inv = sp.simplify(metric.inv())
    d = lambda e, i: sp.diff(e, theta) if i == 0 else 0
    gamma = [[[sp.simplify(sum(inv[k,l]*(d(metric[l,j],i)+d(metric[l,i],j)
               -d(metric[i,j],l)) for l in range(3))/2) for j in range(3)]
               for i in range(3)] for k in range(3)]
    ricci = sp.Matrix(3,3,lambda i,j: sum(d(gamma[k][i][j],k)-d(gamma[k][i][k],j)
        +sum(gamma[k][k][l]*gamma[l][i][j]-gamma[k][j][l]*gamma[l][i][k]
             for l in range(3)) for k in range(3)))
    scalar = sum(inv[i,j]*ricci[i,j] for i in range(3) for j in range(3))
    assert sp.factor(scalar-2*(4*a-h)/a**2) == 0
