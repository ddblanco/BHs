"""Physical regressions: signs, normalization, exterior domain and first law."""

import numpy as np
import pytest

from rotating_bh.myers_perry import MyersPerry


def test_static_tangherlini_normalization():
    bh = MyersPerry(2, 0, G5=3)
    r = np.array([2, 3, 8.0])
    values = bh.functions(r)
    np.testing.assert_allclose(values['b'], 1-4/r**2)
    np.testing.assert_allclose(values['f'], 1-4/r**2)
    np.testing.assert_allclose(values['h'], r**2)
    np.testing.assert_array_equal(values['w'], 0*r)
    th = bh.thermodynamics()
    assert th['M'] == pytest.approx(np.pi/2)
    assert th['T_H'] == pytest.approx(1/(4*np.pi))
    assert th['A_H'] == pytest.approx(16*np.pi**2)
    assert th['S'] == pytest.approx(4*np.pi**2/3)
    assert th['J1'] == th['J2'] == 0


@pytest.mark.parametrize('args', [(0,0,1),(-1,0,1),(1,np.nan,1),
    (np.inf,0,1),(1,0,0),(1,0,-1),(1,0,np.inf),
    (1,1/np.sqrt(2),1),(1,-1/np.sqrt(2),1),(2,0.5,1)])
def test_invalid_or_non_outer_nonextremal_parameters(args):
    with pytest.raises(ValueError):
        MyersPerry(*args)


@pytest.mark.parametrize('value', [np.array([1.0]), 1+0j, '1', True, None])
def test_parameters_must_be_real_scalars(value):
    with pytest.raises(ValueError):
        MyersPerry(value, 0.1)
    with pytest.raises(ValueError):
        MyersPerry(1, value)
    with pytest.raises(ValueError):
        MyersPerry(1, 0.1, value)


@pytest.mark.parametrize('r', [0, -1, 0.9, np.inf, np.nan, [1, 0.9]])
def test_radial_domain(r):
    with pytest.raises(ValueError):
        MyersPerry(1, 0.3).functions(r)


@pytest.mark.parametrize('q', [-0.68, -0.33, 0, 0.33, 0.68])
def test_horizon_exterior_and_thermodynamics(q):
    bh = MyersPerry(1.7, q/1.7, G5=2.3)
    v = bh.functions(bh.r_h)
    th, hd = bh.thermodynamics(), bh.horizon_data()
    assert v['b'] == v['f'] == 0
    assert v['w'] == pytest.approx(th['Omega_H'])
    assert v['h'] == pytest.approx(hd['h_H'])
    assert hd['b1'] > 0 and hd['f1'] > 0
    near = bh.functions(bh.r_h*(1+1e-6))
    assert near['b']/(bh.r_h*1e-6) == pytest.approx(hd['b1'], rel=5e-5)
    assert near['f']/(bh.r_h*1e-6) == pytest.approx(hd['f1'], rel=5e-5)
    outer = bh.functions(bh.r_h*np.geomspace(1.0001,100,30))
    assert np.all(outer['b']>0) and np.all(outer['f']>0)
    assert 2*th['M']/3 == pytest.approx(th['T_H']*th['S']+
        th['Omega_H']*(th['J1']+th['J2']), rel=2e-14)


def test_spin_reversal_and_scale():
    positive, negative = MyersPerry(1,0.4), MyersPerry(1,-0.4)
    p, n = positive.functions(2), negative.functions(2)
    for key in ('b','f','h'):
        assert p[key] == n[key]
    assert p['w'] == -n['w']
    a,b = positive.thermodynamics(), MyersPerry(3,0.4/3).thermodynamics()
    for key, power in [('M',2),('J1',3),('J2',3),('S',3),('A_H',3),('T_H',-1)]:
        assert b[key] == pytest.approx(a[key]*3**power)


@pytest.mark.parametrize('q', [-0.6,0,0.4,0.68])
def test_first_law_two_independent_parameter_variations(q):
    rh, G5 = 1.3, 2.1
    th = MyersPerry(rh,q/rh,G5).thermodynamics()
    for direction in (0,1):
        step = 1e-5
        def evaluate(offset):
            radius = rh+offset if direction == 0 else rh
            spin = q+offset if direction == 1 else q
            return MyersPerry(radius,spin/radius,G5).thermodynamics()
        plus,minus = evaluate(step),evaluate(-step)
        d = {k:(plus[k]-minus[k])/(2*step) for k in th}
        rhs = th['T_H']*d['S']+th['Omega_H']*(d['J1']+d['J2'])
        assert abs(d['M']-rhs)/max(1,abs(d['M'])) < 2e-8


def test_asymptotic_charges_from_sampled_fields():
    bh = MyersPerry(1.2,-0.3,G5=1.8)
    r = bh.r_h*1000
    v,th = bh.functions(r),bh.thermodynamics()
    U = (v['b']-1)*r**2
    W = v['w']*r**4
    assert -3*np.pi*U/(8*bh.G5) == pytest.approx(th['M'],rel=2e-9)
    assert np.pi*W/(4*bh.G5) == pytest.approx(th['J1'],rel=2e-9)
    coefficients = bh.asymptotic_coefficients()
    assert U == pytest.approx(coefficients['U'],rel=2e-9)
    assert W == pytest.approx(coefficients['W'],rel=2e-9)
    assert coefficients['V'] == pytest.approx(-coefficients['W']**2/coefficients['U'])
