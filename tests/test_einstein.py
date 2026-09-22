import numpy as np
import pytest
import sympy as sp

from rotating_bh.einstein import compile_metric, curvature, gauss_bonnet, metric_from_ansatz


t, r, theta, phi1, phi2 = sp.symbols('t r theta phi1 phi2', real=True)
coordinates = (t, r, theta, phi1, phi2)


@pytest.mark.parametrize('K', [0.0, 0.07, -0.04])
def test_constant_curvature_sign(K):
    k = sp.Symbol('K')
    metric = metric_from_ansatz(r, theta, 1-k*r**2, 1-k*r**2, r**2, 0)
    evaluate = compile_metric(metric, coordinates, (k,))
    for radius, angle in [(0.8, 0.3), (1.5, 0.9), (2.1, 1.2)]:
        g, dg, ddg = evaluate(0, radius, angle, 0, 0, K)
        result = curvature(g, dg, ddg)
        np.testing.assert_allclose(result['ricci'], 4*K*g, atol=2e-13)
        assert result['scalar'] == pytest.approx(20*K, abs=2e-13)
        np.testing.assert_allclose(result['einstein'], -6*K*g, atol=2e-13)


@pytest.mark.parametrize('K', [0.0, 0.07, -0.04])
def test_riemann_contracts_to_ricci(K):
    # Independent consistency check: R_{ij}=R^k_{ikj}, not assumed elsewhere.
    k = sp.Symbol('K')
    metric = metric_from_ansatz(r, theta, 1-k*r**2, 1-k*r**2, r**2, 0)
    evaluate = compile_metric(metric, coordinates, (k,))
    g, dg, ddg = evaluate(0, 1.3, 0.6, 0, 0, K)
    result = curvature(g, dg, ddg)
    contracted = np.einsum('rsrn->sn', result['riemann'])
    np.testing.assert_allclose(contracted, result['ricci'], atol=2e-13)


@pytest.mark.parametrize('K', [0.0, 0.07, -0.04])
def test_gauss_bonnet_constant_curvature(K):
    # D=5 maximally symmetric space: L_GB=120K^2, H_{mu nu}=-12K^2 g_{mu nu}.
    k = sp.Symbol('K')
    metric = metric_from_ansatz(r, theta, 1-k*r**2, 1-k*r**2, r**2, 0)
    evaluate = compile_metric(metric, coordinates, (k,))
    for radius, angle in [(0.8, 0.3), (1.5, 0.9), (2.1, 1.2)]:
        g, dg, ddg = evaluate(0, radius, angle, 0, 0, K)
        result = curvature(g, dg, ddg)
        gb = gauss_bonnet(g, result['inverse'], result['riemann'],
                           result['ricci'], result['scalar'])
        assert gb['lagrangian'] == pytest.approx(120*K**2, abs=2e-10)
        np.testing.assert_allclose(gb['H'], -12*K**2*g, atol=2e-10)


def test_gauss_bonnet_nonzero_for_non_constant_curvature_profile():
    # Regression guard against a silently-zero H (e.g. an index-contraction
    # bug that cancels everywhere). b and f vary independently here; a
    # single-function profile (b varying, f=1) turned out to give H=0 for
    # this ansatz and is not used as the guard for that reason.
    epsilon = 0.13
    metric = metric_from_ansatz(r, theta, (1+epsilon*r**2)**2, 1+2*epsilon*r**2, r**2, 0)
    g, dg, ddg = compile_metric(metric, coordinates)(0, 1.4, 0.6, 0, 0)
    result = curvature(g, dg, ddg)
    gb = gauss_bonnet(g, result['inverse'], result['riemann'],
                       result['ricci'], result['scalar'])
    assert np.max(np.abs(gb['H'])) > 1e-6


def test_gauss_bonnet_nonzero_on_schwarzschild_tangherlini():
    # The D=5 GR vacuum solution is Einstein=0 but does not solve EGB: H
    # must be nonzero here, otherwise alpha_GB would not deform it at all.
    rh = sp.Symbol('r_h', positive=True)
    f = 1-(rh/r)**2
    metric = metric_from_ansatz(r, theta, f, f, r**2, 0)
    evaluate = compile_metric(metric, coordinates, (rh,))
    for radius in (1.3, 2.0, 5.0):
        g, dg, ddg = evaluate(0, radius, 0.6, 0, 0, 1.0)
        result = curvature(g, dg, ddg)
        np.testing.assert_allclose(result['einstein'], 0, atol=1e-13)
        gb = gauss_bonnet(g, result['inverse'], result['riemann'],
                           result['ricci'], result['scalar'])
        assert np.max(np.abs(gb['H'])) > 1e-6


def test_ansatz_cross_terms():
    b, f, h, w = sp.symbols('b f h w')
    g = metric_from_ansatz(r, theta, b, f, h, w)
    assert g.shape == (5, 5)
    assert g == g.T
    assert g[0, 0] == -b+h*w**2
    assert g[0, 3] == -h*w*sp.sin(theta)**2
    assert g[0, 4] == -h*w*sp.cos(theta)**2
    assert g[3, 4] == -(r**2-h)*sp.sin(theta)**2*sp.cos(theta)**2


def test_derivative_layout_including_mixed_coordinate_derivatives():
    a = sp.Symbol('a')
    metric = sp.diag(-1, 1, 1, 1, 1)
    metric[0, 3] = metric[3, 0] = a*t*r**2*sp.sin(theta)
    evaluate = compile_metric(metric, coordinates, (a,))
    point = (0.4, 1.3, 0.7, 0.2, 0.1, 0.3)
    g, dg, ddg = evaluate(*point)
    substitution = dict(zip((*coordinates, a), point))
    assert dg.shape == (5, 5, 5)
    assert ddg.shape == (5, 5, 5, 5)
    np.testing.assert_allclose(g, np.array(metric.subs(substitution), dtype=float))
    for k in range(5):
        np.testing.assert_allclose(dg[k], np.array(metric.diff(coordinates[k]).subs(substitution), dtype=float))
        for l in range(5):
            expected = metric.diff(coordinates[k], coordinates[l]).subs(substitution)
            np.testing.assert_allclose(ddg[k, l], np.array(expected, dtype=float))


def test_nonvacuum_lapse_perturbation_is_detected():
    epsilon = 0.13
    metric = metric_from_ansatz(r, theta, (1+epsilon*r**2)**2, 1, r**2, 0)
    g, dg, ddg = compile_metric(metric, coordinates)(0, 1.4, 0.6, 0, 0)
    result = curvature(g, dg, ddg)
    # Static lapse N=1+epsilon*r^2 on flat four-space: R=-2 Laplacian(N)/N.
    assert result['scalar'] == pytest.approx(-16*epsilon/(1+epsilon*1.4**2))
    assert np.max(np.abs(result['einstein'])) > 0.1


def test_flat_metric_in_mixed_time_dependent_coordinates():
    # Nonlinear pullback of Cartesian Minkowski, with off-diagonal components
    # and mixed time/radial/angular derivatives independent of the BH ansatz.
    cartesian = sp.Matrix([t, r+t*theta/3, theta+r**2/5, phi1, phi2])
    jacobian = cartesian.jacobian(coordinates)
    metric = jacobian.T*sp.diag(-1, 1, 1, 1, 1)*jacobian
    evaluate = compile_metric(metric, coordinates)
    for point in [(0.2, 0.8, 0.5, 0, 0), (-0.5, 1.3, 0.9, 0.2, 0.3)]:
        result = curvature(*evaluate(*point))
        np.testing.assert_allclose(result['ricci'], 0, atol=2e-14)
        np.testing.assert_allclose(result['einstein'], 0, atol=2e-14)
        assert abs(result['scalar']) < 2e-14
