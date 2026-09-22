"""Catch sign/coefficient mistakes using full coordinate tensor contractions."""
import importlib.util

import numpy as np
import pytest
import sympy as sp

from rotating_bh.einstein import compile_metric, curvature, metric_from_ansatz


def test_radial_equations_exist():
    assert importlib.util.find_spec('rotating_bh.vacuum_equations') is not None


@pytest.mark.parametrize('radius', [1.2, 2.0, 4.0])
def test_reduced_variations_match_generic_einstein(radius):
    from rotating_bh.vacuum_equations import radial_equations
    t,r,theta,p1,p2 = sp.symbols('t r theta p1 p2', real=True)
    # Deliberately off-shell, including a nonconstant rotation profile.
    expressions = [1-sp.Rational(3,5)/r**2, 1-sp.Rational(2,5)/r**2,
                   r**2+sp.Rational(1,5)/r, sp.Rational(1,7)/r**3]
    metric = metric_from_ansatz(r,theta,*expressions)
    evaluate = compile_metric(metric,(t,r,theta,p1,p2))
    arrays = evaluate(0,radius,0.7,0,0)
    inv = np.linalg.inv(arrays[0])
    upper = inv @ curvature(*arrays)['einstein'] @ inv
    symbols = sp.symbols('b f g h w', positive=True)
    b,f,g,h,w = symbols
    generic = metric_from_ansatz(sp.sqrt(g),theta,b,f,h,w)
    vals = np.array([[float(sp.diff(e,r,k).subs(r,radius)) for e in expressions]
                     for k in range(3)])
    mapping = dict(zip(symbols,[vals[0,0],vals[0,1],radius**2,vals[0,2],vals[0,3]]))
    mapping[theta] = 0.7
    actual = radial_equations(radius,*vals)
    for key, symbol in zip(('b','f','g','h','w'),symbols):
        tangent = np.array(generic.diff(symbol).subs(mapping)).astype(float)
        expected = -radius**2*np.sum(upper*tangent)
        assert actual[key] == pytest.approx(expected,rel=2e-11,abs=2e-12)


@pytest.mark.parametrize('q',[0.0,-0.33,0.6])
def test_vacuum_and_negative_control(q):
    from rotating_bh.vacuum_equations import radial_equations
    r=sp.symbols('r',positive=True)
    mu=1/(1-q*q)
    f=1-mu/r**2+mu*q*q/r**4
    h=r**2+mu*q*q/r**2
    expressions=[r*r*f/h,f,h,mu*q/(r**4+mu*q*q)]
    perturbed=[]
    for radius in (1.05,1.3,2,5):
        vals=np.array([[float(sp.diff(e,r,k).subs(r,radius)) for e in expressions]
                       for k in range(3)])
        assert max(abs(v) for v in radial_equations(radius,*vals).values()) < 1e-8
        if q:
            vals[:,3] *= 1.01
            perturbed.append(max(abs(v) for v in radial_equations(radius,*vals).values()))
    if q:
        assert max(perturbed) > 1e-5
