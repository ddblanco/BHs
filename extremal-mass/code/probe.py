"""Exact arithmetic research probe. No claim of a finite-coupling solution.

Transcribed source: local 2009.00015 PDF, p.17 eq.(58); web v3 eq.(61).
The three additional terms in the default output are DERIVED HERE; the
literal additive prescription fails. certificates.py preserves that control.
Run with work/analytic_metric/.venv/Scripts/python -u work/analytic_metric/probe.py
"""
from pathlib import Path
import sys
import re
import json
import platform
import hashlib
import time
import sympy as s

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
OUT = Path(__file__).resolve().parent
r, k = s.symbols('r k', positive=True)
alpha = s.Symbol('alpha', real=True)


def paper_profiles(corrected=True):
    """r_+^0=1, r_-^0=k, 0<k<1; alpha=alpha_GB; fixed ADM M,J.

    Reverse both paper azimuths: b=h_p/W_p, h=r^2 W_p,
    w=omega_p/2. Thus positive k means positive internal angular velocity.
    The two-parameter MP family is recovered by restoring length scale.
    r_+^0 is the UNPERTURBED horizon; the corrected horizon generally moves.
    """
    mu, nu, d = (1+k**2)/2, k, (1-k**2)/2
    L = s.log(1-k**2/r**2)
    f0 = 1-2*mu/r**2+nu**2/r**4
    W0 = 1+nu**2/r**4
    om0 = 2*s.sqrt(2*mu)*nu/(r**4+nu**2)
    dh = (64*mu*(2*mu**2+nu**2)*(mu-d)/(3*nu**4*r**2)
          -32*mu*(8*mu**2-6*mu*d+nu**2)/(3*nu**2*r**4)
          +8*(9*mu**2+4*nu**2)/(3*r**6)+32*mu*nu**2/(3*r**8)
          -16*nu**4/(3*r**10)
          -64*mu/(3*nu**4*r**4)*(2*mu*r**2*(mu**2+2*nu**2)
          -(2*mu**2+nu**2)*(nu**2+r**4))*L)
    df = (64*mu**2*(mu-d)/nu**4
          -32*mu**2*(2*mu**2-2*mu*d+nu**2)/(nu**4*r**2)
          +64*mu**3/(3*nu**2*r**4)+40*mu**2/(3*r**6)
          +64*mu*nu**2/(3*r**8)-16*nu**4/r**10
          +(64*mu**2*r**2/nu**4-128*mu**3/nu**4
          +64*mu**2*(2*mu**2+nu**2)/(3*nu**4*r**2))*L)
    dW = (-32*mu**2*(mu-d)/nu**4+16*mu**2/(nu**2*r**2)
          +4*(16*mu**4+15*mu**2*nu**2-6*nu**4
          -8*mu*d*(2*mu**2+nu**2))/(3*mu*nu**2*r**4)
          -16*(mu**2+3*nu**2)/(3*r**6)-16*mu*nu**2/(3*r**8)
          +16*nu**4/(3*r**10)
          -32*mu*(mu*r**2/nu**4-mu**2/nu**4-mu/(nu**2*r**2)
          +(mu**2+2*nu**2)/(3*nu**2*r**4))*L)
    dom = (16*s.sqrt(2*mu)*(-6*mu*r**6*(r**2-mu)*(mu-d)
           +2*nu**6-2*mu*nu**4*r**2+mu*nu**2*r**4*(3*r**2-2*mu))
           /(3*nu**3*r**6*(nu**2+r**4))
           -32*mu*s.sqrt(2*mu)*(2*mu**2+nu**2+3*r**4-6*mu*r**2)
           /(3*nu**3*(nu**2+r**4))*L)
    base = [f0/W0, f0, r**2*W0, om0/2]
    delta = [dh/W0-f0*dW/W0**2, df, r**2*dW, dom/2]
    if corrected:
        # Derived here, NOT the literal additive prescription of printed eq.(57).
        # Equivalently promote W0 to W in f0=W0-2mu/r^2 and in omega0.
        # All five linearized equations verify these three additions exactly.
        delta[0] += dW/W0
        delta[1] += dW
        delta[3] -= (om0/2)*dW/W0
    return [s.factor(v) for v in base], [s.factor(v) for v in delta]


def raw_equations():
    """Use generated rational equations, replacing literal Python divisions.

    No approximation: 1/4 and 1/2 in generated source must stay rational.
    Denominators are cleared only in the regular open exterior.
    """
    path = ROOT/'src/rotating_bh/_egb_rotating_equations_generated.py'
    source = path.read_text(encoding='utf-8')
    source = re.sub(r'\((\d+)/(\d+)\)', r'Rational(\1,\2)', source)
    namespace = {'Rational': s.Rational}
    exec(compile(source, str(path), 'exec'), namespace)
    jets = s.symbols('b f h w bp fp hp wp bpp fpp hpp wpp')
    expressions = namespace['equations'](r, jets[:4], jets[4:8], jets[8:], alpha)
    assert not any(e.atoms(s.Float) for e in expressions)
    return jets, [s.fraction(s.factor(e))[0] for e in expressions]


def jet(profiles):
    return profiles+[s.diff(v, r) for v in profiles]+[s.diff(v,r,2) for v in profiles]


def zero(expr):
    return s.cancel(s.expand(expr)) == 0


def main():
    start = time.perf_counter()
    print('Python', platform.python_version(), 'SymPy', s.__version__, flush=True)
    base, delta = paper_profiles()
    jets, equations = raw_equations()
    background = dict(zip(jets, jet(base)))
    variation = jet(delta)
    for name, eq in zip('bfghw', equations):
        e0 = eq.subs(alpha, 0)
        assert zero(e0.xreplace(background)), name+' MP'
        print('PASS exact E_'+name+' MP', flush=True)
        linear = s.cancel(s.diff(eq, alpha).xreplace(background))
        for v, dv in zip(jets, variation):
            coefficient = s.cancel(s.diff(e0,v).xreplace(background))
            if coefficient != 0:
                linear += s.cancel(coefficient*dv)
        reduced = s.cancel(linear)
        assert reduced == 0, name+' O(alpha): '+str(reduced)[:1000]
        print('PASS exact E_'+name+': coefficients alpha^0=alpha^1=0', flush=True)
    print('elapsed_seconds', round(time.perf_counter()-start, 2), flush=True)


if __name__ == '__main__':
    main()
