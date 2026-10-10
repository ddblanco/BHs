"""Exact first-order extremal exterior, with fixed physical J.

Starts from the previously checked Ma-Li-Lu first-order exterior in the
project's conventions. Adds an MP parameter variation to impose dM/dalpha=pi,
dJ/dalpha=0. This is an exterior perturbation, not a uniform horizon theorem.
"""
from pathlib import Path
import datetime
import hashlib
import json
import platform
import sys
import time

import sympy as s

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe import paper_profiles, raw_equations, jet, r, k, alpha


def extremal_profiles():
    mu, nu = s.symbols('mu nu', positive=True)
    f = 1-2*mu/r**2+nu**2/r**4
    W = 1+nu**2/r**4
    mp = [f/W, f, r**2*W, s.sqrt(2*mu)*nu/(r**4+nu**2)]
    base, fixed_charges = paper_profiles()
    b0 = [s.factor(v.subs(k, 1)) for v in base]
    delta = []
    for old, field in zip(fixed_charges, mp):
        shift = (s.Rational(4, 3)*s.diff(field, mu)
                 -s.Rational(2, 3)*s.diff(field, nu)).subs({mu: 1, nu: 1})
        delta.append(s.factor(old.subs(k, 1)+shift))
    assert s.diff(3*s.pi*mu/4, mu)*s.Rational(4, 3) == s.pi
    assert s.simplify((s.Rational(4, 3)*s.diff(s.pi*s.sqrt(2*mu)*nu/4, mu)
                      -s.Rational(2, 3)*s.diff(s.pi*s.sqrt(2*mu)*nu/4, nu)).subs({mu: 1, nu: 1})) == 0
    return b0, delta


def main():
    start = time.monotonic()
    output = []
    def say(line):
        print(line, flush=True)
        output.append(line)
    say(f'ENVIRONMENT Python {platform.python_version()} SymPy {s.__version__}')
    base, delta = extremal_profiles()
    jj, ee = raw_equations()
    bg = dict(zip(jj, jet(base)))
    variation = jet(delta)
    for name, eq in zip('bfghw', ee):
        e0 = eq.subs(alpha, 0)
        assert s.cancel(e0.xreplace(bg)) == 0
        linear = s.cancel(s.diff(eq, alpha).xreplace(bg))
        for v, dv in zip(jj, variation):
            coeff = s.cancel(s.diff(e0, v).xreplace(bg))
            if coeff != 0:
                linear += coeff*dv
        assert s.cancel(s.expand(linear)) == 0, name
        say(f'PASS exact extremal exterior E_{name} through first order')
    L = s.Symbol('L')
    log = s.log(1-r**-2)
    coeffs = []
    for field in delta:
        p = s.Poly(s.cancel(field.xreplace({log: L})), L)
        assert p.degree() <= 1
        coeffs.append([s.factor(p.nth(0)), s.factor(p.nth(1))])
    # Determine common displacement of the double zero from first derivatives.
    horizon_shift = []
    for i in (0, 1):
        zeroth = s.limit(delta[i], r, 1, dir='+')
        first = s.limit(s.diff(delta[i], r), r, 1, dir='+')
        assert zeroth == 0
        horizon_shift.append(s.simplify(-first/s.diff(base[i], r, 2).subs(r, 1)))
    assert horizon_shift[0] == horizon_shift[1]
    rh1 = horizon_shift[0]
    omega1 = s.simplify(s.limit(delta[3], r, 1, dir='+')+rh1*s.diff(base[3], r).subs(r, 1))
    assert omega1 == 0
    say(f'PASS common displaced double root to first order: r_H=1+({rh1})*alpha+O(alpha^2)')
    say('PASS fixed-J Omega_H has zero linear correction; dM/dalpha=pi, dJ/dalpha=0')
    # A finite exterior solution need not have a Taylor expansion uniform at r_H.
    log_orders = []
    for name, (rational, logcoef) in zip('bfhw', coeffs):
        lead = s.factor(logcoef).as_leading_term(r) if logcoef == 0 else None
        u = s.Symbol('u', positive=True)
        lead = s.simplify(logcoef.subs(r, 1+u)).as_leading_term(u)
        log_orders.append(dict(field=name, leading=str(lead)))
        say(f'LOG coefficient near r=1: {name}: {lead}')
    receipt = dict(date=datetime.datetime.now().isoformat(),
                   command=[sys.executable, *sys.argv], python=platform.python_version(),
                   sympy=s.__version__, output=output,
                   base=[str(v) for v in base], delta=[str(v) for v in delta],
                   decomposition=[dict(rational=str(a), logarithm=str(b)) for a,b in coeffs],
                   horizon_shift=str(rh1), omega_linear=str(omega1), log_orders=log_orders,
                   seconds=time.monotonic()-start,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   not_checked=['second-order double-root condition', 'uniform extremal/horizon expansion',
                                'full finite-alpha metric', 'second-order mass coefficient'])
    (HERE/'extremal_perturbation.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
