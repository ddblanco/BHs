"""Field equations for extremal black holes in the gauge of arXiv:2303.12471,
generated as NumPy code.

Their ansatz (sec. 3.1), with a = 1 and P_i = exp(2 F_i):
    f0 = b = P1 r^4/(u^2+1),  f1 = 1/f = P1 u/r^2,  f2 = g = P2 u,
    f3 = h = P2 P3 u (1 + 1/u^2),  w = sqrt(2)/(u^2+1) + W,  u = r^2 + 1.
The gauge is b*f = r^6/(u (u^2+1)): fixed, independent of the unknowns. This
is what excludes nearby non-extremal solutions from the representation (they
would need a singular change of radial coordinate), unlike the g = r^2 gauge,
where b and f can both carry the same 1/x factor.

Radial coordinate xi = r^2/(1+r^2) in [0, 1]: xi = 0 is the degenerate horizon
(r = 0), xi = 1 infinity. The reduced Lagrangian has the same form in any
radial coordinate when f is read as g^{xi xi}, so derive.py's equations are
used with r -> xi and
    b = P1 xi^2/(1+(1-xi)^2),  f = 4 xi^2 (1-xi)^3 / P1,  g = P2/(1-xi),
    h = P2 P3 (1+(1-xi)^2)/(1-xi),  w = sqrt(2)(1-xi)^2/(1+(1-xi)^2) + W.
At alpha = 0, P1 = P2 = P3 = 1, W = 0 is extremal Myers-Perry (checked).

Equations collocated: the variations of the gauge-fixed action,
    E_P1 = (b E_b - f E_f)/P1,  E_P2 = (g E_g + h E_h)/P2,  E_P3 = h E_h/P3,
    E_W = E_w ;
the constraint E_f is kept as a diagnostic (it must vanish on solutions with
a regular horizon). Numerators are divided by the largest powers of xi and
(1-xi) that divide them for generic jets.

Run: python kkr_equations.py p   (radial coordinate s, xi = s^p)
Output: _kkr_generated_p{p}.py
"""
import inspect
import pickle
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
XI = sp.Symbol('xi')
ALPHA = sp.Symbol('alpha')
NAMES = ('P1', 'P1x', 'P1xx', 'P2', 'P2x', 'P2xx', 'P3', 'P3x', 'P3xx', 'W', 'Wx', 'Wxx')
JET = sp.symbols(' '.join(NAMES))
J = dict(zip(NAMES, JET))


def load():
    with open(HERE/'equations.pkl', 'rb') as handle:
        return pickle.load(handle)['eq']


def ansatz(power):
    """The radial coordinate is s (the symbol XI). power = p: xi = s^p,
    p = 1 is xi = r^2/(1+r^2) itself. power = 0 means s = r/(1+r), the r of
    arXiv:2303.12471 compactified: integer powers of r are smooth at both
    ends in s. f is always g^{ss}."""
    P1, P2, P3, W = (sp.Function(n)(XI) for n in ('P1', 'P2', 'P3', 'W'))
    if power == 0:
        r = XI/(1-XI)
        u = r**2+1
        phys = dict(b=P1*r**4/(u**2+1), f=(r**2/u)*(1-XI)**4/P1, g=P2*u,
                    h=P2*P3*u*(1+1/u**2), w=sp.sqrt(2)/(u**2+1)+W)
        phys = {k: sp.cancel(sp.together(v)) for k, v in phys.items()}
        return (P1, P2, P3, W), phys
    xi = XI**power
    q = 1+(1-xi)**2
    phys = dict(b=P1*xi**2/q, f=4*XI**2*(1-xi)**3/(power**2*P1), g=P2/(1-xi),
                h=P2*P3*q/(1-xi), w=sp.sqrt(2)*(1-xi)**2/q+W)
    return (P1, P2, P3, W), phys


def to_jets(expr, funcs):
    subs = {}
    for name, fn in zip(('P1', 'P2', 'P3', 'W'), funcs):
        subs[sp.diff(fn, XI, 2)] = J[name+'xx']
    for name, fn in zip(('P1', 'P2', 'P3', 'W'), funcs):
        subs[sp.diff(fn, XI)] = J[name+'x']
    for name, fn in zip(('P1', 'P2', 'P3', 'W'), funcs):
        subs[fn] = J[name]
    return expr.subs(subs, simultaneous=True)


def strip(num):
    """Divide out the largest powers of xi and (1 - xi) dividing num."""
    poly = sp.Poly(num, XI)
    low = min(m[0] for m in poly.monoms())
    num = sp.expand(sp.cancel(num/XI**low))
    Z = sp.Symbol('zeta')
    in_z = sp.Poly(sp.expand(num.subs(XI, 1-Z)), Z)
    low_z = min(m[0] for m in in_z.monoms())
    num = sp.expand(sp.cancel(sp.expand(num.subs(XI, 1-Z))/Z**low_z).subs(Z, 1-XI))
    return num, (low, low_z)


def main(power=1):
    eq = load()
    r = eq['r']
    funcs, phys = ansatz(power)
    # normalise while b, f, h are positive functions (square roots cancel)
    norm = {}
    for name in ('b', 'f', 'g', 'h', 'w'):
        e = sp.expand(eq['E'][name]/sp.sqrt(eq['b']*eq['h']/eq['f']))
        assert not any(not p.exp.is_integer for p in e.atoms(sp.Pow)), name
        norm[name] = e
    # combinations of the gauge-fixed action, at the level of b,f,g,h,w
    b, f, g, h = (eq[k] for k in 'bfgh')
    P1, P2, P3, W = funcs
    combos = {
        'P1': b*norm['b']-f*norm['f'],         # times 1/P1, applied after substitution
        'P2': g*norm['g']+h*norm['h'],
        'P3': h*norm['h'],
        'W': norm['w'],
        'C': norm['f'],                         # constraint, diagnostic
    }
    subs = {}
    for k in 'bfghw':
        subs[sp.diff(eq[k], r, 2)] = sp.diff(phys[k], XI, 2)
    for k in 'bfghw':
        subs[sp.diff(eq[k], r)] = sp.diff(phys[k], XI)
    for k in 'bfghw':
        subs[eq[k]] = phys[k]
    residuals, powers, middle = {}, {}, {}
    for name, e in combos.items():
        e = e.subs(subs, simultaneous=True).subs(r, XI)
        e = to_jets(e, funcs).subs(eq['alpha'], ALPHA)
        unexpanded = e
        e = sp.together(e)
        num, den = sp.fraction(e)
        num = sp.expand(num)
        # sqrt(2) from w0 is a number; no fractional power of a variable may remain
        assert not any(not p.exp.is_integer and not p.base.is_number
                       for p in num.atoms(sp.Pow)), name
        num, powers[name] = strip(num)
        residuals[name] = num
        # the same function, unexpanded: e * den / (xi^k (1-xi)^m), for the
        # middle of the interval where both expanded forms lose digits
        k_low, m_low = powers[name]
        middle[name] = unexpanded*den/(XI**k_low*(1-XI)**m_low)
        print(f'E_{name}: stripped xi^{powers[name][0]} (1-xi)^{powers[name][1]}, '
              f'{len(sp.Add.make_args(num))} terms')
    # extremal Myers-Perry at alpha = 0
    mp = {J['P1']: 1, J['P2']: 1, J['P3']: 1, J['W']: 0}
    for n in NAMES:
        mp.setdefault(J[n], 0)
    for name, e in residuals.items():
        value = sp.expand(e.subs(mp).subs(ALPHA, 0))
        assert value == 0, (name, value)
    print('extremal Myers-Perry (P = 1, W = 0) solves all residuals at alpha = 0: OK')
    # Two expanded forms of each residual: in powers of xi (used for xi <= 1/2)
    # and in powers of zeta = 1 - xi (used for xi > 1/2). A single expansion in
    # xi loses ~7 digits to binomial cancellation next to infinity.
    Z = sp.Symbol('zeta')
    args = (XI, Z, *JET, ALPHA)
    code = ['"""Generated by kkr_equations.py; do not edit."""', 'import numpy as np',
            'from numpy import sqrt', '']
    sig = ', '.join(str(a) for a in args)
    for name, e in residuals.items():
        e_z = sp.expand(e.subs(XI, 1-Z))
        e_m = middle[name]
        for tag, form in (('xi', e), ('zeta', e_z), ('mid', e_m)):
            # long sums overflow the compiler's recursion limit: emit chunks
            terms = sp.Add.make_args(form)
            parts = []
            for c, start in enumerate(range(0, len(terms), 150)):
                chunk = sp.Add(*terms[start:start+150])
                fn = sp.lambdify(args, chunk, modules='numpy', cse=True)
                part = f'_E_{name}_{tag}_{c}'
                code.append(inspect.getsource(fn).replace('def _lambdifygenerated', f'def {part}'))
                parts.append(part)
            code.append(f'def E_{name}_{tag}({sig}):')
            code.append('    return ' + ' + '.join(f'{part}({sig})' for part in parts))
            code.append('')
        code.append(f'def E_{name}({sig}):')
        code.append(f'    with np.errstate(all="ignore"):')
        code.append(f'        mid = E_{name}_mid({sig})')
        code.append(f'    out = np.where(xi <= 0.5, E_{name}_xi({sig}), E_{name}_zeta({sig}))')
        code.append(f'    inner = (xi > 0.12) & (xi < 0.88)')
        code.append(f'    return np.where(inner, mid, out)')
        code.append('')
    code.append(f'def residual({sig}):')
    code.append(f'    return (E_P1({sig}), E_P2({sig}), E_P3({sig}), E_W({sig}))')
    code.append('')
    code.append(f'def constraint({sig}):')
    code.append(f'    return E_C({sig})')
    code.append(f'POWERS = {powers!r}')
    code.append(f'POWER = {power}')
    (HERE/f'_kkr_generated_p{power}.py').write_text('\n'.join(code)+'\n')
    with open(HERE/f'kkr_residuals_p{power}.pkl', 'wb') as handle:
        pickle.dump(dict(residuals=residuals, powers=powers), handle)
    print(f'wrote _kkr_generated_p{power}.py')


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
