"""Deterministic, sampled validation of the analytic GR reference."""

from functools import lru_cache

import numpy as np
import sympy as sp

from rotating_bh.einstein import compile_metric, curvature, metric_from_ansatz
from rotating_bh.myers_perry import MyersPerry

SPINS = (0.0, -0.33, 0.33, 0.68)
SCALES = (0.7, 1.0, 2.0)
RADII = (1.05, 1.3, 2.0, 5.0, 20.0)
ANGLES = (0.2, 0.7, 1.2)
TOLERANCES = {'einstein': 1e-8, 'smarr': 1e-13, 'first_law': 2e-8,
              'horizon': 1e-13, 'asymptotic': 2e-8}


@lru_cache(maxsize=1)
def metric_evaluator():
    """Compile MP in mass/rotation parameters, independently of functions().

    mu = -U = 2*M_paper (not physical mass), a=r_h²*Omega_H.
    epsilon multiplies w by 1+epsilon for the nonvacuum negative control.
    """
    t,r,theta,phi1,phi2 = sp.symbols('t r theta phi1 phi2',real=True)
    mu,a,epsilon = sp.symbols('mu a epsilon',real=True)
    h = r**2+mu*a**2/r**2
    f = 1-mu/r**2+mu*a**2/r**4
    b = r**2*f/h
    w = (1+epsilon)*mu*a/(r**4+mu*a**2)
    metric = metric_from_ansatz(r,theta,b,f,h,w)
    return compile_metric(metric,(t,r,theta,phi1,phi2),(mu,a,epsilon))


def scaled_residual(tensor,r,r_h):
    """Coordinate-component max; angular covectors scaled by r, then r_h².

    This is a dimensionless coordinate diagnostic, not an invariant norm or
    a rigorous bound. The chart axes and horizon are excluded from sampling.
    """
    scale = np.array([1,1,r,r,r],dtype=float)
    return float(np.max(np.abs(tensor/np.outer(scale,scale)))*r_h**2)


def first_law_errors(bh):
    """Centered finite differences in independent (r_h,q), at two step sizes."""
    q = bh.r_h*bh.omega_h
    th = bh.thermodynamics()
    errors = []
    for fraction in (2e-5,1e-5):
        for direction in ('r_h','q'):
            step = fraction*bh.r_h if direction == 'r_h' else fraction
            values = []
            for sign in (1,-1):
                radius = bh.r_h+sign*step if direction == 'r_h' else bh.r_h
                spin = q+sign*step if direction == 'q' else q
                values.append(MyersPerry(radius,spin/radius,bh.G5).thermodynamics())
            d = {key:(values[0][key]-values[1][key])/(2*step) for key in th}
            rhs = th['T_H']*d['S']+th['Omega_H']*(d['J1']+d['J2'])
            # Natural derivative scales avoid an arbitrary dimensionful floor.
            unit = th['M']/bh.r_h if direction == 'r_h' else th['M']
            errors.append({'direction':direction,'step':step,
                           'relative_error':float(abs(d['M']-rhs)/unit)})
    return errors


def run_benchmark():
    evaluate = metric_evaluator()
    rows = []
    for r_h in SCALES:
        for q in SPINS:
            bh = MyersPerry(r_h,q/r_h,G5=1.0)
            th = bh.thermodynamics()
            mu,a = r_h**2/(1-q*q),r_h*q
            samples = []
            for rho in RADII:
                for theta in ANGLES:
                    r = r_h*rho
                    tensors = curvature(*evaluate(0,r,theta,0,0,mu,a,0))
                    samples.append({'rho':rho,'theta':theta,
                        'einstein':scaled_residual(tensors['einstein'],r,r_h),
                        'ricci':scaled_residual(tensors['ricci'],r,r_h),
                        'scalar':float(abs(tensors['scalar'])*r_h**2)})
            v = bh.functions(r_h)
            horizon_error = float(max(abs(v['b']),abs(v['f']),
                abs((v['w']-th['Omega_H'])*r_h)))
            # chi²=-b+h*(w-Omega_H)², evaluated without singular g_rr.
            horizon_null = float(abs(-v['b']+v['h']*(v['w']-th['Omega_H'])**2))
            asymptotic = []
            coefficients = bh.asymptotic_coefficients()
            for rho in (100.0,1000.0):
                r = rho*r_h
                far = bh.functions(r)
                mass = -3*np.pi*(far['b']-1)*r*r/(8*bh.G5)
                angular = np.pi*far['w']*r**4/(4*bh.G5)
                asymptotic.append({'rho':rho,'mass_relative_error':float(abs(mass/th['M']-1)),
                    'spin_scaled_error':float(abs(angular-th['J1'])/(th['M']*r_h)),
                    'flatness_error':float(max(abs(far['b']-1),abs(far['f']-1),
                        abs(far['h']/r**2-1),abs(far['w']*r_h)))})
            smarr = abs(2*th['M']/3-th['T_H']*th['S']-
                        th['Omega_H']*(th['J1']+th['J2']))/th['M']
            rows.append({'r_h':r_h,'q':q,'G5':bh.G5,'thermodynamics':th,
                'asymptotic_coefficients':coefficients,'asymptotic_samples':asymptotic,
                'horizon_error':horizon_error,'horizon_null_error':horizon_null,
                'smarr_error':float(smarr),'first_law':first_law_errors(bh),
                'einstein_residual':max(s['einstein'] for s in samples),
                'samples':samples})
    # Fixed nonzero spin, deliberate 1% change in frame dragging only.
    q,r = 0.33,1.3
    tensors = curvature(*evaluate(0,r,0.7,0,0,1/(1-q*q),q,0.01))
    negative = scaled_residual(tensors['einstein'],r,1.0)
    checks = [
        {'name':'all_25_einstein_components_below_tolerance',
         'passed':all(row['einstein_residual'] < TOLERANCES['einstein'] for row in rows)},
        {'name':'horizon_and_null_generator',
         'passed':all(max(row['horizon_error'],row['horizon_null_error']) < TOLERANCES['horizon'] for row in rows)},
        {'name':'smarr', 'passed':all(row['smarr_error'] < TOLERANCES['smarr'] for row in rows)},
        {'name':'first_law_two_directions_two_steps',
         'passed':all(e['relative_error'] < TOLERANCES['first_law'] for row in rows for e in row['first_law'])},
        {'name':'asymptotic_charges',
         'passed':all(max(s['mass_relative_error'],s['spin_scaled_error']) < TOLERANCES['asymptotic']
                      for row in rows for s in row['asymptotic_samples'])},
        {'name':'asymptotic_flatness_decay',
         'passed':all(row['asymptotic_samples'][1]['flatness_error'] <
                      row['asymptotic_samples'][0]['flatness_error']/90 for row in rows)},
        {'name':'perturbed_metric_detected', 'passed':negative > 1e-5},
    ]
    rho = np.linspace(1,6,251)
    profiles = []
    for q in (0.0,0.33,0.68):
        v = MyersPerry(1,q).functions(rho)
        profiles.append({'q':q,'rho':rho.tolist(),'b':v['b'].tolist(),
                         'f':v['f'].tolist(),'h_over_r2':(v['h']/rho**2).tolist(),
                         'r_h_w':v['w'].tolist()})
    return {'parameters':{'spins_q':SPINS,'scales_r_h':SCALES,'radii_rho':RADII,
                'angles_theta':ANGLES,'G5':1.0,'tolerances':TOLERANCES,
                'negative_control':{'r_h':1,'q':0.33,'rho':1.3,'theta':0.7,'w_fraction':0.01},
                'profile_r_h':1,'profile_rho_range':[1,6],'profile_points':251},
            'runs':rows,'negative_control_residual':negative,
            'profiles':profiles,'checks':checks}
