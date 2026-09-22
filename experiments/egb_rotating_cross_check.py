"""Reproduce the adaptive/spectral cross-check at omega_H=.3, alpha_GB=.1.

Run from project/ with its locked environment and PYTHONPATH=src.
No old result is regenerated; only the new result and its manifest entry
are written. The spectral solution is an initial guess, never boundary data.
"""
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
for name in ('TEMP', 'TMP'):
    (ROOT/'.cache/tmp').mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(ROOT/'.cache/tmp')

import numpy as np
import scipy
import sympy
from scipy.integrate import quad

from rotating_bh.egb_rotating_adaptive import solve as adaptive, SolverFailure
from rotating_bh.egb_rotating_angular import first_integral
from rotating_bh.egb_rotating_bvp import solve as spectral
from rotating_bh.egb_rotating_validation import diagnose, tensor_residual
from rotating_bh._egb_rotating_horizon_generated import horizon
from rotating_bh.myers_perry import MyersPerry
from rotating_bh.provenance import validate_manifest, validate_record

PROFILE_TOL = 1e-5
TENSOR_TOL = 1e-5
GRID = np.linspace(.02, .98, 151)


def numerical_completion(row):
    return bool(row.get('converged', False)
                and 0 <= row['boundary_residual'] < 1e-8
                and 0 <= row['collocation_residual'] <= row['tolerance'])


def mp_difference(solution):
    z = 1-GRID
    closed = MyersPerry(r_h=1., omega_h=.3).functions(1/z)
    amplitudes = np.asarray([closed['b']/(1-z*z), closed['f']/(1-z*z),
                              (closed['h']*z*z-1)/z**4, closed['w']/z**4])
    return float(np.max(np.abs(solution.evaluate(GRID)-amplitudes)))


def infinity_f_defect(solution):
    eps = solution.cutoff
    x = 1-eps
    return float(abs(solution.evaluate(x)[1]+eps*solution.evaluate(x, 1)[1]
                     +eps**2*solution.evaluate(x, 2)[1]/2-1))


def angular_diagnostic(solution):
    values, first = solution.evaluate(GRID), solution.evaluate(GRID, 1)
    current = first_integral(GRID, values, first, solution.alpha_gb)
    z = 1-GRID
    C = 1+4*solution.alpha_gb*z*z*(1-(1-z*z)*values[1]-3*z**4*values[2])
    return dict(first_integral_spread=float(np.ptp(current)),
                first_integral_at_midpoint=float(first_integral(.5, solution.evaluate(.5),
                                                                solution.evaluate(.5, 1), solution.alpha_gb)),
                coefficient_min_sampled=float(np.min(C)))


def run():
    seed = None
    spectral_runs = []
    for alpha in [0., .02, .05, .1]:
        seed = spectral(.3, alpha, resolution=24, tol=1e-11, previous=seed)
        diagnostic = diagnose(seed, alpha, count=31)
        row = dict(alpha_gb=alpha, iterations=seed.iterations,
                   boundary_residual=seed.boundary_residual,
                   angular=angular_diagnostic(seed), **diagnostic)
        spectral_runs.append(row)
        print('spectral', row, flush=True)
        if diagnostic['max_tensor_residual'] >= 1e-6:
            raise RuntimeError('spectral seed failed the independent tensor gate')
        if alpha == 0:
            vacuum = seed
    mp_adaptive = adaptive(.3, 0., previous=vacuum)
    mp = dict(spectral_difference=mp_difference(vacuum),
              adaptive_difference=mp_difference(mp_adaptive),
              **diagnose(mp_adaptive, 0., count=31))
    print('myers_perry', mp, flush=True)

    class PerturbedSeed:
        def evaluate(self, x, derivative=0):
            return 1.001*seed.evaluate(x, derivative)

    runs, solutions = [], []
    cases = [(.006, 1e-7, False), (.003, 1e-6, False),
             (.003, 1e-7, False), (.0015, 1e-7, False),
             (.003, 1e-8, False), (.003, 1e-7, True)]
    for cutoff, tol, perturb in cases:
        row = dict(cutoff=cutoff, tolerance=tol, perturbed_seed=perturb)
        try:
            sol = adaptive(.3, .1, previous=PerturbedSeed() if perturb else seed,
                           cutoff=cutoff, tol=tol)
        except SolverFailure as error:
            row.update(converged=False, reason=str(error))
            solutions.append(None)
        else:
            errors = np.max(np.abs(sol.evaluate(GRID)-seed.evaluate(GRID)), axis=1)
            row.update(converged=True, nodes=int(sol.nodes.size), iterations=sol.iterations,
                       boundary_residual=sol.boundary_residual,
                       collocation_residual=sol.collocation_residual,
                       amplitude_errors=errors.tolist(), max_difference=float(np.max(errors)),
                       infinity_f_defect=infinity_f_defect(sol),
                       angular=angular_diagnostic(sol),
                       **diagnose(sol, .1, count=51))
            solutions.append(sol)
        runs.append(row)
        print('adaptive', row, flush=True)

    nominal = solutions[2]
    negative = tensor_residual(nominal, .5, .15) if nominal is not None else None
    seed_difference = (float(np.max(np.abs(nominal.evaluate(GRID)-solutions[-1].evaluate(GRID))))
                       if nominal is not None and solutions[-1] is not None else None)
    resource_failure = None
    try:
        adaptive(.3, .1, previous=seed, resolution=8, max_nodes=8, tol=1e-10)
    except SolverFailure as error:
        resource_failure = str(error)
    horizon_rows = horizon(*np.r_[seed.evaluate(0.), seed.evaluate(0., 1)[[0, 2, 3]]],
                           seed.evaluate(0., 1)[1], seed.evaluate(0., 2)[3], .1)

    def inverse_angular_coefficient(x):
        B, F, H, W = seed.evaluate(x)
        z = 1-x
        A = 1+z**4*H
        C = 1+4*.1*z*z*(1-(1-z*z)*F-3*z**4*H)
        return z**3/(np.sqrt(F/B)*A**1.5*C)

    integral, quadrature_error = quad(inverse_angular_coefficient, 0., 1., epsabs=1e-12,
                                     epsrel=1e-12)
    angular_velocity = -spectral_runs[-1]['angular']['first_integral_at_midpoint']*integral
    angular = dict(spectral_horizon_relations=list(horizon_rows),
                    inverse_coefficient_integral=integral, quadrature_error_estimate=quadrature_error,
                    omega_from_first_integral=angular_velocity,
                    omega_error=abs(angular_velocity-.3))
    checks = [
        dict(name='numerical_completion', passed=all(numerical_completion(row) for row in runs)),
        dict(name='myers_perry', passed=mp['adaptive_difference'] < PROFILE_TOL
             and mp['spectral_difference'] < PROFILE_TOL and mp['max_tensor_residual'] < TENSOR_TOL),
        dict(name='methods_agree', passed=all(row.get('converged', False)
             and row['max_difference'] < PROFILE_TOL for row in runs)),
        dict(name='independent_tensor', passed=all(row.get('converged', False)
             and row['max_tensor_residual'] < TENSOR_TOL for row in runs)),
        dict(name='unimposed_infinity_f', passed=all(row.get('converged', False)
             and row['infinity_f_defect'] < PROFILE_TOL for row in runs)),
        dict(name='cutoff_refinement', passed=all(row.get('converged', False) for row in runs)
             and runs[3]['max_difference'] < runs[2]['max_difference'] < runs[0]['max_difference']
             and runs[3]['max_tensor_residual'] < runs[2]['max_tensor_residual'] < runs[0]['max_tensor_residual']),
        dict(name='seed_perturbation', passed=seed_difference is not None and seed_difference < 1e-7),
        dict(name='wrong_alpha_detected', passed=negative is not None and negative > 1e-4),
        dict(name='resource_exhaustion_reported', passed=resource_failure is not None),
        dict(name='angular_first_integral', passed=all(row.get('converged', False)
             and row['angular']['first_integral_spread'] < 1e-5
             and row['angular']['coefficient_min_sampled'] > 0 for row in runs)
             and abs(horizon_rows[3]) < 1e-7 and angular['omega_error'] < 1e-8),
    ]
    return dict(parameters=dict(omega_h=.3, alpha_gb=.1, r_h=1., spectral_resolution=24,
                                spectral_tolerance=1e-11, adaptive_initial_nodes=81,
                                adaptive_max_nodes=4000, profile_tolerance=PROFILE_TOL,
                                tensor_tolerance=TENSOR_TOL, profile_interval=[.02, .98],
                                profile_samples=151, tensor_samples=51),
                spectral_runs=spectral_runs, myers_perry=mp, adaptive_runs=runs,
                seed_perturbation_difference=seed_difference, negative_control_residual=negative,
                resource_failure=resource_failure, angular=angular,
                checks=checks)


def main():
    manifest = ROOT/'artifacts/manifest.json'
    validate_manifest(manifest)
    result = run()
    sources = ['src/rotating_bh/'+name+'.py' for name in
               ('egb_rotating_bvp', 'egb_rotating_adaptive', 'egb_rotating_angular', 'egb_rotating_validation',
                '_egb_rotating_compact_generated', '_egb_rotating_horizon_generated',
                'myers_perry', 'einstein', 'provenance')]
    sources += ['experiments/egb_rotating_cross_check.py', 'environment/requirements-lock.txt']
    result['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    result['versions'] = dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, sympy=sympy.__version__)
    passed = all(check['passed'] for check in result['checks'])
    path = ROOT/'results/egb-rotating-cross-check.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    record = dict(id='egb-rotating-cross-check', kind='result',
                   path=path.relative_to(ROOT).as_posix(),
                   command='python experiments/egb_rotating_cross_check.py (PYTHONPATH=src)',
                   commit='working-tree', inputs=sources, parameters=result['parameters'],
                   environment='environment/requirements-lock.txt', agent='codex',
                   prompt_refs=['prompts/prompt-log.md', 'docs/2026-09-09-handoff-codex-hito-4b.md'],
                   decisions=['Adaptive Lobatto collocation initialized from spectral continuation; '
                              'second-order Taylor boundary transport independent of the seed',
                              'Full tensor on represented derivatives, cutoff/tolerance sweeps, '
                              'perturbed seed, closed Myers-Perry and wrong-alpha controls',
                              'Only omega_h=.3, alpha_gb=.1; no Hito 4C or merge decision'],
                   checks=result['checks'], status='verified' if passed else 'rejected',
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    errors = validate_record(record)
    if errors:
        raise ValueError('; '.join(errors))
    records = json.loads(manifest.read_text(encoding='utf-8'))
    records = [r for r in records if r['id'] != record['id']]+[record]
    manifest.write_text(json.dumps(records, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    validate_manifest(manifest)
    print(json.dumps(dict(status=record['status'], checks=result['checks'])), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
