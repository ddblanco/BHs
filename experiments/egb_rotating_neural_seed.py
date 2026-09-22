"""Hito 5: does a neural seed reach couplings the conventional seeds cannot?

The comparison is designed so it can lose. The spectral solver converges from
its own trivial guess over much of the domain, so a network that only matches it
there proves nothing. The benchmark is therefore placed at high coupling, where
the trivial guess and the Myers-Perry anchor both exhaust their budget, and
where the Hito 4C sweep reached the solution only by continuing through many
steps in alpha.

Every seed is judged by the Hito 4 acceptance gates unchanged -- full tensor
residual below 1e-6 off the collocation nodes, boundary residual below 1e-8 --
so a seed that merely makes the solver's own iteration terminate does not count.
That distinction is not hypothetical: an earlier trained seed drove the Newton
to a point that satisfied its internal tolerance while the independent tensor
check read 8.35e+02.

Both architectures and every random seed are reported, not the ones that
worked. Sensitivity to those choices is the ROADMAP's "dependence on human
decisions", and it is the result most easily lost by reporting a best case.

Run with PYTHONPATH=src from project/.
"""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sympy as sp
from scipy.optimize import least_squares

from rotating_bh.egb_rotating_bvp import solve
from rotating_bh.egb_rotating_validation import diagnose
from rotating_bh.egb_rotating_observables import measure
from rotating_bh.neural_seed import (MyersPerryAnchor, Problem, Seed, TrivialSeed,
                                     check_gradient, initial_parameters, n_parameters)
from rotating_bh.neural_seed_validation import checks as neural_checks

ROOT = Path(__file__).resolve().parents[1]

OMEGA_H = .33
COUPLINGS = [.1, .5, .75]
ARCHITECTURES = [(1, 12, 12, 4), (1, 24, 24, 4)]
SEEDS = [0, 1, 2, 3]
BUDGET = 15                      # Newton iterations allowed to every seed alike
TRAINING_EVALUATIONS = 300
TENSOR_GATE, BOUNDARY_GATE = 1e-6, 1e-8


def refine(omega_h, alpha_gb, previous, budget=BUDGET):
    """Hand a seed to the untouched solver and judge it by the Hito 4 gates."""
    started = time.perf_counter()
    try:
        solution = solve(omega_h, alpha_gb, resolution=32, tol=1e-11,
                         previous=previous, max_iterations=budget)
    except Exception as error:
        return dict(converged=False, accepted=False, reason=str(error)[:120],
                    seconds=time.perf_counter()-started)
    checked = diagnose(solution, alpha_gb, count=51)
    accepted = (checked['max_tensor_residual'] < TENSOR_GATE
                and solution.boundary_residual < BOUNDARY_GATE)
    return dict(converged=True, accepted=bool(accepted),
                iterations=int(solution.iterations),
                max_tensor_residual=checked['max_tensor_residual'],
                boundary_residual=float(solution.boundary_residual),
                seconds=time.perf_counter()-started,
                observables=measure(solution) if accepted else None)


def continuation_baseline(omega_h, alpha_gb, step=.025):
    """The conventional route, with its real cost: every solve it needs."""
    started = time.perf_counter()
    previous, solves, iterations = None, 0, 0
    for value in np.round(np.arange(0., alpha_gb+1e-9, step), 6):
        previous = solve(omega_h, float(value), resolution=32, tol=1e-11,
                         previous=previous, max_iterations=50)
        solves += 1
        iterations += int(previous.iterations)
    checked = diagnose(previous, alpha_gb, count=51)
    return dict(step=step, solves=solves, newton_iterations=iterations,
                seconds=time.perf_counter()-started,
                max_tensor_residual=checked['max_tensor_residual'],
                accepted=bool(checked['max_tensor_residual'] < TENSOR_GATE
                              and previous.boundary_residual < BOUNDARY_GATE))


def train(problem, theta0):
    """Gauss-Newton on the residual vector.

    L-BFGS on the sum of squares leaves the least-squares structure unused and
    plateaus three orders higher; the solver this is competing with is itself a
    Newton method, so a first-order method was never the right comparison.
    """
    started = time.perf_counter()
    result = least_squares(problem.residual_vector, theta0,
                           jac=problem.residual_jacobian, method='trf',
                           xtol=1e-15, ftol=1e-15, gtol=1e-15,
                           max_nfev=TRAINING_EVALUATIONS)
    bulk, horizon = problem.residual_norms(result.x)
    return result.x, dict(evaluations=int(result.nfev),
                          seconds=time.perf_counter()-started,
                          loss=float(2*result.cost),
                          max_bulk_residual=bulk, max_horizon_residual=horizon)


def run():
    gradient_check = []
    for alpha_gb in [.1, .5]:
        problem = Problem(OMEGA_H, alpha_gb)
        for seed in [0, 1]:
            theta = initial_parameters(np.random.default_rng(seed))
            worst, _, _ = check_gradient(problem, theta)
            gradient_check.append(dict(alpha_gb=alpha_gb, seed=seed,
                                       worst_relative_deviation=worst))

    boundary_check = []
    problem = Problem(OMEGA_H, .5)
    for seed in range(6):
        theta = np.random.default_rng(seed).normal(0., 2., n_parameters())
        x = np.array([1., 0.])
        value = Seed(problem, theta).evaluate(x)
        first = Seed(problem, theta).evaluate(x, 1)
        boundary_check.append(dict(seed=seed,
            B_infinity_error=abs(float(value[0][0])-1.),
            F_infinity_error=abs(float(value[1][0])-1.),
            dH_dx_infinity=abs(float(first[2][0])),
            dW_dx_infinity=abs(float(first[3][0])),
            W_horizon_error=abs(float(value[3][1])-OMEGA_H),
            H_infinity=float(value[2][0]), W_infinity=float(value[3][0])))

    controls, cells = [], []
    for alpha_gb in COUPLINGS:
        problem = Problem(OMEGA_H, alpha_gb)
        zero = np.zeros(n_parameters())
        controls.append(dict(alpha_gb=alpha_gb, alpha_paper=4*alpha_gb,
            trivial_seed=refine(OMEGA_H, alpha_gb, TrivialSeed(OMEGA_H)),
            untrained_network=refine(OMEGA_H, alpha_gb, Seed(problem, zero)),
            untrained_is_myers_perry=bool(np.max(np.abs(
                Seed(problem, zero).evaluate(problem.x)
                - MyersPerryAnchor(OMEGA_H)(problem.x)[0])) == 0.),
            continuation=continuation_baseline(OMEGA_H, alpha_gb)))
        print(f'controls done for alpha_gb={alpha_gb}', flush=True)

        for layers in ARCHITECTURES:
            for seed in SEEDS:
                problem = Problem(OMEGA_H, alpha_gb, layers=layers)
                theta0 = initial_parameters(np.random.default_rng(seed), layers)
                theta, training = train(problem, theta0)
                cells.append(dict(alpha_gb=alpha_gb, alpha_paper=4*alpha_gb,
                                  layers=list(layers), parameters=n_parameters(layers),
                                  seed=seed, training=training,
                                  refinement=refine(OMEGA_H, alpha_gb,
                                                    Seed(problem, theta))))
                print(f'  alpha_gb={alpha_gb} layers={layers} seed={seed}: '
                      f'max|bulk|={training["max_bulk_residual"]:.3e} '
                      f'accepted={cells[-1]["refinement"]["accepted"]}', flush=True)

    # A trained seed that steers the Newton to a point passing its own tolerance
    # while failing the independent tensor check is the failure mode this whole
    # apparatus exists to catch. Look for it explicitly rather than hoping.
    spurious = [c for c in cells
                if c['refinement'].get('converged') and not c['refinement']['accepted']
                and c['refinement'].get('max_tensor_residual', 0) > TENSOR_GATE]

    evidence = dict(gradient_check=gradient_check, boundary_check=boundary_check,
                    controls=controls, cells=cells,
                    spurious_convergences=[dict(alpha_gb=c['alpha_gb'],
                                                layers=c['layers'], seed=c['seed'],
                                                max_tensor_residual=c['refinement']
                                                ['max_tensor_residual'])
                                           for c in spurious],
                    budget=BUDGET, tensor_gate=TENSOR_GATE,
                    boundary_gate=BOUNDARY_GATE, omega_h=OMEGA_H,
                    seeds=SEEDS, architectures=[list(a) for a in ARCHITECTURES])
    checks = neural_checks(evidence)

    sources = [Path(__file__),
               ROOT/'src/rotating_bh/neural_seed.py',
               ROOT/'src/rotating_bh/neural_seed_validation.py',
               ROOT/'src/rotating_bh/egb_rotating_bvp.py',
               ROOT/'src/rotating_bh/egb_rotating_validation.py',
               ROOT/'src/rotating_bh/egb_rotating_observables.py',
               ROOT/'environment/requirements-lock.txt']
    sources += sorted((ROOT/'src/rotating_bh').glob('_egb_rotating*_generated.py'))

    data = dict(schema=1,
        units='r_H=G5=1; J is each spin; alpha_paper = 4 alpha_gb',
        scope='Neural seed for the rotating EGB spectral solver. The network only '
              'supplies the solver\'s `previous` argument; the solver, the adaptive '
              'method, the Hito 4A derivation and the generated equations are '
              'untouched. Acceptance uses the unchanged Hito 4 gates, so a seed that '
              'merely terminates the solver\'s own iteration does not count.',
        method='Myers-Perry-anchored tanh network in u=z^2, boundary conditions exact '
               'by construction, analytic gradient verified against a complex-step '
               'oracle over every parameter, trained by Gauss-Newton on the residual '
               'vector.',
        **evidence, checks=checks,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, sympy=sp.__version__),
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    output = ROOT/'results/egb-rotating-neural-seed.json'
    output.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

    manifest_path = ROOT/'artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    record = dict(id='egb-rotating-neural-seed', kind='result',
        path=str(output.relative_to(ROOT)).replace('\\', '/'),
        command='python experiments/egb_rotating_neural_seed.py',
        commit='working-tree', inputs=list(data['source_sha256']),
        parameters=dict(source_sha256=data['source_sha256'], omega_h=OMEGA_H,
                        couplings=COUPLINGS, architectures=[list(a) for a in ARCHITECTURES],
                        seeds=SEEDS, newton_budget=BUDGET,
                        training_evaluations=TRAINING_EVALUATIONS),
        environment='environment/requirements-lock.txt', agent='claude',
        prompt_refs=['prompts/prompt-log.md'],
        decisions=[
            'Benchmark placed where the conventional seeds fail, so the comparison can lose',
            'Both architectures and every random seed reported, not the ones that worked',
            'Acceptance by the unchanged Hito 4 gates, not by the solver terminating',
            'Cost reported as wall clock in this interpreter, which is not a speed claim'],
        checks=checks,
        status='verified' if all(c['passed'] for c in checks) else 'candidate',
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest = [r for r in manifest if r['id'] != record['id']]+[record]
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')

    for check in checks:
        print(f"{check['name']:<40} {'pass' if check['passed'] else 'FAIL'}")
    return 0 if all(c['passed'] for c in checks) else 1


if __name__ == '__main__':
    raise SystemExit(run())
