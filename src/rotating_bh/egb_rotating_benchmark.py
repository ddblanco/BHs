"""First-pass benchmark for the rotating EGB solver (Hito 4B).

Scope, explicitly: ONE (omega_h) with a short continuation in alpha_GB, a
two-resolution convergence check, a negative control and a resource-failure
control. This is NOT the Hito 4C family/comparison sweep (several omega_h,
a second coupling, external comparison) -- see docs/egb-rotating.md and
ROADMAP.md for what that still needs (an adaptive method for cross-checking
the spectral one, a resolution/tolerance study at the hardest point, and
the external comparison itself). Kept small because each solve() here costs
several seconds to tens of seconds (the EGB equations are much larger than
vacuum's), unlike Hitos 1-3's benchmarks.
"""
import numpy as np

from .egb_rotating_bvp import solve, SolverFailure
from .egb_rotating_validation import diagnose, tensor_residual
from .myers_perry import MyersPerry

OMEGA_H = 0.3
ALPHA_GB_SWEEP = [0.0, 0.02, 0.05, 0.1]
RESOLUTIONS = [16, 24]
CHECK_POINTS = 21


def run_benchmark():
    runs = []
    previous = None
    solutions = {}
    for alpha_gb in ALPHA_GB_SWEEP:
        solution = solve(OMEGA_H, alpha_gb, resolution=32, previous=previous)
        previous = solution
        solutions[alpha_gb] = solution
        result = diagnose(solution, alpha_gb, count=CHECK_POINTS)
        runs.append(dict(alpha_gb=alpha_gb, resolution=solution.initial_resolution,
                          iterations=solution.iterations,
                          boundary_residual=solution.boundary_residual, **result))

    # GR limit (alpha_gb=0): exact closed-form Myers-Perry comparison, no
    # solver tolerance involved in computing the reference itself.
    mp = MyersPerry(r_h=1.0, omega_h=OMEGA_H)
    mp_diff = []
    sol0 = solutions[0.0]
    for r in (1.5, 2.0, 5.0, 10.0, 20.0):
        zz = 1/r
        x = 1-zz
        B, F, H, W = sol0.evaluate(np.array([x]))
        b_repr, f_repr, w_repr = (1-zz**2)*B[0], (1-zz**2)*F[0], zz**4*W[0]
        expected = mp.functions(np.array([r]))
        mp_diff.append(max(abs(b_repr-expected['b'][0]), abs(f_repr-expected['f'][0]),
                            abs(w_repr-expected['w'][0])))
    max_mp_difference = float(max(mp_diff))

    # Resolution convergence at the hardest coupling reached above.
    hardest_alpha = ALPHA_GB_SWEEP[-1]
    resolution_sweep = []
    prev_for_resolution = None
    for n in RESOLUTIONS:
        sol = solve(OMEGA_H, 0.0, resolution=n)
        for alpha_gb in ALPHA_GB_SWEEP[1:]:
            sol = solve(OMEGA_H, alpha_gb, resolution=n, previous=sol)
        result = diagnose(sol, hardest_alpha, count=CHECK_POINTS)
        resolution_sweep.append(dict(resolution=n, **result))

    negative_control = tensor_residual(solutions[hardest_alpha], 0.5, hardest_alpha+0.05)

    failures = []
    try:
        solve(OMEGA_H, hardest_alpha, resolution=16, max_iterations=0)
    except SolverFailure as error:
        failures.append(dict(converged=False, reason=str(error)))
    else:
        failures.append(dict(converged=True))

    checks = [
        dict(name='myers_perry_limit', passed=max_mp_difference < 1e-6, value=max_mp_difference),
        dict(name='resolution_convergence',
             passed=resolution_sweep[-1]['max_tensor_residual'] < resolution_sweep[0]['max_tensor_residual']),
        dict(name='tensor_residual_accepted',
             passed=all(r['max_tensor_residual'] < 1e-6 for r in runs)),
        dict(name='perturbation_detected', passed=negative_control > 1e-5, value=negative_control),
        dict(name='real_resource_failures_reported', passed=all(not f['converged'] for f in failures)),
    ]
    return dict(
        parameters=dict(r_h=1.0, omega_h=OMEGA_H, alpha_gb_sweep=ALPHA_GB_SWEEP,
                         resolutions=RESOLUTIONS, default_resolution=32, check_points=CHECK_POINTS,
                         tensor_tolerance=1e-6),
        runs=runs, myers_perry_max_difference=max_mp_difference, resolution_sweep=resolution_sweep,
        negative_control_residual=negative_control, failure_controls=failures, checks=checks)
