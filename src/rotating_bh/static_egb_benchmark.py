"""Deterministic sweep, GR-limit convergence and independent checks."""
import numpy as np

from .static_egb import StaticEGB
from .static_egb_bvp import solve, SolverFailure
from .static_egb_validation import diagnose, tensor_residual

ALPHA_HAT = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SPECTRAL_RESOLUTIONS = [16, 20, 24, 32, 48]
ADAPTIVE_TOLERANCES = [1e-8, 1e-10, 1e-12, 1e-13]
HARDEST_ALPHA = 0.5


def run_benchmark():
    runs = []
    x = np.linspace(.01, .98, 201)
    selected = {}
    previous = None
    for alpha in ALPHA_HAT:
        spectral = solve(alpha, method='spectral', resolution=32, tol=1e-11, previous=previous)
        previous = spectral
        adaptive = solve(alpha, method='adaptive', tol=1e-13)
        for method, solution in (('spectral', spectral), ('adaptive', adaptive)):
            runs.append(dict(method=method, alpha_hat=alpha,
                              resolution=solution.initial_resolution, tolerance=solution.tolerance,
                              iterations=solution.iterations, **diagnose(solution, alpha)))
        selected[alpha] = dict(spectral=spectral, adaptive=adaptive)

    method_differences = [float(np.max(np.abs(
        selected[a]['spectral'].evaluate(x)-selected[a]['adaptive'].evaluate(x))))
        for a in ALPHA_HAT]

    # GR limit: exact closed-form difference against alpha_hat=0 (no solver involved),
    # measured, not assumed to follow a fixed power a priori.
    r = np.geomspace(1.01, 50., 40)
    gr = StaticEGB(r_h=1., alpha_gb=0.).functions(r)['f']
    gr_limit = [dict(alpha_hat=a, max_difference=float(np.max(np.abs(
        StaticEGB(r_h=1., alpha_gb=a).functions(r)['f']-gr))))
        for a in ALPHA_HAT]

    resolution_sweep = [dict(resolution=n, **diagnose(
        solve(HARDEST_ALPHA, method='spectral', resolution=n, tol=1e-11), HARDEST_ALPHA))
        for n in SPECTRAL_RESOLUTIONS]
    tolerance_sweep = [dict(tolerance=tol, **diagnose(
        solve(HARDEST_ALPHA, method='adaptive', tol=tol), HARDEST_ALPHA))
        for tol in ADAPTIVE_TOLERANCES]

    baseline = solve(HARDEST_ALPHA, method='spectral', resolution=32, tol=1e-11)
    negative_control = tensor_residual(baseline, 2.0, HARDEST_ALPHA+0.05)  # wrong alpha_GB on purpose

    failures = []
    try:
        solve(HARDEST_ALPHA, method='spectral', resolution=10, tol=1e-14, max_iterations=0)
    except SolverFailure as error:
        failures.append(dict(method='spectral', converged=False, reason=str(error)))
    else:
        failures.append(dict(method='spectral', converged=True))

    fine = [r for r in runs if (r['method'] == 'spectral' and r['resolution'] >= 32)
            or (r['method'] == 'adaptive' and r['tolerance'] <= 1e-12)]
    checks = [
        dict(name='fine_solutions_independently_accepted', passed=all(r['accepted'] for r in fine)),
        dict(name='spectral_resolution_convergence',
             passed=resolution_sweep[-1]['profile_error'] < resolution_sweep[0]['profile_error']/1e4),
        dict(name='adaptive_tolerance_convergence',
             passed=tolerance_sweep[-1]['einstein_residual'] < tolerance_sweep[0]['einstein_residual']/10),
        dict(name='methods_agree', passed=max(method_differences) < 1e-6, value=max(method_differences)),
        dict(name='gr_limit_shrinks_with_alpha',
             # ALPHA_HAT is increasing; the GR-oracle difference must grow with
             # it (equivalently: shrink to 0 as alpha_hat -> 0), not the reverse.
             passed=all(gr_limit[i]['max_difference'] >= gr_limit[i-1]['max_difference']
                        for i in range(1, len(gr_limit))),
             value=[g['max_difference'] for g in gr_limit]),
        dict(name='perturbation_detected', passed=negative_control > 1e-5, value=negative_control),
        dict(name='real_resource_failures_reported', passed=all(not f['converged'] for f in failures)),
    ]
    return dict(
        parameters=dict(r_h=1., G5=1., alpha_hat_sweep=ALPHA_HAT,
                         spectral_resolutions=SPECTRAL_RESOLUTIONS, spectral_default_resolution=32,
                         adaptive_tolerances=ADAPTIVE_TOLERANCES, adaptive_default_tolerance=1e-13,
                         hardest_alpha_hat=HARDEST_ALPHA, cutoff=1e-3, check_points=201,
                         profile_tolerance=1e-6, ode_tolerance=1e-6, tensor_tolerance=1e-6,
                         horizon_tolerance=1e-8),
        runs=runs, method_differences=method_differences, gr_limit=gr_limit,
        resolution_sweep=resolution_sweep, tolerance_sweep=tolerance_sweep,
        negative_control_residual=negative_control, failure_controls=failures, checks=checks)
