"""Resolution/tolerance study for the single accepted Hito 4B point.

No equations or solver algorithms are changed. Every run starts from the
same coarse numerical seed multiplied by 1.01, so tightening tolerance
does not merely return an already converged fine solution.
"""
from numbers import Real

import numpy as np

from .egb_rotating_adaptive import solve as adaptive
from .egb_rotating_bvp import solve as spectral, SolverFailure
from .egb_rotating_validation import diagnose
from .myers_perry import MyersPerry

RESOLUTIONS = (12, 16, 24, 32, 40)
TOLERANCES = (1e-5, 1e-8, 1e-11)
GRID = np.linspace(.02, .98, 151)


def _finite(value):
    return isinstance(value, Real) and np.isfinite(value)


def accepted(row):
    """Physical acceptance is stricter than the exploratory loose tolerance."""
    bounds = [('boundary_residual', 1e-8), ('max_tensor_residual', 1e-6),
              ('adaptive_difference', 1e-5), ('infinity_residual', 1e-8)]
    positive = ['min_B', 'min_F', 'min_h_over_r2', 'horizon_b_slope', 'horizon_f_slope']
    return bool(row.get('converged', False) and row.get('iterations', 0) > 0
                and all(_finite(row.get(k)) and 0 <= row[k] < limit for k, limit in bounds)
                and all(_finite(row.get(k)) and row[k] > 0 for k in positive))


def closure_checks(result):
    """Reject missing cells; permit documented strict-tolerance failures."""
    rows = result['runs']
    cells = {(row['resolution'], row['tolerance']): row for row in rows}
    expected = {(n, tol) for n in RESOLUTIONS for tol in TOLERANCES}
    nominal = [cells.get((n, 1e-8), {}) for n in (24, 32, 40)]
    strict = [cells.get((n, 1e-11), {}) for n in (24, 32, 40)]
    coarse = cells.get((12, 1e-8), {}).get('max_tensor_residual')
    fine = cells.get((40, 1e-8), {}).get('max_tensor_residual')
    stable = result.get('fine_resolution_difference')
    tensor = result.get('adaptive_max_tensor_residual')
    boundary = result.get('adaptive_boundary_residual')
    collocation = result.get('adaptive_collocation_residual')
    seed_rows = result.get('seed_runs', [])
    seed_alphas = [row.get('alpha_gb') for row in seed_rows]
    mp = result.get('myers_perry_difference')
    seed_valid = (len(seed_rows) == 4 and set(seed_alphas) == {0., .02, .05, .1}
                  and _finite(mp) and 0 <= mp < 1e-5
                  and all(_finite(row.get('max_tensor_residual'))
                          and 0 <= row['max_tensor_residual'] < 1e-6 for row in seed_rows))
    return [
        dict(name='complete_resolution_tolerance_grid', passed=len(rows) == len(expected)
             and set(cells) == expected),
        dict(name='independently_verified_seed', passed=bool(seed_valid)),
        dict(name='three_fine_nominal_solutions', passed=all(accepted(row) for row in nominal)),
        dict(name='two_fine_strict_solutions', passed=sum(accepted(row) for row in strict) >= 2),
        dict(name='tensor_resolution_improvement', passed=bool(_finite(coarse) and _finite(fine)
             and coarse > 0 and fine < coarse/100)),
        dict(name='fine_profiles_stable', passed=bool(_finite(stable) and 0 <= stable < 1e-8)),
        dict(name='independent_adaptive_tensor', passed=bool(_finite(tensor) and 0 <= tensor < 1e-6
             and _finite(boundary) and 0 <= boundary < 1e-8
             and _finite(collocation) and 0 <= collocation <= 1e-7)),
    ]


def _geometry(solution):
    values = solution.evaluate(np.linspace(0., 1., 501))
    x = np.linspace(0., 1., 501)
    horizon = solution.evaluate(0.)
    infinity = solution.evaluate(1.)
    first = solution.evaluate(1., 1)
    return dict(min_B=float(np.min(values[0])), min_F=float(np.min(values[1])),
                min_h_over_r2=float(np.min(1+(1-x)**4*values[2])),
                horizon_b_slope=float(2*horizon[0]), horizon_f_slope=float(2*horizon[1]),
                infinity_residual=float(np.max(np.abs(np.r_[infinity[:2]-1, first[2:]]))))


def run_study(*, resolutions=RESOLUTIONS, tolerances=TOLERANCES, tensor_count=51, progress=None):
    """Run requested cells; only the complete default grid can pass closure."""
    result = dict(parameters=dict(omega_h=.3, alpha_gb=.1, r_h=1.,
                                  resolutions=list(resolutions), tolerances=list(tolerances),
                                  seed_resolution=16, seed_tolerance=1e-9, seed_factor=1.01,
                                  max_iterations=15, tensor_count=tensor_count,
                                  profile_interval=[.02, .98], profile_count=151,
                                  adaptive_cutoff=.0015, adaptive_tolerance=1e-7),
                  seed_runs=[], runs=[], seed_verified=False, myers_perry_difference=None,
                  adaptive_max_tensor_residual=None, fine_resolution_difference=None)
    result['parameters']['acceptance'] = dict(boundary_tolerance=1e-8, tensor_tolerance=1e-6,
                                              methods_tolerance=1e-5, fine_profile_tolerance=1e-8,
                                              minimum_tensor_improvement=100,
                                              minimum_fine_strict_solutions=2)
    seed = None
    for alpha in (0., .02, .05, .1):
        seed = spectral(.3, alpha, resolution=16, tol=1e-9, previous=seed)
        diagnostic = diagnose(seed, alpha, count=tensor_count)
        result['seed_runs'].append(dict(alpha_gb=alpha, **diagnostic))
        if diagnostic['max_tensor_residual'] >= 1e-6:
            raise SolverFailure(f'coarse seed failed independent tensor gate at alpha={alpha}')
        if alpha == 0:
            z = 1-GRID
            closed = MyersPerry(r_h=1., omega_h=.3).functions(1/z)
            reference = np.array([closed['b']/(1-z*z), closed['f']/(1-z*z),
                                  (closed['h']*z*z-1)/z**4, closed['w']/z**4])
            result['myers_perry_difference'] = float(np.max(np.abs(seed.evaluate(GRID)-reference)))
    result['seed_verified'] = result['myers_perry_difference'] < 1e-5
    reference = adaptive(.3, .1, previous=seed, cutoff=.0015, tol=1e-7)
    result['adaptive_max_tensor_residual'] = diagnose(reference, .1, count=tensor_count)['max_tensor_residual']
    result['adaptive_boundary_residual'] = reference.boundary_residual
    result['adaptive_collocation_residual'] = reference.collocation_residual
    result['adaptive_nodes'] = int(reference.nodes.size)
    reference_values = reference.evaluate(GRID)

    class PerturbedSeed:
        def evaluate(self, x, derivative=0):
            return 1.01*seed.evaluate(x, derivative)

    nominal_profiles = {}
    for tol in tolerances:
        for n in resolutions:
            row = dict(resolution=n, tolerance=tol)
            try:
                sol = spectral(.3, .1, resolution=n, tol=tol, previous=PerturbedSeed(),
                               max_iterations=15)
            except SolverFailure as error:
                row.update(converged=False, accepted=False, reason=str(error))
            else:
                values = sol.evaluate(GRID)
                row.update(converged=True, iterations=sol.iterations,
                           boundary_residual=sol.boundary_residual,
                           adaptive_difference=float(np.max(np.abs(values-reference_values))),
                           **_geometry(sol), **diagnose(sol, .1, count=tensor_count))
                row['accepted'] = accepted(row)
                if tol == 1e-8:
                    nominal_profiles[n] = values
            result['runs'].append(row)
            if progress is not None:
                progress(row)
    if 32 in nominal_profiles and 40 in nominal_profiles:
        result['fine_resolution_difference'] = float(np.max(np.abs(nominal_profiles[32]-nominal_profiles[40])))
    result['checks'] = closure_checks(result)
    return result
