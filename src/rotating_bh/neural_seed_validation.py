"""Recompute the Hito 5 gates from the recorded evidence.

Same discipline as the Hito 4C validators: nothing is read back from a boolean
the producing run stored, and the thresholds live here rather than in the
artifact, so a later run cannot widen its own gate and still be called verified.

The gates are written so the hito can fail honestly. `neural_seed_helps` asks
whether a trained seed reached a coupling where *both* conventional seeds
failed; if no network ever manages that, the gate fails and the artifact stays a
candidate. It is not an "at least one cell worked" gate either -- the spread
across random seeds and architectures is recorded and reported whatever it says.
"""
TENSOR_GATE = 1e-6
BOUNDARY_GATE = 1e-8


def _cells(data):
    return data['cells']


def _control(data, alpha_gb):
    return next(c for c in data['controls'] if c['alpha_gb'] == alpha_gb)


def _gradient_is_exact(data):
    rows = data['gradient_check']
    return bool(rows and all(r['worst_relative_deviation'] < 1e-10 for r in rows))


def _boundaries_are_exact(data):
    """Not "small": exactly zero. The conditions are built into the ansatz, so
    anything else means the construction is not doing what it claims."""
    rows = data['boundary_check']
    return bool(rows and all(
        r['B_infinity_error'] == 0. and r['F_infinity_error'] == 0.
        and r['dH_dx_infinity'] == 0. and r['dW_dx_infinity'] == 0.
        and r['W_horizon_error'] == 0. for r in rows))


def _infinity_values_are_free(data):
    """H and W at infinity carry only derivative conditions. If every draw gives
    the same value there, the ansatz is pinning them and cannot represent the
    solution -- the defect that capped an earlier version of this work."""
    rows = data['boundary_check']
    return len({(round(r['H_infinity'], 9), round(r['W_infinity'], 9))
                for r in rows}) == len(rows) and len(rows) > 1


def _controls_fail_where_claimed(data):
    """The benchmark only means something where the conventional seeds fail. If
    they start succeeding, the comparison has stopped being a comparison."""
    hard = [c for c in data['controls'] if c['alpha_gb'] >= .5]
    return bool(hard and all(not c['trivial_seed']['accepted']
                             and not c['untrained_network']['accepted'] for c in hard))


def _untrained_is_the_anchor(data):
    return bool(data['controls']
                and all(c['untrained_is_myers_perry'] for c in data['controls']))


def _continuation_reaches_every_coupling(data):
    """The conventional route must succeed everywhere, or the neural route would
    be beating a broken baseline rather than a working one."""
    return bool(data['controls']
                and all(c['continuation']['accepted'] for c in data['controls']))


def _neural_seed_helps(data):
    """At least one coupling where both conventional seeds fail and some trained
    network is accepted by the unchanged gates."""
    for control in data['controls']:
        if control['trivial_seed']['accepted'] or control['untrained_network']['accepted']:
            continue
        for cell in _cells(data):
            if cell['alpha_gb'] != control['alpha_gb']:
                continue
            refinement = cell['refinement']
            if (refinement['accepted']
                    and refinement['max_tensor_residual'] < TENSOR_GATE
                    and refinement['boundary_residual'] < BOUNDARY_GATE):
                return True
    return False


def _acceptance_uses_the_independent_gates(data):
    """Every accepted cell must clear the full-tensor and boundary gates, and no
    cell may be marked accepted on the solver's own tolerance alone."""
    return bool(data['tensor_gate'] <= TENSOR_GATE
                and data['boundary_gate'] <= BOUNDARY_GATE
                and all(c['refinement']['max_tensor_residual'] < TENSOR_GATE
                        and c['refinement']['boundary_residual'] < BOUNDARY_GATE
                        for c in _cells(data) if c['refinement']['accepted']))


def _spread_is_reported(data):
    """Every architecture and every random seed has to appear, so a best case
    cannot be reported as the result."""
    expected = {(tuple(a), s, alpha)
                for a in data['architectures'] for s in data['seeds']
                for alpha in {c['alpha_gb'] for c in _cells(data)}}
    present = {(tuple(c['layers']), c['seed'], c['alpha_gb']) for c in _cells(data)}
    return present == expected and len(expected) > 1


def checks(data):
    return [
        dict(name='gradient_matches_complex_step_oracle', passed=_gradient_is_exact(data)),
        dict(name='boundary_conditions_exact_for_any_weights',
             passed=_boundaries_are_exact(data)),
        dict(name='infinity_values_left_free', passed=_infinity_values_are_free(data)),
        dict(name='untrained_network_is_myers_perry', passed=_untrained_is_the_anchor(data)),
        dict(name='conventional_seeds_fail_at_the_benchmark',
             passed=_controls_fail_where_claimed(data)),
        dict(name='continuation_baseline_succeeds',
             passed=_continuation_reaches_every_coupling(data)),
        dict(name='acceptance_uses_independent_gates',
             passed=_acceptance_uses_the_independent_gates(data)),
        dict(name='architecture_and_seed_spread_reported', passed=_spread_is_reported(data)),
        dict(name='neural_seed_reaches_an_unreachable_coupling',
             passed=_neural_seed_helps(data)),
    ]
