"""The metric functions b, f, h, w at every coupling the paper reports.

Added in revision. Every number in the paper is a functional of the four radial
amplitudes, but until now nothing in the distribution showed those amplitudes
themselves at more than two couplings (`results/egb-rotating-family.json`), and
a reader had no way to see that the nonlinear radial boundary-value problem was
in fact solved at each of the thirteen couplings of `egb_extremality.py` rather
than at a handful of them with the rest inferred.

This experiment re-solves that problem from scratch at every one of those
couplings, at the single spin `q = START_SPIN` that all thirteen walks share, so
that the comparison isolates the coupling and nothing else, and it records:

* the four profiles on a common radial grid, for `figures/fig-profiles.pdf`;
* the horizon and asymptotic values a reader can check by eye -- `b(r_H)`,
  `f(r_H)`, `w(r_H)`, `h(r_H)/r_H^2` and the tails;
* the observables, compared against the corresponding accepted state already in
  `results/egb-extremality.json`.

The last item is the point of the experiment as a check rather than as a
picture. The route here is deliberately different from the production one: a
spin ladder in the vacuum followed by a coupling ladder at fixed spin, whereas
`egb_extremality.py` descends in the spin at fixed coupling from a
response-seeded predictor. Two different continuation paths to the same
solutions agree or they do not, and the gate below refuses the artifact if they
do not.

Run with PYTHONPATH=src from project/.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from rotating_bh.egb_rotating_observables import measure
from rotating_bh.egb_rotating_predictor import solve
from rotating_bh.extremality import BOUNDARY_GATE, RELATIVE_TENSOR_GATE, scaled_diagnose

ROOT = Path(__file__).resolve().parents[1]

# The couplings of egb_extremality.EXTREMAL_COUPLINGS, which are the rows of the
# paper's extremal table; kept as a literal here so that a change to either list
# shows up as a disagreement rather than propagating silently.
COUPLINGS = [0., .005, .01, .02, .035, .05, .075, .1, .15, .2, .3, .4, .5]
# egb_extremality.START_SPIN: the one spin every walk passes through.
START_SPIN = .60
RESOLUTION = 64
TOLERANCE = 1e-10
# Ladder steps for the two continuations. Small enough that the previous
# solution seeds the next Newton solve; no response predictor is used here, so
# that this route shares as little as possible with the production one.
SPIN_STEP = .02
COUPLING_STEP = .01
# Radial sampling for the figure: uniform in the compact coordinate, which
# concentrates points near the horizon where the profiles bend.
SAMPLES = 601
# r beyond which the profiles are flat to plotting accuracy; the stored grid is
# truncated there so the artifact holds a plottable range rather than 1/z as
# z -> 0.
R_MAX = 30.


def profile(solution):
    """The four metric functions of eq. (4) on a common radial grid."""
    x = np.linspace(0., 1.-1./R_MAX, SAMPLES)
    B, F, H, W = solution.evaluate(x)
    z = 1.-x
    return dict(r=(1./z).tolist(), b=((1.-z*z)*B).tolist(), f=((1.-z*z)*F).tolist(),
                h_over_r2=(1.+z**4*H).tolist(), h=((1.+z**4*H)/(z*z)).tolist(),
                w=(z**4*W).tolist())


def run():
    reference = json.loads((ROOT/'results/egb-extremality.json').read_text(encoding='utf-8'))
    walks = {float(k): v for k, v in reference['walks'].items()}

    # Continuation one: spin ladder in the vacuum, up to the shared spin.
    seed = None
    for q in np.arange(SPIN_STEP, START_SPIN+1e-12, SPIN_STEP):
        seed = solve(float(q), 0., resolution=RESOLUTION, tol=TOLERANCE, previous=seed)

    rows = []
    for alpha_gb in COUPLINGS:
        # Continuation two: coupling ladder at fixed spin.
        solution = seed
        if alpha_gb > 0:
            steps = np.linspace(0., alpha_gb, max(3, int(round(alpha_gb/COUPLING_STEP))+1))
            for step in steps[1:]:
                solution = solve(START_SPIN, float(step), resolution=RESOLUTION,
                                 tol=TOLERANCE, previous=solution)
        observed = measure(solution)
        diagnosed = scaled_diagnose(solution, alpha_gb, count=51)

        # The accepted production state at the same coupling and spin.
        group = walks[alpha_gb]
        near = min(group, key=lambda s: abs(s['omega_h']-START_SPIN))
        matched = abs(near['omega_h']-START_SPIN) < 1e-9
        agreement = {name: abs(observed[name]-near[name])/abs(near[name])
                     for name in ('E', 'J', 'S', 'T_H')} if matched else None

        horizon = solution.evaluate(0.)
        rows.append(dict(
            alpha_gb=alpha_gb, q=START_SPIN, resolution=RESOLUTION,
            iterations=int(solution.iterations),
            boundary_residual=float(solution.boundary_residual),
            max_relative_tensor_residual=diagnosed['max_relative_tensor_residual'],
            E=observed['E'], J=observed['J'], S=observed['S'], T_H=observed['T_H'],
            # Read off the ansatz at the horizon: b and f vanish there, w equals
            # the horizon angular velocity, and h/r^2 is the squashing of the
            # horizon three-sphere, which is 1/(1-q^2) in the vacuum.
            b_horizon=float((1.-1.)*horizon[0]), f_horizon=float((1.-1.)*horizon[1]),
            h_over_r2_horizon=float(1.+horizon[2]), w_horizon=float(horizon[3]),
            production_state=(dict(omega_h=near['omega_h'], E=near['E'], J=near['J'],
                                   S=near['S'], T_H=near['T_H']) if matched else None),
            agreement=agreement,
            profile=profile(solution)))

    squashing = [r['h_over_r2_horizon'] for r in rows]
    worst_agreement = max(max(r['agreement'].values()) for r in rows if r['agreement'])
    gates = [
        dict(name='every_paper_coupling_solved',
             passed=len(rows) == len(COUPLINGS),
             detail=f'{len(rows)} of {len(COUPLINGS)} couplings returned a solution'),
        dict(name='boundary_residual_under_gate',
             passed=max(r['boundary_residual'] for r in rows) < BOUNDARY_GATE,
             detail=f"worst {max(r['boundary_residual'] for r in rows):.3e}"),
        dict(name='independent_tensor_residual_under_gate',
             passed=max(r['max_relative_tensor_residual'] for r in rows) < RELATIVE_TENSOR_GATE,
             detail=f"worst {max(r['max_relative_tensor_residual'] for r in rows):.3e}"),
        dict(name='agrees_with_production_states',
             passed=worst_agreement < 1e-8,
             detail=f'worst relative difference {worst_agreement:.3e} over E, J, S, T_H'),
        dict(name='vacuum_horizon_squashing_matches_myers_perry',
             passed=abs(squashing[0]-1./(1.-START_SPIN**2)) < 1e-9,
             detail=f'h(r_H)/r_H^2 = {squashing[0]:.9f} against 1/(1-q^2) = '
                    f'{1./(1.-START_SPIN**2):.9f}'),
        dict(name='horizon_angular_velocity_imposed',
             passed=max(abs(r['w_horizon']-START_SPIN) for r in rows) < 1e-12,
             detail=f"worst {max(abs(r['w_horizon']-START_SPIN) for r in rows):.3e}"),
    ]

    sources = [Path(__file__).resolve(),
               ROOT/'src/rotating_bh/egb_rotating_bvp.py',
               ROOT/'src/rotating_bh/egb_rotating_predictor.py',
               ROOT/'src/rotating_bh/egb_rotating_observables.py',
               ROOT/'results/egb-extremality.json']
    data = dict(
        schema='egb-rotating-profile-atlas/1',
        units='G_5 = r_H = 1; alpha_gb = alpha_GB/r_H^2; q = r_H Omega_H',
        scope='The four radial amplitudes of eq. (4) at the thirteen couplings of '
              'the paper, all at the shared spin q = 0.60, re-solved along a '
              'continuation route that differs from the production one.',
        start_spin=START_SPIN, couplings=COUPLINGS, resolution=RESOLUTION,
        tolerance=TOLERANCE, samples=SAMPLES, r_max=R_MAX,
        spin_step=SPIN_STEP, coupling_step=COUPLING_STEP,
        rows=rows, checks=gates,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__),
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    output = ROOT/'results/egb-rotating-profiles.json'
    output.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    manifest_path = ROOT/'artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    record = dict(
        id='egb-rotating-profiles', kind='result',
        path=str(output.relative_to(ROOT)).replace('\\', '/'),
        command='python experiments/egb_rotating_profile_atlas.py',
        commit='working-tree', inputs=list(data['source_sha256']),
        parameters=dict(source_sha256=data['source_sha256'], couplings=COUPLINGS,
                        start_spin=START_SPIN, resolution=RESOLUTION,
                        tolerance=TOLERANCE, spin_step=SPIN_STEP,
                        coupling_step=COUPLING_STEP, samples=SAMPLES, r_max=R_MAX),
        environment='environment/requirements-lock.txt', agent='claude',
        # The prompt log is not part of this distribution; every other record
        # here carries an empty list for the same reason.
        prompt_refs=[],
        decisions=[
            'One shared spin for all couplings, so the figure isolates the coupling',
            'A continuation route different from the production one, so agreement is evidence',
            'Agreement with the production states is a gate, not a remark',
            'The vacuum horizon squashing is checked against the closed Myers-Perry value'],
        checks=gates,
        status='verified' if all(c['passed'] for c in gates) else 'candidate',
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest = [r for r in manifest if r['id'] != record['id']]+[record]
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')

    for gate in gates:
        print(f"{gate['name']:<46} {'pass' if gate['passed'] else 'FAIL'}  {gate['detail']}")
    return 0 if all(c['passed'] for c in gates) else 1


if __name__ == '__main__':
    raise SystemExit(run())
