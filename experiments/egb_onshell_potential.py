"""The Gauss-Bonnet potential as an on-shell action integral.

The extended first law defines `Psi = dM/dalpha|_{S,J}` and the manuscript
measures it by implicit linear response. There is a second reading of the same
quantity. At fixed temperature and angular velocities the Gibbs potential is
the on-shell action per unit time, and the only *explicit* dependence of that
action on the coupling is the Gauss-Bonnet term itself, so

    Psi = -(1/16 pi) Int_Sigma sqrt(-g) L_GB d^4x ,

the integral running over a stationary spatial slice outside the horizon. The
identity is a statement about the solution, not about the solver: it uses no
Jacobian, no linear solve, no parameter derivative, and -- unlike the Smarr
route -- no asymptotic charge extraction. This script tests it on every
archived spectral profile.

`L_GB` comes from the independent curvature evaluator that already gates
acceptance (`einstein.curvature` + `einstein.gauss_bonnet`); the metric
determinant comes from the same compiled metric. The angular integral is done
in closed form: on this ansatz `sqrt(-g) L_GB = sin(theta) cos(theta) D(r)`,
which is checked here rather than assumed, so

    Int dtheta dphi_1 dphi_2 = 2 pi^2  and  Psi = -(pi/8) Int_{r_H}^infty D dr .

The radial quadrature is Gauss-Legendre in the compactified coordinate
`x = 1 - 1/r`. It cannot be refined indefinitely: `L_GB` is a difference of
curvature squares that cancels to `O(r^-8)` at large `r`, so nodes pushed close
to `x = 1` lose the density to float64 cancellation while `dr/dx = (1-x)^-2`
amplifies what is left. The node count is therefore a recorded parameter and
the convergence ladder is stored with the result.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from rotating_bh.egb_rotating_saved import SavedProfile
from rotating_bh.egb_rotating_validation import (_metric_jets, diagnose,
                                                 physical_jets)
from rotating_bh.einstein import curvature, gauss_bonnet
from rotating_bh.environment import environment_report
from rotating_bh.gb_response import response_field, response_observables

ROOT = Path(__file__).resolve().parents[1]
NODES = 80                 # production quadrature
LADDER = (40, 60, 80, 120, 160)
TENSOR_GATE = 1e-6
THETA = .6


def _density(solution, x, theta):
    """sqrt(-g) L_GB at compact coordinate x and polar angle theta."""
    jets = physical_jets(solution, np.array([x]))[:, 0]
    r_, b, bp, bpp, f, fp, fpp, h, hp, hpp, w, wp, wpp = jets
    arrays = _metric_jets()(0, r_, theta, 0, 0, r_, b, bp, bpp, f, fp, fpp,
                            h, hp, hpp, w, wp, wpp)
    result = curvature(*arrays)
    lovelock = gauss_bonnet(arrays[0], result['inverse'], result['riemann'],
                            result['ricci'], result['scalar'])
    return float(np.sqrt(-np.linalg.det(arrays[0]))*lovelock['lagrangian'])


def _radial(solution, x):
    """D(r) = sqrt(-g) L_GB / (sin theta cos theta), the angular factor removed."""
    return _density(solution, x, THETA)/(np.sin(THETA)*np.cos(THETA))


def onshell_potential(solution, nodes=NODES):
    """-(1/16 pi) Int sqrt(-g) L_GB d^4x, the angular integral in closed form."""
    points, weights = np.polynomial.legendre.leggauss(nodes)
    points, weights = .5*(points+1), .5*weights
    total = sum(weight*_radial(solution, float(point))/(1-point)**2
                for point, weight in zip(points, weights))
    return -np.pi/8*total


def angular_factorisation(solution, x=.5):
    """Largest relative spread of sqrt(-g) L_GB / (sin cos) over theta."""
    values = [_density(solution, x, t)/(np.sin(t)*np.cos(t))
              for t in (.2, .6, 1.1, 1.4)]
    return float(max(abs(v/values[0]-1) for v in values))


def run():
    family = json.loads((ROOT/'results/egb-rotating-family.json').read_text())
    states, seen = [], set()
    for record in family['records']:
        key = (record['alpha_gb'], record['q'])
        if (record['accepted'] and record['q'] > 0 and record['alpha_gb'] > 0
                and record['N'] == 32 and key not in seen):
            seen.add(key)
            states.append(record)
    states.sort(key=lambda r: (r['alpha_gb'], r['q']))

    rows, worst_tensor, factorisation = [], 0., 0.
    for record in states:
        solution = SavedProfile(record)
        alpha = record['alpha_gb']
        tensor = diagnose(solution, alpha, count=51)['max_tensor_residual']
        if tensor >= TENSOR_GATE:
            raise RuntimeError(f'stored state fails the tensor gate: {record["label"]}')
        worst_tensor = max(worst_tensor, tensor)
        factorisation = max(factorisation, angular_factorisation(solution))
        response = response_observables(
            solution, response=response_field(solution, resolution=record['N']))
        psi = response['psi_gb']
        ladder = {int(n): float(onshell_potential(solution, n)) for n in LADDER}
        value = ladder[NODES]
        rows.append(dict(alpha_gb=alpha, q=record['q'], N=record['N'],
                         psi_response=float(psi), psi_onshell=float(value),
                         absolute=float(abs(value-psi)),
                         relative=float(abs(value-psi)/abs(psi)),
                         tensor=float(tensor), ladder=ladder))
        print(f'alpha={alpha:<8g} q={record["q"]:<6g} '
              f'psi={psi: .8f} onshell={value: .8f} '
              f'rel={rows[-1]["relative"]:.2e}', flush=True)

    worst_relative = max(r['relative'] for r in rows)
    worst_absolute = max(r['absolute'] for r in rows)
    # The two routes must agree in sign as well as in value: a positive Psi
    # is a negative integrated Gauss-Bonnet density over the exterior slice.
    same_sign = all(r['psi_response']*r['psi_onshell'] > 0 for r in rows)
    result = dict(
        schema=1,
        units='r_H=G5=1; J is each spin; alpha_paper = 4 alpha_gb',
        scope='Psi of the rotating EGB family read as an on-shell action '
              'integral, tested against the implicit linear response on every '
              'archived spectral profile. The states are the moderate-spin '
              'family members whose Chebyshev coefficients were stored; the '
              'near-extremal production states of egb-extremality.json carry '
              'no profiles and are not covered here.',
        identity='Psi = -(1/16 pi) Int_Sigma sqrt(-g) L_GB d^4x',
        nodes=NODES, ladder=list(LADDER), theta=THETA,
        tensor_gate=TENSOR_GATE,
        states=len(rows), rows=rows,
        worst_relative=worst_relative, worst_absolute=worst_absolute,
        worst_tensor_residual=worst_tensor,
        angular_factorisation=factorisation,
        alpha_range=[min(r['alpha_gb'] for r in rows),
                     max(r['alpha_gb'] for r in rows)],
        q_range=[min(r['q'] for r in rows), max(r['q'] for r in rows)],
        checks=dict(
            identity_holds=bool(worst_relative < 1e-5),
            signs_agree=bool(same_sign),
            angular_factorisation=bool(factorisation < 1e-12),
            states_pass_tensor_gate=bool(worst_tensor < TENSOR_GATE),
        ),
        versions=environment_report(),
    )
    files = ['experiments/egb_onshell_potential.py',
             'src/rotating_bh/einstein.py',
             'src/rotating_bh/egb_rotating_saved.py',
             'src/rotating_bh/egb_rotating_validation.py',
             'src/rotating_bh/gb_response.py',
             'results/egb-rotating-family.json']
    result['source_sha256'] = {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in dict.fromkeys(files)}
    output = ROOT/'results/egb-onshell-potential.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    manifest_path = ROOT/'artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    record = dict(
        id='egb-onshell-potential', kind='result',
        path=str(output.relative_to(ROOT)).replace('\\', '/'),
        command='python experiments/egb_onshell_potential.py',
        commit='working-tree', inputs=list(result['source_sha256']),
        parameters=dict(nodes=NODES, ladder=list(LADDER), theta=THETA,
                        source_sha256=result['source_sha256']),
        environment='environment/requirements-lock.txt', agent='claude',
        prompt_refs=['prompts.txt'],
        decisions=['Read Psi as an on-shell action integral and test it '
                   'against the implicit response without using the '
                   'asymptotic charge extraction'],
        checks=[dict(name=k, passed=bool(v)) for k, v in result['checks'].items()],
        status='verified' if all(result['checks'].values()) else 'rejected',
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest = [r for r in manifest if r['id'] != record['id']]+[record]
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(result['checks']), flush=True)
    print(f'worst relative {worst_relative:.3e}  '
          f'worst absolute {worst_absolute:.3e}', flush=True)


if __name__ == '__main__':
    run()
