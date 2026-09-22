"""Hito 6: the Gauss-Bonnet conjugate potential of the rotating EGB family.

Hito 4C verified the first law at fixed coupling. This measures the charge
conjugate to the coupling itself, which neither benchmark of this project
computes, and asks one falsifiable question: the potential's sign change sits at
q = 0.6349 perturbatively -- does that locus move at finite coupling?

Three things are set up so the answer can be "no", or "the premise was wrong":

* The perturbative formula being tested is recorded in
  `reports/originality-precheck.md` as unvalidated. Its gate is allowed to fail;
  if the measured alpha -> 0 limit disagrees, the artifact reports a refutation.
* The sign-change locus is only recorded when a genuine change of sign brackets
  it. Interpolating between same-signed points would manufacture one.
* Psi_GB comes from central differences whose step is refined, and a value that
  does not settle is not accepted as a value.

The existence of a Gauss-Bonnet conjugate potential and of its perturbative sign
change are **not new** -- see the precheck for antecedents. What is offered here
is numerical evidence at finite coupling in the rotating sector, under this
project's own independent acceptance gates.

Run with PYTHONPATH=src from project/.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import brentq

from rotating_bh.egb_rotating_observables import measure
from rotating_bh.gb_potential import (BOUNDARY_GATE, Cache, TENSOR_GATE, VacuumSeed,
                                      accepted_solution, perturbative_potential,
                                      perturbative_zero, potential, reach,
                                      smarr_residual)
from rotating_bh.gb_potential_validation import checks as gb_checks
from rotating_bh.myers_perry import MyersPerry

ROOT = Path(__file__).resolve().parents[1]

SPINS = [.3, .5, .6, .65]
# The sign-change search needs spins the grid does not: the perturbative zero
# sits at q = 0.6349, and at finite coupling the potential is still negative
# there, so the locus has to be chased upward. The solver stops converging above
# about q = 0.68 at this coupling, which bounds how far it can be chased.
SIGN_CHANGE_SPINS = [.3, .5, .6, .65, .66, .68]
# The central-difference step is capped at the ladder half-step: alpha_gb/4
# alone reaches 0.05 at the largest coupling, and a jump that size from the
# central solution is rejected by the solver even though the ladder climbs past
# it without trouble.
DIFFERENCE_CAP = .01
# Completed points survive an interruption, so a rerun resumes rather than
# restarting an hour of work. Intermediate state, not evidence: `results/` is
# where the evidence lives.
PROGRESS = '.cache/hito-6-progress.json'
COUPLINGS = [.02, .06, .1, .2]   # every one a multiple of LADDER
LADDER = .02                      # continuation step used to climb to a coupling
VANISHING = [.02, .04, .06, .08]  # the small-coupling ladder extrapolated to zero


def climb(omega_h, couplings, step=LADDER, resolution=32, cache=None):
    """Continue upward once, returning the accepted solution at each coupling.

    Every coupling requested must be a multiple of the ladder step, so the climb
    lands on it exactly rather than near it.
    """
    wanted = sorted(couplings)
    for value in wanted:
        if abs(value/step - round(value/step)) > 1e-9:
            raise ValueError(f'coupling {value} is not on the {step} ladder')
    # Start from the closed-form vacuum: above about q = 0.55 the solver's
    # trivial guess no longer converges at alpha_gb=0, which is exactly the rung
    # where a closed-form answer is available.
    bases, previous = {}, VacuumSeed(omega_h)
    for index in range(round(max(wanted)/step)+1):
        value = round(index*step, 10)
        previous, _ = accepted_solution(omega_h, value, previous, resolution, cache)
        for target in wanted:
            if abs(target-value) < 1e-12:
                bases[target] = previous
    return bases


def vacuum_smarr():
    """2E = 3TS + 6 Omega J against the Hito 3 closed forms.

    This fixes the Smarr coefficients rather than assuming them, and it is the
    one place in this experiment with an exact answer to check against.
    """
    rows = []
    for omega_h in [.1, .2, .3, .33, .4, .5, .6]:
        thermo = MyersPerry(r_h=1., omega_h=omega_h).thermodynamics()
        observables = dict(E=thermo['M'], J=thermo['J1'], T_H=thermo['T_H'],
                           S=thermo['S'])
        residual = smarr_residual(observables, omega_h, 0., 0.)
        rows.append(dict(omega_h=omega_h, **residual))
    return rows


def cached_bases(omega_h, couplings, steps, cache, resolution=32):
    """Climb only if something is actually missing.

    The ladder is the expensive half, so resuming a point whose value is already
    stored must not pay for it again.
    """
    if cache is not None and all(
            cache.get(Cache.key(omega_h, a, s, resolution)) is not None
            for a, s in zip(couplings, steps)):
        return None
    return climb(omega_h, couplings, step=LADDER, resolution=resolution, cache=cache)


def step_refinement(omega_h, alpha_gb, steps, cache=None):
    """Psi_GB at a shrinking sequence of difference steps."""
    bases = cached_bases(omega_h, [alpha_gb]*len(steps), steps, cache)
    base = bases[alpha_gb] if bases else None
    ladder = []
    for step in steps:
        measured = potential(omega_h, alpha_gb, step, base=base, cache=cache)
        ladder.append(dict(step=step, psi_gb=measured['psi_gb'],
                           max_tensor_residual=measured['max_tensor_residual']))
    return dict(omega_h=omega_h, alpha_gb=alpha_gb, ladder=ladder)


def vanishing_coupling(omega_h, cache=None):
    """Extrapolate Psi_GB to alpha_GB = 0 and compare against the oracle.

    A quadratic fit in the coupling, with the residual of a linear fit reported
    alongside so the extrapolation's own uncertainty is visible rather than
    implied.
    """
    values = []
    steps = [min(a/4, DIFFERENCE_CAP) for a in VANISHING]
    bases = cached_bases(omega_h, VANISHING, steps, cache)
    for alpha_gb in VANISHING:
        measured = potential(omega_h, alpha_gb, min(alpha_gb/4, DIFFERENCE_CAP), base=bases[alpha_gb] if bases else None, cache=cache)
        values.append((alpha_gb, measured['psi_gb']))
    couplings = np.array([a for a, _ in values])
    potentials = np.array([p for _, p in values])
    quadratic = np.polyfit(couplings, potentials, 2)
    linear = np.polyfit(couplings, potentials, 1)
    limit = float(np.polyval(quadratic, 0.))
    spread = abs(limit - float(np.polyval(linear, 0.)))
    oracle = float(perturbative_potential(omega_h))
    return dict(omega_h=omega_h, samples=[dict(alpha_gb=a, psi_gb=p) for a, p in values],
                extrapolated=limit, extrapolation_spread=spread,
                oracle=oracle, absolute_deviation=abs(limit-oracle),
                relative_deviation=abs(limit-oracle)/abs(oracle))


def sign_change(alpha_gb, spins, cache=None):
    """Bracket where Psi_GB changes sign at fixed coupling, if it does."""
    samples = []
    for omega_h in spins:
        bases = cached_bases(omega_h, [alpha_gb], [min(alpha_gb/4, DIFFERENCE_CAP)], cache)
        base = bases[alpha_gb] if bases else None
        measured = potential(omega_h, alpha_gb, min(alpha_gb/4, DIFFERENCE_CAP), base=base, cache=cache)
        samples.append(dict(q=omega_h, psi_gb=measured['psi_gb']))
    bracket = None
    for low, high in zip(samples, samples[1:]):
        if low['psi_gb']*high['psi_gb'] < 0:
            bracket = (low, high)
            break
    if bracket is None:
        return dict(alpha_gb=alpha_gb, samples=samples, bracketed=False)

    def at(omega_h):
        step = min(alpha_gb/4, DIFFERENCE_CAP)
        bases = cached_bases(float(omega_h), [alpha_gb], [step], cache)
        base = bases[alpha_gb] if bases else None
        return potential(float(omega_h), alpha_gb, min(alpha_gb/4, DIFFERENCE_CAP), base=base, cache=cache)['psi_gb']

    # Every evaluation costs a continuation ladder and several minutes, so the
    # locus is located to about five milli-units in q. The quantity being tested
    # is a shift of roughly 0.035 away from the perturbative zero at 0.6349, so
    # this resolves it several times over; chasing more digits would buy nothing
    # and cost a great deal. The bracket itself is the hard evidence and is
    # recorded alongside.
    q_star = float(brentq(at, bracket[0]['q'], bracket[1]['q'], xtol=5e-3))
    return dict(alpha_gb=alpha_gb, samples=samples, bracketed=True,
                bracket_low=bracket[0], bracket_high=bracket[1], q_star=q_star)


def run():
    cache = Cache(ROOT/PROGRESS)
    vacuum = vacuum_smarr()
    print(f'vacuum Smarr: worst relative {max(r["relative"] for r in vacuum):.2e}',
          flush=True)

    # Refine where it is hard, not only where it is easy. An earlier version
    # sampled only the well-behaved low-spin points, which is exactly the wrong
    # place to look for trouble: the first run showed Psi_GB behaving
    # non-monotonically in the coupling at q >= 0.6, with Smarr residuals an
    # order of magnitude larger there than at q = 0.3.
    refinement = [step_refinement(.3, .1, [.02, .01, .005], cache),
                  step_refinement(.5, .06, [.015, .0075, .00375], cache),
                  step_refinement(.65, .06, [.015, .0075, .00375], cache),
                  step_refinement(.65, .2, [.02, .01, .005], cache)]
    print('step refinement done', flush=True)

    # And check the answer does not depend on the spectral resolution, at the
    # points where the measurement is weakest.
    resolution_check = []
    for omega_h, alpha_gb in [(.5, .06), (.5, .2), (.65, .06), (.65, .2)]:
        step = min(alpha_gb/4, DIFFERENCE_CAP)
        coarse_base = climb(omega_h, [alpha_gb], step=LADDER, resolution=32,
                            cache=cache)[alpha_gb]
        coarse = potential(omega_h, alpha_gb, step, resolution=32, cache=cache,
                           base=coarse_base)
        # Seed the fine resolution from the coarse solution rather than climbing
        # a fresh N=40 ladder from the vacuum, which fails at q = 0.65. This is
        # the same route Hito 4C used for its own N40 cross-check.
        fine_base, _ = accepted_solution(omega_h, alpha_gb, coarse_base,
                                         resolution=40, cache=cache)
        fine = potential(omega_h, alpha_gb, step, resolution=40, cache=cache,
                         base=fine_base)
        resolution_check.append(dict(
            omega_h=omega_h, alpha_gb=alpha_gb,
            psi_gb_n32=coarse['psi_gb'], psi_gb_n40=fine['psi_gb'],
            absolute_difference=abs(coarse['psi_gb']-fine['psi_gb'])))
        print(f"  resolution q={omega_h} alpha_gb={alpha_gb}: "
              f"|N32-N40| = {resolution_check[-1]['absolute_difference']:.3e}", flush=True)

    points = []
    for omega_h in SPINS:
        # One ladder per spin, picking up each coupling as it passes. Climbing
        # from zero for every point separately would repeat the same solves
        # dozens of times over for identical results.
        steps = [min(a/4, DIFFERENCE_CAP) for a in COUPLINGS]
        bases = cached_bases(omega_h, COUPLINGS, steps, cache)
        for alpha_gb in COUPLINGS:
            base = bases[alpha_gb] if bases else None
            measured = potential(omega_h, alpha_gb, min(alpha_gb/4, DIFFERENCE_CAP),
                                 base=base, cache=cache)
            # `base` is None whenever the ladder was skipped because everything
            # it would feed was already cached, so take the central observables
            # from the measurement itself rather than re-deriving them.
            observables = measured['observables']
            points.append(dict(omega_h=omega_h, alpha_gb=alpha_gb,
                               alpha_paper=4*alpha_gb, step=measured['step'],
                               psi_gb=measured['psi_gb'],
                               derivatives=measured['derivatives'],
                               observables=observables,
                               max_tensor_residual=measured['max_tensor_residual'],
                               smarr=smarr_residual(observables, omega_h, alpha_gb,
                                                    measured['psi_gb']),
                               perturbative=float(perturbative_potential(omega_h))))
            print(f"  q={omega_h} alpha_gb={alpha_gb}: Psi={measured['psi_gb']:.6f} "
                  f"smarr={points[-1]['smarr']['relative']:.1e}", flush=True)

    limits = [vanishing_coupling(q, cache) for q in SPINS]
    print('vanishing-coupling limits done', flush=True)

    loci = []
    for alpha_gb in [.02, .1, .2]:
        located = sign_change(alpha_gb, SIGN_CHANGE_SPINS, cache)
        loci.append(located)
        print(f"  sign change at alpha_gb={alpha_gb}: "
              f"{located.get('q_star', 'not bracketed')}", flush=True)

    evidence = dict(vacuum_smarr=vacuum, step_refinement=refinement,
                    resolution_check=resolution_check, points=points,
                    vanishing_coupling=limits,
                    sign_change=[l for l in loci if l.get('bracketed')],
                    sign_change_attempts=loci,
                    perturbative_zero=perturbative_zero(),
                    spins=SPINS, couplings=COUPLINGS,
                    sign_change_spins=SIGN_CHANGE_SPINS,
                    tensor_gate=TENSOR_GATE, boundary_gate=BOUNDARY_GATE)
    gates = gb_checks(evidence)

    sources = [Path(__file__),
               ROOT/'src/rotating_bh/gb_potential.py',
               ROOT/'src/rotating_bh/gb_potential_validation.py',
               ROOT/'src/rotating_bh/egb_rotating_bvp.py',
               ROOT/'src/rotating_bh/egb_rotating_observables.py',
               ROOT/'src/rotating_bh/egb_rotating_validation.py',
               ROOT/'src/rotating_bh/myers_perry.py',
               ROOT/'environment/requirements-lock.txt']
    sources += sorted((ROOT/'src/rotating_bh').glob('_egb_rotating*_generated.py'))

    data = dict(schema=1,
        units='r_H=G5=1; J is each spin; alpha_paper = 4 alpha_gb; Psi_GB = 4 Psi_paper',
        scope='Gauss-Bonnet conjugate potential of the rotating EGB family, measured '
              'from the extended first law at fixed r_H and Omega_H and cross-checked '
              'against the Smarr scaling. Neither benchmark of this project computes '
              'it. The existence of this potential and of its perturbative sign change '
              'are NOT new -- see reports/originality-precheck.md for antecedents. No '
              'claim of publishability is made.',
        **evidence, checks=gates,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__),
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    output = ROOT/'results/egb-rotating-gb-potential.json'
    output.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

    manifest_path = ROOT/'artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    record = dict(id='egb-rotating-gb-potential', kind='result',
        path=str(output.relative_to(ROOT)).replace('\\', '/'),
        command='python experiments/egb_rotating_gb_potential.py',
        commit='working-tree', inputs=list(data['source_sha256']),
        parameters=dict(source_sha256=data['source_sha256'], spins=SPINS,
                        couplings=COUPLINGS, sign_change_spins=SIGN_CHANGE_SPINS,
                        continuation_step=LADDER,
                        difference_step_cap=DIFFERENCE_CAP,
                        vanishing_ladder=VANISHING),
        environment='environment/requirements-lock.txt', agent='claude',
        prompt_refs=['prompts/prompt-log.md'],
        decisions=[
            'Smarr coefficients fixed by the exact alpha_gb=0 limit, not assumed',
            'The perturbative oracle is treated as a hypothesis and its gate may fail',
            'A sign-change locus is recorded only when a real change of sign brackets it',
            'Existence of the potential and of its sign change are not claimed as new'],
        checks=gates,
        status='verified' if all(c['passed'] for c in gates) else 'candidate',
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest = [r for r in manifest if r['id'] != record['id']]+[record]
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')

    for gate in gates:
        print(f"{gate['name']:<42} {'pass' if gate['passed'] else 'FAIL'}")
    return 0 if all(c['passed'] for c in gates) else 1


if __name__ == '__main__':
    raise SystemExit(run())
