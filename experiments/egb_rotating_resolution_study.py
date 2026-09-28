"""The complete cold sequence and the T=0 extrapolation at every coupling and
three radial resolutions.

The referee's third revision: the manuscript quoted comparison-resolution walks
at three of thirteen couplings, none of them near the valley where the extremal
slope turns, which is too sparse for a thirteen-point curve whose claim is a
change in slope.  This repeats the whole ascending walk and the whole
extrapolation at `N = 48` and `N = 56` at every coupling, so that the production
`N = 64` result can be compared against two coarser ones and a trend read rather
than a single difference.

Three things make the comparison mean what it says:

* The production side is *not* recomputed.  It is `extremal_state` applied to
  the saved states of `results/egb-extremality.json`, which reproduces the
  shipped extremal table to 1.4e-14 absolute.  Only the coarse sides are solved.
* Each coarse walk uses the production convention exactly -- the same start
  spin, the same acceptance gates, the same escalation to `N = 96` when a state
  is refused -- so the only difference is the base resolution.  Anything else
  would compare two conventions rather than two resolutions.
* Because of that escalation the coldest states can end up at the same
  resolution on both sides.  The census below records, for every fitted state on
  every side, the resolution that actually accepted it, so a reader can see how
  much of each sequence genuinely differed instead of assuming all of it did.

Only the ascending branch is walked: `extremal_state` fits the twelve coldest
states, and the descending branch of the production run contributes none of
them.

Run from the project root:  PYTHONPATH=src python3 experiments/egb_rotating_resolution_study.py
"""
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from rotating_bh.extremality import extremal_state, invariants, state_at, walk
from rotating_bh.gb_response import predictor_ladder

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT = ROOT/'results/egb-extremality.json'
OUTPUT = ROOT/'results/egb-rotating-resolution-study.json'
PROGRESS = ROOT/'.cache/resolution-study-progress.json'

# The coarse resolutions to add. 48 is the one the manuscript already reports at
# three couplings, kept so that those three rows stay comparable; 56 is new and
# is what turns a difference into a trend.
COARSE_RESOLUTIONS = (48, 56)
# The shipped measurement was produced on a different linear-algebra stack from
# the one this study runs on, so a coarse-against-shipped difference bounds the
# resolution change *and* the environment change together. Re-walking at the
# production resolution here separates the two. Three couplings are enough to
# size the environment term: one at each end of the range and one in the valley
# where the extremal slope turns.
ENVIRONMENT_CONTROL = (.02, .15, .5)
START_SPIN = .60
COUPLING_STEP = .005
SHAPE = (1.5**1.5)*np.sqrt(np.pi)          # j = SHAPE * J / M^{3/2}


class Progress:
    """Disk memo keyed by what defines a walk, so a rerun costs nothing."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                self.entries = {}

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, value):
        self.entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, allow_nan=False),
                             encoding='utf-8')
        return value


def ascend(alpha_gb, resolution, refined, cache):
    """Climb to the coupling, then walk the spin up to the extremal end."""
    key = f'ascend|{alpha_gb:.12g}|{resolution}|{refined}'
    stored = cache.get(key)
    if stored is not None:
        for state in stored:
            state.update(invariants(state))
        return stored
    started = time.time()
    base = predictor_ladder(START_SPIN, alpha_gb, step=COUPLING_STEP,
                            resolution=resolution)
    states = [state_at(base)]
    states += walk(base, alpha_gb, direction=+1,
                   resolutions=(resolution, refined), floor=1e-6)
    states.sort(key=lambda s: s['omega_h'])
    for state in states:
        state.update(invariants(state))
    cache.put(key, states)
    print(f'  alpha_gb={alpha_gb:<6g} N={resolution}+ {len(states):3d} states, '
          f'T down to {min(s["T_H"] for s in states):.2e}  '
          f'[{time.time()-started:.0f}s]', flush=True)
    return states


def census(states, window=12):
    """How many of the fitted states each resolution actually accepted."""
    cold = sorted(states, key=lambda s: s['T_H'])[:window]
    counts = Counter(int(s['resolution']) for s in cold)
    return {str(n): counts[n] for n in sorted(counts)}


def scaled_spin(record):
    return float(SHAPE*record['J']['value']/record['E']['value']**1.5)


def compare(fine, coarse):
    """Every extrapolated observable, fine against coarse, column by column.

    A relative difference is `None` where the reference vanishes -- `y = 0` at
    `alpha_gb = 0` is the case that occurs -- rather than a NaN, which is not
    representable in JSON and which a reader could mistake for a failed solve.
    """
    def pair(a, b):
        return dict(fine=float(a), coarse=float(b), absolute=float(abs(a-b)),
                    relative=float(abs(a-b)/abs(a)) if a else None)

    row = dict(alpha_gb=fine['alpha_gb'])
    for key in ('E', 'J', 'S', 'psi_gb'):
        row[key] = pair(fine[key]['value'], coarse[key]['value'])
    for key in ('y', 'mu', 'sigma'):
        row[key] = pair(fine[key], coarse[key])
    row['j'] = pair(scaled_spin(fine), scaled_spin(coarse))
    return row


def run():
    measurement = json.loads(MEASUREMENT.read_text(encoding='utf-8'))
    couplings = list(measurement['extremal_couplings'])
    production = int(measurement['resolution'])
    refined = int(measurement['refined_resolution'])
    saved = {float(k): v for k, v in measurement['walks'].items()}

    # The production side, recomputed from the shipped states rather than
    # re-solved, and asserted against the shipped extremal table.
    fine = {}
    worst = 0.
    shipped = {r['alpha_gb']: r for r in measurement['extremals']}
    for coupling in couplings:
        record = extremal_state(saved[coupling])
        fine[coupling] = record
        for key in ('E', 'J', 'S', 'psi_gb'):
            worst = max(worst, abs(record[key]['value']
                                   - shipped[coupling][key]['value']))
    print(f'production side reproduced from saved states to {worst:.2e} '
          f'absolute', flush=True)
    assert worst < 1e-10, 'the saved states do not reproduce the shipped table'

    cache = Progress(PROGRESS)
    rows, walks = [], {}
    for resolution in COARSE_RESOLUTIONS:
        print(f'coarse walks at N={resolution}', flush=True)
        for coupling in couplings:
            states = ascend(coupling, resolution, refined, cache)
            walks[f'{resolution}|{coupling:.12g}'] = states
            record = extremal_state(states)
            row = compare(fine[coupling], record)
            row.update(
                resolution=resolution, production_resolution=production,
                refined_resolution=refined,
                coarse_states=len(states),
                coarse_coldest_temperature=float(min(s['T_H'] for s in states)),
                fine_coldest_temperature=float(
                    min(s['T_H'] for s in saved[coupling])),
                coarse_census=census(states),
                fine_census=census(saved[coupling]),
                worst_relative_tensor_residual=float(
                    record['worst_relative_tensor_residual']),
                worst_smarr=float(record['worst_smarr']))
            rows.append(row)
            print(f'    alpha_gb={coupling:<6g} |dPsi|={row["psi_gb"]["absolute"]:.2e}'
                  f'  |dmu|={row["mu"]["absolute"]:.2e}'
                  f'  |dy|={row["y"]["absolute"]:.2e}', flush=True)

    # The environment control: the production resolution, re-walked here.
    print(f'environment control at N={production}', flush=True)
    control = []
    for coupling in ENVIRONMENT_CONTROL:
        states = ascend(coupling, production, refined, cache)
        walks[f'{production}|{coupling:.12g}'] = states
        row = compare(fine[coupling], extremal_state(states))
        row.update(resolution=production, production_resolution=production,
                   coarse_states=len(states),
                   coarse_coldest_temperature=float(min(s['T_H'] for s in states)),
                   fine_coldest_temperature=float(
                       min(s['T_H'] for s in saved[coupling])),
                   coarse_census=census(states),
                   fine_census=census(saved[coupling]))
        control.append(row)
        print(f'    alpha_gb={coupling:<6g} |dPsi|={row["psi_gb"]["absolute"]:.2e}'
              f'  |dmu|={row["mu"]["absolute"]:.2e}  (same N, other stack)',
              flush=True)
    worst_control = max(r['psi_gb']['absolute'] for r in control)
    print(f'worst |dPsi_ext| attributable to the environment alone: '
          f'{worst_control:.2e}', flush=True)

    # Does the difference shrink as the coarse resolution rises? One answer per
    # coupling per observable, so that no single worst case stands for the set.
    trend = []
    low, high = COARSE_RESOLUTIONS[0], COARSE_RESOLUTIONS[-1]
    for coupling in couplings:
        first = next(r for r in rows
                     if r['alpha_gb'] == coupling and r['resolution'] == low)
        second = next(r for r in rows
                      if r['alpha_gb'] == coupling and r['resolution'] == high)
        entry = dict(alpha_gb=coupling)
        for key in ('E', 'J', 'S', 'psi_gb', 'y', 'mu', 'sigma', 'j'):
            a, b = first[key]['absolute'], second[key]['absolute']
            # `None` rather than an infinity where the finer side is exactly
            # equal: JSON has no infinity, and a reader should see "the
            # difference vanished" rather than a sentinel number.
            entry[key] = dict(low=a, high=b, shrinks=bool(b <= a),
                              ratio=float(a/b) if b else None)
        trend.append(entry)
    shrinking = sum(1 for e in trend if e['psi_gb']['shrinks'])
    print(f'the difference in Psi_ext shrinks from N={low} to N={high} at '
          f'{shrinking} of {len(trend)} couplings', flush=True)

    worst_psi = max(r['psi_gb']['absolute'] for r in rows)
    worst_mu = max(r['mu']['absolute'] for r in rows)
    print(f'worst |dPsi_ext| over all couplings and resolutions: {worst_psi:.2e}',
          flush=True)
    print(f'worst |dmu| over all couplings and resolutions:      {worst_mu:.2e}',
          flush=True)

    sources = [Path(__file__),
               ROOT/'src/rotating_bh/extremality.py',
               ROOT/'src/rotating_bh/gb_response.py',
               ROOT/'src/rotating_bh/egb_rotating_predictor.py',
               ROOT/'src/rotating_bh/egb_rotating_bvp.py',
               ROOT/'src/rotating_bh/egb_rotating_observables.py',
               ROOT/'environment/requirements-lock.txt']
    sources += sorted((ROOT/'src/rotating_bh').glob('_egb_rotating*_generated.py'))

    evidence = dict(
        schema=1,
        units='G_5 = r_H = 1; J is each of the two equal angular momenta',
        scope='radial-resolution study of the T=0 extrapolation, all couplings',
        production_resolution=production, refined_resolution=refined,
        coarse_resolutions=list(COARSE_RESOLUTIONS),
        start_spin=START_SPIN, coupling_step=COUPLING_STEP,
        extremal_couplings=couplings,
        production_reproduced_to=float(worst),
        comparisons=rows, trend=trend,
        environment_control=control,
        environment_control_couplings=list(ENVIRONMENT_CONTROL),
        worst_environment_difference=float(worst_control),
        shipped_versions=measurement.get('versions'),
        worst_psi_difference=float(worst_psi),
        worst_mu_difference=float(worst_mu),
        psi_shrinking_couplings=int(shrinking),
        walks={key: value for key, value in walks.items()},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, platform=platform.platform()),
        source_sha256={str(p.relative_to(ROOT)):
                       hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources})
    OUTPUT.write_text(json.dumps(evidence, indent=1, allow_nan=False)+'\n',
                      encoding='utf-8')
    print(f'wrote {OUTPUT.relative_to(ROOT)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(run())
