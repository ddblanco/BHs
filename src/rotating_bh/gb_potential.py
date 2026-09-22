"""The Gauss-Bonnet conjugate potential for the rotating EGB family (Hito 6).

Hito 4C verified the first law at *fixed* coupling, varying the spin. Extending
it to variations of the coupling needs the conjugate charge,

    Psi_GB = dE/dalpha_GB - T_H dS/dalpha_GB - 2 Omega_H dJ/dalpha_GB

at fixed `r_H` and `Omega_H`, with `J` each angular momentum. Neither benchmark
of this project computes it.

Two things keep the measurement honest. The derivatives come from central
differences whose step is refined, so a value that does not converge is not a
value. And the same observables feed a second relation, the Smarr scaling

    2E = 3 T_H S + 6 Omega_H J + 2 alpha_GB Psi_GB

which shares inputs with the first -- so its agreement is consistency, not
independent confirmation, while its failure would be decisive. At `alpha_GB=0`
that relation reduces to one the Hito 3 closed forms settle exactly, which is
what fixes its coefficients rather than assuming them.

Nothing here touches the solver, the adaptive method, the Hito 4A derivation or
the generated equations.
"""
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct

from .egb_rotating_bvp import solve
from .egb_rotating_observables import measure
from .egb_rotating_saved import SavedProfile
from .egb_rotating_validation import diagnose
from .neural_seed import MyersPerryAnchor

__all__ = ['TENSOR_GATE', 'BOUNDARY_GATE', 'perturbative_potential',
           'perturbative_zero', 'accepted_solution', 'reach', 'potential',
           'smarr_residual', 'VacuumSeed', 'Cache']


TENSOR_GATE, BOUNDARY_GATE = 1e-6, 1e-8


class Cache:
    """Disk memo for measured potentials, keyed by the parameters that define
    them.

    Each `potential` call costs a continuation ladder plus two neighbour solves,
    and the full measurement runs for the better part of an hour before it has
    anything to write. An interruption at minute fifty then loses everything.
    This keeps completed points so a rerun resumes instead of restarting, the
    same way `egb_rotating_family` checkpoints its sweep.

    It stores measured numbers, never acceptance decisions: everything reloaded
    was produced by a run that had already passed the Hito 4 gates, and the gates
    are recomputed from the evidence regardless.
    """

    def __init__(self, path):
        self.path = Path(path) if not hasattr(path, 'read_text') else path
        self._entries = {}
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                self._entries = {}

    @staticmethod
    def key(*parts):
        return '|'.join(f'{p:.12g}' if isinstance(p, float) else str(p) for p in parts)

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        self._entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=1, allow_nan=False),
                             encoding='utf-8')
        return value


class VacuumSeed:
    """The closed-form Myers-Perry solution, as a seed for the alpha_gb=0 rung.

    Every ladder here starts at the vacuum, and above about q = 0.55 the solver's
    trivial guess no longer converges there -- it exhausts fifty iterations at a
    residual of order ten. The vacuum is exactly where an answer is already known
    in closed form, so the ladder starts from it instead of from a guess. The
    anchor is the one Hito 5 verified against this same solver at alpha_gb=0.
    """

    def __init__(self, omega_h):
        self._anchor = MyersPerryAnchor(omega_h)

    def evaluate(self, x, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1 or 2')
        return self._anchor(np.asarray(x, dtype=float))[derivative]


def perturbative_potential(q):
    """The oracle proposed in `reports/originality-precheck.md`.

    Recorded there as explicitly unvalidated, so it is a hypothesis to be
    tested by the measured `alpha_GB -> 0` limit, never a reference to fit to.
    """
    q = np.asarray(q, dtype=float)
    return -(np.pi/4)*(q**4/(1-q*q)**2 - 14*q*q/(1-q*q) + 9)


def perturbative_zero():
    """Where that oracle changes sign: q^2 = 9/(16 + 2 sqrt(10))."""
    return float(np.sqrt(9/(16+2*np.sqrt(10))))


def accepted_solution(omega_h, alpha_gb, previous=None, resolution=32, cache=None):
    """Solve and refuse to return anything failing the unchanged Hito 4 gates.

    A cached profile for this exact point is used as the *starting guess* only,
    never as the answer. The solve and the gates run regardless -- from an exact
    seed the Newton converges in an iteration or two, so resuming is nearly free
    without the result ever being something loaded from disk. That keeps the
    project's rule that loading a stored profile is not accepting it.
    """
    key = Cache.key('profile', omega_h, alpha_gb, resolution) if cache else None
    if key is not None:
        # Prefer a stored profile for this exact point over whatever seed the
        # caller supplied: it is the closer guess by construction, so the solve
        # converges in an iteration or two. Consulting the cache only when no
        # seed was given made every explicitly seeded call re-solve from scratch,
        # which is the difference between three seconds and three minutes.
        stored = cache.get(key)
        if stored is not None:
            previous = SavedProfile(stored)
    solution = solve(omega_h, alpha_gb, resolution=resolution, tol=1e-11,
                     previous=previous, max_iterations=50)
    checked = diagnose(solution, alpha_gb, count=51)
    if not (checked['max_tensor_residual'] < TENSOR_GATE
            and solution.boundary_residual < BOUNDARY_GATE):
        raise RuntimeError(f'unaccepted solution at omega_h={omega_h}, '
                           f'alpha_gb={alpha_gb}: {checked}')
    if key is not None and cache.get(key) is None:
        coefficients = dct(solution.evaluate(solution.nodes), type=1, axis=1)/(resolution-1)
        coefficients[:, [0, -1]] *= .5
        cache.put(key, dict(q=omega_h, alpha_gb=alpha_gb, N=resolution,
                            chebyshev_coefficients=coefficients.tolist()))
    return solution, checked


def reach(omega_h, alpha_gb, step=.02, resolution=32, cache=None):
    """Continue up to a coupling, seeding each step from the previous solution.

    Continuation is used upward only. Stepping *down* towards alpha_gb=0 from a
    nearby solution fails in this system -- the same effect recorded in the 4C
    closure addendum, where the solver's trivial guess beats a neighbouring
    solution -- so every ladder starts at alpha_gb=0 and climbs.
    """
    if alpha_gb < 0:
        raise ValueError('alpha_gb must be non-negative')
    # An arange ladder stops short whenever the target is not a multiple of the
    # step -- it would return the solution at the last rung and call it the
    # target, silently. Space the rungs so the last one *is* the target.
    rungs = (np.linspace(0., alpha_gb, int(np.ceil(alpha_gb/step))+1)
             if alpha_gb > 0 else np.array([0.]))
    previous = VacuumSeed(omega_h)
    for value in rungs:
        # Each rung consults the cache itself, so an interruption costs at most
        # the single solve it was in the middle of rather than the whole ladder.
        stored = (cache.get(Cache.key('profile', omega_h, float(value), resolution))
                  if cache is not None else None)
        seed = None if stored is not None else previous
        previous, _ = accepted_solution(omega_h, float(value), seed, resolution, cache)
    return previous


def potential(omega_h, alpha_gb, step, base=None, resolution=32, cache=None):
    """Psi_GB by central differences in the coupling, at fixed r_H and Omega_H.

    `step` must leave both neighbours strictly positive: alpha_gb=0 is reached
    only by the upward ladder, never as a difference neighbour.
    """
    if not 0 < step < alpha_gb:
        raise ValueError('step must satisfy 0 < step < alpha_gb')
    memo = cache.key(omega_h, alpha_gb, step, resolution) if cache is not None else None
    if memo is not None:
        stored = cache.get(memo)
        if stored is not None:
            return stored
    if base is None:
        base = reach(omega_h, alpha_gb, resolution=resolution, cache=cache)
    centre = measure(base)
    neighbours, worst, reclimbed = {}, 0., []
    for sign in (-1, 1):
        target = alpha_gb+sign*step
        try:
            solution, checked = accepted_solution(omega_h, target, base, resolution,
                                                  cache)
        except Exception:
            # Seeding a neighbour from the central solution is a jump of `step`,
            # and this system rejects jumps the upward ladder handles fine. Fall
            # back to climbing to the neighbour, which reaches the same branch
            # through the same gates; record that it happened.
            # Reach the neighbour at the coarse resolution first and use it as
            # the seed. A fine ladder from the vacuum fails at high spin, while
            # the coarse one converges and is usually already cached, so this is
            # both the reliable route and the cheap one. The gates still decide.
            coarse, _ = accepted_solution(
                omega_h, target, reach(omega_h, target, resolution=32, cache=cache),
                32, cache)
            solution, checked = (coarse, _) if resolution == 32 else                 accepted_solution(omega_h, target, coarse, resolution, cache)
            reclimbed.append(float(target))
        worst = max(worst, checked['max_tensor_residual'])
        neighbours[sign] = measure(solution)
    derivative = {key: (neighbours[1][key]-neighbours[-1][key])/(2*step)
                  for key in ('E', 'S', 'J')}
    psi = (derivative['E'] - centre['T_H']*derivative['S']
           - 2*omega_h*derivative['J'])
    result = dict(omega_h=omega_h, alpha_gb=alpha_gb, step=step,
                  psi_gb=float(psi), derivatives=derivative, observables=centre,
                  max_tensor_residual=float(worst), reclimbed_neighbours=reclimbed)
    return cache.put(memo, result) if memo is not None else result


def smarr_residual(observables, omega_h, alpha_gb, psi_gb):
    """Residual of 2E = 3 T_H S + 6 Omega_H J + 2 alpha_GB Psi_GB.

    At alpha_gb=0 the last term drops and what remains is checked against the
    Hito 3 closed forms, which is what pins the coefficients down.
    """
    left = 2*observables['E']
    right = (3*observables['T_H']*observables['S'] + 6*omega_h*observables['J']
             + 2*alpha_gb*psi_gb)
    return dict(left=float(left), right=float(right),
                absolute=float(left-right),
                relative=float(abs(left-right)/abs(left)))
