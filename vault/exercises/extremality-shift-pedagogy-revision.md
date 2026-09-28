# Extremality-shift paper: pedagogy and novelty revision

Second revision of `manuscript/main.tex`, on a prompt asking for four things: say what is
new relative to arXiv:2303.12471, add background and pedagogy to abstract and
introduction, justify computing `dM_ext/dalpha` rather than `M_ext` itself, and interpret
the positive slope and its minimum. Constraint: change no existing number, fit or
precision claim. Continues [[extremality-shift-paper-revision]].

## The novelty question, answered honestly

arXiv:2303.12471 (Kleihaus, Kunz & Radu) *does* construct extremal EGB black holes
directly and *does* compute their mass: eq. (3.15) there gives `M`, `J`, `A_H`, `S` from
the numerical output. So the first draft of my novelty claim — "they did not extract
`M_ext(J, alpha)`" — was too strong, and I had to weaken it after checking.

Scale invariance makes the extremal branch a single curve in their reduced variables, and
that curve already fixes `mu(y)`. With `mu = M_ext/J^(2/3)` and `y = alpha/J^(2/3)`, the
definitions alone give

    mu = mu(0) * j^(-2/3),      y = 4*x*mu/(3*pi),

verified against our own table to the printed digits at all 13 couplings. The published
domain of existence therefore *contains* the extremal mass at fixed `J`.

What it does not contain is the derivative. That is the defensible novelty, and the paper
now says so: we measure `dM_ext/dalpha|_J` as a thermodynamic response at each coupling —
one linear solve, no step size, no second solution — rather than differencing masses of
separately constructed extremal solutions, and the minimum at `y ~ 0.175` is resolved
because the sampling is a grid in `alpha`, not a plotted boundary.

## What was verified rather than asserted

**`Psi` as an on-shell action.** The prompt asked for the statement
`Psi = -(1/16 pi) Int_Sigma sqrt(-g) L_GB d^4x`. Rather than assert it I tested it:
`experiments/egb_onshell_potential.py`, 43 archived spectral profiles,
`alpha` in `[0.0025, 0.75]`, `q` in `[0.1, 0.4]`.

    worst relative 4.674e-07   worst absolute 4.014e-08

against the implicit response. This route uses no Jacobian, no linear solve, no parameter
derivative and — unlike the Smarr route — *no asymptotic charge extraction*; it reads
`Psi` off the curvature. It is the strongest independent check the paper now has.

Two things the run had to pin down rather than assume. The integrand factorises,
`sqrt(-g) L_GB = sin(theta) cos(theta) D(r)`, checked at four polar angles, so the angular
integral is exactly `2 pi^2`. And the radial quadrature **cannot be refined**: `L_GB`
cancels to `O(r^-8)` while `dr/dchi = (1-chi)^-2` amplifies, so Gauss-Legendre nodes
pushed towards infinity destroy the answer. Measured, at `alpha=0.1, q=0.4`:

    nodes   80: 1.6e-09     400: 1.7e-05     1600: 1.3e-01
    nodes  200: 2.6e-07     800: 6.3e-04     3200: 3.8e+00

80 nodes is the production choice; the ladder is stored with the result. See
[[quadrature-that-cannot-be-refined]].

**The slope minimum is a maximum of `Omega_H J^(1/3)`.** Smarr at `T=0` gives
`alpha Psi_ext = M_ext - 3 Omega_H J`, hence with `omega = Omega_H J^(1/3)`

    3 omega(y) = mu(y) - y mu'(y),     3 omega'(y) = -y mu''(y),

so for `y > 0` the inflection of the extremal mass is exactly the stationary point of
`omega`, and a minimum of the slope is a *maximum* of `omega`. Tested by extrapolating
`omega` on its own from the same walks with the production cubic: identity satisfied to
`5.74e-6`, maximum `omega = 0.762` at `y = 0.175` — the same sampled coupling at which
`Psi_ext` is smallest. At `y = 0` the extrapolant returns `0.732296` against the analytic
extremal Myers-Perry value `pi^(1/3)/2`, to `1.16e-13`, which nothing in the fit imposed.

**The slope has to rise again.** `M > 3 pi alpha/4` (arXiv:2303.12471, holding for the
rotating solutions too) is `mu(y) > 3 pi y / 4`. If `mu' <= c < 3pi/4` everywhere then
`mu(y) <= mu(0) + c y`, which violates the bound for `y > mu(0)/(3pi/4 - c)`. So `mu'`
must return to at least `3pi/4 = 2.356`, above every value measured. The observed rise
from `1.07` to `1.44` is the start of that, and if their endpoint conjecture holds the
slope tends to `3pi/4`. Our largest extremal `x` is `0.4170` against the conjectured
endpoint at `x = 1`, so this is consistency, not evidence.

**Positivity restated.** `j_ext = [mu(0)/mu(y)]^(3/2)`, satisfied by the separately
extrapolated `j` and `mu` to `3.11e-11`; so a positive slope *is* the statement `j_ext < 1`
that arXiv:2303.12471 reports as a bound. And `Psi_ext > 0` *is* `M_ext > 3 Omega_H J`.

## A defect I introduced and caught

Rewriting the introduction deleted the display equation carrying `\label{eq:firstlaw}`,
which seven later `\eqref`s point at. Tectonic exited 0 and printed no undefined-reference
warning; only an explicit label/ref cross-check found it. Restored into section 2.2, where
the law is first used. `check_manuscript.py` does not check labels — it checks macros,
literal decimals and citations. See [[latex-checks-that-do-not-fire]].

## New content in the paper

- Figure 4 (`figures/fig-massext.pdf`): `mu` vs `y` with the `O(alpha)` line, and the same
  branch in the `(x, j)` plane with the published bounds and conjectured endpoint.
- Section 5, "The potential as an on-shell action".
- Section 6.1, "Interpretation", with the four readings above; section 6.2 "Limitations"
  now carries every caveat that was scattered through the text.
- Appendix A: the complex-step recipe and the conditioning discussion moved out of the
  body, plus "The on-shell integral".

## Run receipt

    date: 2026-09-28
    env:  .venv, Python 3.12.3, NumPy 1.26.4, SciPy 1.11.4  (not the locked 3.11 stack;
          see [[environment-drift-in-recorded-measurements]] -- the on-shell check is a
          fresh measurement, so no recorded value was re-derived under a different stack)

    $ PYTHONPATH=src .venv/bin/python experiments/egb_onshell_potential.py
    {"identity_holds": true, "signs_agree": true, "angular_factorisation": true,
     "states_pass_tensor_gate": true}
    worst relative 4.674e-07  worst absolute 4.014e-08

    $ PYTHONPATH=src .venv/bin/python manuscript/make_figures.py
    wrote numbers.tex with 143 macros        (+19 new, 0 changed)
    wrote figures/fig-massext.pdf  29301 bytes

    $ .venv/bin/python manuscript/check_manuscript.py
    ok: 147 generated quantities, 20 references, no literal numbers in the body

    $ tectonic -X compile manuscript/main.tex
    Writing `manuscript/main.pdf` (28 pages)

    $ .venv/bin/python checks/verify_recorded_digests.py
    OK: available inputs match, apart from the explicitly listed exceptions

    $ PYTHONPATH=src .venv/bin/python -m pytest -q
    1 failed, 494 passed, 2 skipped in 485.84s
    FAILED tests/test_distribution.py::test_published_distribution_is_intact

That failure was `artifacts/distribution-sha256.json`, the release snapshot that
`checks/verify_distribution.py` deliberately never updates. It was already failing before
this work: `prompts.txt` was modified in the working tree, and its hash alone breaks the
inventory. Refreshed afterwards, on request, in a separate commit — see
[[refreshing-the-distribution-snapshot]]. The audit then reports 237 files and 0
failures, and the full suite 495 passed, 2 skipped, 0 failed.

`tests/test_manuscript.py` needed one edit: its fixture copies a fixed list of inputs into
a temp tree before running `make_figures.py`, and the new `results/egb-onshell-potential.json`
had to be added to that list.

## Not done

- The prompt mentions a PINN result by Garraffo et al. It is not in `papers/`, so it is
  not cited. Same for the Giribet-Garraffo Lovelock review and Emparan-Reall, which the
  prompt named as models of style, not as citations.
- The on-shell check covers `q <= 0.4`. The near-extremal states that carry the `T -> 0`
  extrapolation store no spectral coefficients, so the identity is **not** tested at the
  extremal end.
