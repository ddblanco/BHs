# Extremality-shift paper revision

Acting on `prompts.txt` and the referee report in [[extremality-shift-paper-review]]:
audit, then rewrite `manuscript/main.tex`, rebuild the figures and numbers, recompile.
Deliverables: `manuscript/main.pdf` (23 pp.), `revision-report/revision-report.html`,
`work/manuscript_audit.py`.

## Result

The physics held. Every printed identity I could check symbolically is correct, and the
13-coupling claim is real: 356 accepted states, 13 couplings, each state a solve of the
nonlinear radial problem with its own resolution and its own tensor residual. Figure 1 and
the potential figure plot rotating solutions, not the static potential of
[[boulware-deser-solution]] replotted; the static values appear only as the closed-form
crosses at the `q -> 0` limit.

Two defects were real:

- **The source did not compile.** `namely $\alpha=\Nmaplist$` nested `$...$` inside `$...$`,
  because the generated macro already carries its own maths. TeX aborts with *Display math
  should end with $$*. This is why the committed `main.pdf` was a stale 15-page build
  disagreeing with its own source, which was the referee's first complaint. Fixed by moving
  the symbol into the macro's first entry.
- **A mislabelled number.** "Across the full range it accumulates to `4.38e-4`" quoted the
  largest *running* discrepancy of the integral identity, not the value across the range,
  which is `3.81e-4` at the last interval. Now two macros, each named for what it is.

## What was added

- `\Psi` from the Smarr relation alone, `(2M - 3TS - 6*Omega*J)/(2*alpha)`, which uses no
  Jacobian, no linear solve and no parameter derivative and therefore tests the
  implicit-response machinery of [[implicit-response-for-psi]] rather than repeating it.
  Agreement `1.45e-6` relative over 313 states, `1.59e-6` at the `T -> 0` intercepts.
- A figure of the extrapolation itself: fitted points, production cubic, all twenty
  competing intercepts, and residuals.
- Fit residual (`1.77e-6`) and drop-one-state stability (`1.09e-5`), both far below every
  sensitivity in the extremal table, so the envelope measures the *length* of the
  extrapolation and not a bad fit.
- An uncertainty budget for every extrapolated column, `delta j` propagated through the
  extrapolated `(M, J)` pair rather than quadrature-summed.
- A conventions table, resolution and finite-difference tables, and the condition number's
  norm and variables.

Wording narrowed throughout from "minimum mass" to "the extremal-branch mass", and every
extremal entry is now called a central extrapolant.

## The resolution study

Run afterwards, on request: `experiments/egb_rotating_resolution_study.py`, 29 walks.
The complete ascending sequence and the complete `T=0` extrapolation re-solved at `N=48`
and `N=56` at every one of the 13 couplings (26 sequences), plus 3 control walks at the
production `N=64`. The production side is not re-solved: it is `extremal_state` on the
saved states of `results/egb-extremality.json`, which reproduces the shipped extremal
table to `1.4e-14` absolute.

Worst over the 26 sequences: `dPsi_ext = 6.89e-5`, `dmu = 1.68e-5`, `dy = 5.59e-7`,
and `5.6e-5`, `8.1e-5`, `7.1e-5` relative in `M`, `J`, `S`. Below the fit sensitivity at
the same coupling in **all 26**, worst ratio `7.8e-4`. The valley at `y = 0.1752`, the one
the referee singled out, moves by `7.7e-7`. Going `48 -> 56` shrinks the difference at 9 of
13 couplings.

Three caveats, all in the paper. The escalation rule puts the coldest states at `N=96` on
both sides at 20 of 26 sequences. A coarse walk need not reach as cold, so both sides were
refitted over a common temperature range: worst `5.43e-5`. And the shipped measurement was
made under Python 3.11.9 / NumPy 2.4.6 / SciPy 1.17.1 while this machine has 3.12.3 /
1.26.4 / 1.11.4 — see [[environment-drift-in-recorded-measurements]]. The environment
control (same `N=64`, this stack, at `alpha = 0.02, 0.15, 0.5`) attributes up to `1.13e-5`
to the stack alone, which at `alpha = 1/2` exceeds both resolution differences at that
coupling. So the tabulated differences bound resolution *and* environment: upper bounds on
the resolution effect, not estimates of it.

Two bugs in the study script, both at the JSON write after all solves had finished:
`float('nan')` for a relative difference against `y = 0`, and `float('inf')` for a trend
ratio against an exactly vanishing difference. `json.dumps(..., allow_nan=False)` rejects
both. Now `None`. The disk memo meant neither crash cost a solve.

## Not done

No comparison against an independent solver; no solution constructed at exactly `T = 0`;
the leading near-extremal powers of `T` are still not derived. No commit. The pytest suite
was not run (`pytest` absent; the working agreement forbids installing into the base
environment). Literature attributions were not re-verified: there is no `papers/` directory
in this tree.

## Run receipt

- Date: 2026-09-28
- Commands, from the project root:

```bash
PYTHONPATH=src python3 experiments/egb_rotating_resolution_study.py
python3 work/manuscript_audit.py
PYTHONPATH=src python3 manuscript/make_figures.py
python3 manuscript/check_manuscript.py
tectonic -X compile manuscript/main.tex --outdir <tmp>
```

- Environment: Linux 6.8.0-142; Python 3.12.3; NumPy 1.26.4; SciPy 1.11.4; SymPy 1.12;
  matplotlib 3.6.3; Tectonic 0.17.0 with TeX Live bundle v33
- Output:

```text
production side reproduced from saved states to 1.42e-14 absolute
the difference in Psi_ext shrinks from N=48 to N=56 at 9 of 13 couplings
worst |dPsi_ext| over all couplings and resolutions: 6.89e-05
worst |dmu| over all couplings and resolutions:      1.68e-05
worst |dPsi_ext| attributable to the environment alone: 1.13e-05
wrote results/egb-rotating-resolution-study.json
43 passed, 0 failed
wrote supplementary/solutions.csv with 356 rows
wrote numbers.tex with 128 macros
wrote figures/fig-entropy.pdf  22842 bytes
wrote figures/fig-extrapolation.pdf  42312 bytes
wrote figures/fig-potential.pdf  28566 bytes
wrote figures/fig-profiles.pdf  37969 bytes
wrote figures/fig-resolution.pdf  24124 bytes
wrote figures/fig-shift.pdf  27092 bytes
ok: 128 generated quantities, 20 references, no literal numbers in the body
25 pages; no errors, no undefined references, no overfull boxes
```

The audit re-derives the paper's algebra with `sympy` and asserts each difference vanishes,
then reads `results/egb-extremality.json` back and asserts it against what the paper prints.
It does **not** re-solve the boundary-value problem; the solver outputs are taken as given.
