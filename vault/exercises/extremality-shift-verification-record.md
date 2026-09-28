# Extremality-shift verification record

What this project rederived, what it reproduced numerically, and what it only called.
Deliverables: `verification-report/verification-report.{html,tex,pdf}`, the ledger
`results/verification-ledger.json`, the suite `work/verification_suite.py`, and the
sources themselves in `papers/`.

**58 entries: 47 machine-checked (0 failures), 11 called and not reproduced.**

## The thing this closed

[[extremality-shift-paper-review]] listed the literature attributions as unverifiable
because the tree had no `papers/`. It does now — 17 arXiv PDFs — so the transcriptions
are checked against the sources rather than against the manuscript's account of them.
Three conversions carry every finite-coupling comparison, and an error in any of them
would move all of them without breaking a single internal check. Each is now verified
twice by independent routes:

- **`alpha_paper = 4 alpha`.** Read off eq. (2.1) of 1010.0860, whose action is
  `R + (alpha/4) L_GB`. *Independently*: the ratio of the GB piece of their entropy (3.10)
  to its Einstein piece equals what our Jacobson–Myers form gives **only** for that
  factor. Two routes, one answer.
- **`J` is per plane.** The sentence introducing their (3.6); and their
  `E = -3 V_3 U/16 pi G`, `J = V_3 W/8 pi G` with `V_3 = 2 pi^2` reduce to our
  `-3 pi U/8` and `pi W/4`.
- **`k = 2 k_paper`.** Their near-horizon metric carries `sigma_3 + 2 k r dt` where ours
  carries `sigma_3 + k r dt`; since `J = dE/dk`, `J_paper = 2 J`. This reconciles their
  `S_ext = pi J_paper` at zero coupling with our `S_ext = 2 pi J`, and both with extremal
  Myers–Perry (`S = 2 pi^2 a^3`, `J = pi a^3`). Without it, an apparent factor-two
  discrepancy; with it, one statement. See [[near-horizon-entropy-function]].

## Two transcriptions rederived rather than copied

- **Our metric ansatz is identically their eq. (2.8)** with `g = r^2`. Their form has
  `h sin^2(t)(dphi1 - w dt)^2 + h cos^2(t)(dphi2 - w dt)^2 + (g-h) sin^2 cos^2 (dphi1-dphi2)^2`;
  ours has a single squared one-form plus `r^2 sin^2 cos^2 (dphi1-dphi2)^2`. The
  difference expands to zero.
- **Our eq. (17) is their eq. (33).** Specialising Wu–Lü's general-`D` perturbation to
  `D = 5` with equal spins gives `-(pi/4 r_+^4)(a^4 - 14 a^2 r_+^2 + 9 r_+^4)`, and after
  `u = a^2/r_+^2` that is exactly `-(pi/4)(u^2 - 14u + 9)`. Derived symbolically from the
  published general formula, not transcribed.

## The four external anchors

| anchor | what it fixes | worst deviation |
|---|---|---|
| Myers–Perry, `alpha=0` | mass, horizon squashing | `3.3e-13`, `8.3e-14` |
| Boulware–Deser, zero spin, finite `alpha` | `M`, `T`, `S`, `Psi` | `3.2e-11`, `1.1e-13`, `1e-16`, `2.6e-9` |
| Wu–Lü perturbative `Psi_0(q)`, `alpha=0` | the response itself | `2.9e-8` |
| 1010.0860 near-horizon, finite `alpha` | `sigma = S_ext/J` | `4.1e-6` |

Plus the published extremality shift (2009.00015): `mu(0)` to `4.6e-11` and
`Psi_ext(0) = pi` to `1.2e-7`.

## What it does not establish

The vacuum and static anchors are exact but sit at the *edges* of the family — zero
coupling, zero spin — and the near-horizon anchor constrains entropy, not mass. **No
external result exists for the rotating mass at finite coupling**, which is the quantity
the manuscript reports. That rests on the solver, its convergence behaviour
([[environment-drift-in-recorded-measurements]]) and the `T -> 0` extrapolation.

Still open: the ergosurface radius at `alpha = 1/2`, ours `1.102101` against their
`1.104`. Unresolved, and listed as called-and-not-reproduced rather than quietly dropped.

Three pre-arXiv sources (Myers–Perry 1986, Boulware–Deser 1985, Lovelock 1971) cannot be
downloaded and are listed as called.

## Run receipt

- Date: 2026-09-28
- Commands, from the project root:

```bash
PYTHONPATH=src python3 work/verification_suite.py
PYTHONPATH=src python3 verification-report/make_figures.py
PYTHONPATH=src python3 verification-report/build_report.py
tectonic -X compile verification-report/verification-report.tex
```

- Environment: Linux 6.8.0-142; Python 3.12.3; NumPy 1.26.4; SymPy 1.12;
  matplotlib 3.6.3; Tectonic 0.17.0
- Output:

```text
47 checked, 0 failed; 11 entries called and not reproduced
wrote results/verification-ledger.json
wrote verification-report/figures/fig-boulware-deser.pdf  26315 bytes
wrote verification-report/figures/fig-myers-perry.pdf  24277 bytes
wrote verification-report/figures/fig-near-horizon.pdf  23853 bytes
wrote verification-report/figures/fig-perturbative.pdf  28199 bytes
wrote verification-report/verification-report.tex
wrote verification-report/verification-report.html
9 pages; no errors, no undefined references
```

Both report formats are generated from the same ledger by
`verification-report/build_report.py`, so they cannot disagree with each other or with
the suite. Related: [[extremality-shift-paper-revision]].
