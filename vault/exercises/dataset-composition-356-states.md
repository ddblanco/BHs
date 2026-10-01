# Questions answered from the stored artifacts

Four questions about the manuscript and the talk, each answered from a repository
artifact rather than recollection. Deliverable:
`presentation-claude/questions-answers.tex` (+ `.pdf`, 6 pages).
Subjects: [[extremality-shift-rotating-black-holes]], [[walk-continuation-in-spin]],
[[rotating-egb-spectral-solver]].

**356 is a total over couplings, not a count per coupling.** Each of the 13 couplings
carries one [[walk-continuation-in-spin]]; 356 is the summed length of the 13 walks.
262 come from the 6 couplings walked over the whole spin range (alpha = 0, 0.02, 0.05,
0.1, 0.2, 0.5), 94 from the 7 narrow walks that start at q = 0.6. This reproduces
`\Nstates`, `\Ncouplings`, `\Nmapcouplings`, `\Nnarrowcouplings` in `manuscript/numbers.tex`.

**The alpha grid is not equispaced.** Gaps widen from 0.005 to 0.1 — about a factor 20 —
finest where the departure from the first-order `M_ext = (3/2) pi^(1/3) J^(2/3) + pi alpha`
has to be resolved.

**A walk** is a chain of solutions at fixed alpha, stepped in `omega_h` with
`spin_step=.01`, each Newton solve seeded by the previous one, from a base built by
climbing in the coupling at `omega_h = 0.6`. Both directions: down to `omega_h = 0`,
up until `T < 1e-6`. States are recorded on a schedule (`spin_record=.02` or a factor
`temperature_record=1.3` in T), so 356 counts *recorded* states and the number of
Newton solves behind them is larger. See [[walk-continuation-in-spin]].

The phrasing in section 4 is ambiguous in exactly the way that prompted the question —
"356 accepted solutions at 13 couplings" reads as per-coupling. Worth a one-sentence fix.

**The "rotating solver"** of the cost-slide notes is
`src/rotating_bh/egb_rotating_bvp.py` — Chebyshev-Lobatto spectral collocation, damped
Newton with a complex-step Jacobian, compact amplitudes B,F,H,W. Hito 4: 11.2 h, 508 M
tokens, the largest block. The notes' "five times" is 508/101.6 against hitos 0-3, i.e.
hours for one side and tokens for the other. Five solver routes were discarded first.
See [[rotating-egb-spectral-solver]].

**It is not the solver of arXiv:1010.0860** (they used COLSYS; we wrote a spectral one and
rederived the equations). More to the point, and worth remembering:
*the rotating bulk solver has no successful finite-alpha cross-check against their
numerically obtained values.* Of the eight external contrasts, two are Einstein-limit
(no Gauss-Bonnet content), the tightest Gauss-Bonnet ones are against closed forms or use
`near_horizon.py` instead of the bulk solver, and the one genuine bulk comparison —
the section 4.1.1 ergosurface radius — fails: 1.102101 vs 1.104. The manuscript contains
the failure by arguing r_e enters none of the result's ingredients. Sound, but a referee
will press it. See [[external-contrasts-that-do-not-close]].

## Run receipt

- Date: 2026-09-30
- Command: `python3` on `results/egb-extremality.json`, counting `walks` per coupling
  (script body reproduced in `presentation-claude/questions-answers.tex`, Provenance)
- Environment: Linux 6.8.0; Python 3.12.3; stdlib `json` only
- Output:

```text
couplings        : 13
alpha grid       : [0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
gaps             : [0.005, 0.005, 0.01, 0.015, 0.015, 0.025, 0.025, 0.05, 0.05, 0.1, 0.1, 0.1]
equispaced       : False
states per alpha : {'0': 43, '0.005': 10, '0.01': 11, '0.02': 40, '0.035': 11, '0.05': 45,
                    '0.075': 16, '0.1': 45, '0.15': 17, '0.2': 45, '0.3': 16, '0.4': 13, '0.5': 44}
full-walk states : 262
narrow states    : 94
total states     : 356
ASSERTIONS PASS: total==356, couplings==13, full==6, narrow==7
```

The solver and contrast answers carry no new computation: they are readings of
`src/rotating_bh/egb_rotating_bvp.py`, `prompts/uso-de-tiempo-y-tokens.md`,
`results/egb-rotating-external-contrasts.json` and section 4.1.1 of the source PDF in
`papers/`. The only arithmetic is 508/101.6 = 5.0.

## Not checked

- No walk was re-run and no solve was performed; the 356 stored rows were not individually
  re-tested against the acceptance thresholds. The counts are read from the stored
  artifact, not reproduced.
- The reading of `walk`/`walk_to_extremality` is from source, not from instrumenting a run.
- The external contrasts were read from the stored JSON, not recomputed.
