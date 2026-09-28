# Environment drift in recorded measurements

A saved numerical result carries the linear-algebra stack that produced it. Comparing a
freshly computed value against a recorded one measures the change you intended *plus* the
change of stack, and the second is not always the smaller of the two.

Measured in [[extremality-shift-paper-revision]]. `results/egb-extremality.json` records
Python 3.11.9 / NumPy 2.4.6 / SciPy 1.17.1; the machine that ran the resolution study has
3.12.3 / 1.26.4 / 1.11.4. Re-walking the *same* boundary-value problem at the *same*
resolution `N=64` on the second stack moved the extrapolated `Psi_ext` by up to `1.13e-5` —
at `alpha = 1/2`, more than either of the two genuine resolution changes at that coupling
(`3.9e-8` at `N=48`, `5.9e-7` at `N=56`).

Mechanism: the walk is adaptive. A different BLAS changes the last bits of a Newton step,
which changes whether a state passes the acceptance gate, which changes the step schedule
and hence *which* states get sampled. The drift is therefore not floating-point noise on a
fixed computation; it is a different set of solutions being fitted.

Consequences for how to report such a comparison:

- A coarse-against-recorded difference is an **upper bound** on the resolution effect, not
  an estimate of it. Say so.
- Run a control at the *production* resolution on the new stack. Without it the two
  contributions cannot be separated and the reader has to take the label on trust.
- A comparison whose two sides were computed in one environment is worth keeping even when
  it has less coverage, because it is the only one attributable to a single cause.
- This is the same failure that makes recorded source digests disagree after an unrelated
  edit: a reproducibility claim needs the environment pinned, not just the code.

Related: [[extremality-shift-paper-revision]], [[extremality-shift-paper-review]].
