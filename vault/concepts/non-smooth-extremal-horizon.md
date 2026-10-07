# Non-smooth extremal horizon (equal-spin EGB)

The extremal equal-spin Einstein–Gauss–Bonnet throat (AdS2 fibred over a squashed S^3)
has U(2)-symmetric perturbations `delta ~ rho^gamma` (rho the AdS2 radius, gauge
`g^{rho rho}` fixed). The exponents come in AdS2 pairs `(gamma, -1-gamma)`.

- alpha = 0 (extremal Myers–Perry): integers {-3, -2, 0, 0, 1, 2} — smooth.
- finite coupling: the growing exponent 2 becomes non-integer gamma(y), y = alpha/J^(2/3):
  minimum 1.773 at y ≈ 0.029, back through 2 at y ≈ 0.71, through 3 at y ≈ 4.5,
  5.29 at y ≈ 36.
- gamma < 2: metric functions C^1 but not C^2 in rho at the horizon (curvature in a
  parallelly propagated frame not computed).
- Consequence: a pure power series in KKR's r (r^2 ∝ rho) at the horizon is not exact;
  needs r^(2 gamma) terms. Spectral methods converge algebraically wherever the mass is
  read through a boundary derivative.
- Recalled, not re-read: the analogue for Kerr with higher-derivative corrections is
  arXiv:2303.07358.

Evidence: the linear analysis (`work/kkr-extremal/nearhorizon.py`) only. The full
solutions are *consistent* with it (algebraic convergence in the P representation;
constraint and resolution differences peak next to the horizon), but a direct fit of the
Chebyshev tail cannot resolve the exponent in double precision
(`gamma_check.py`: the tail reaches round-off by n ≈ 38).

Related: [[kkr-extremal-reproduction]], [[extremality-shift-rotating-black-holes]],
[[arxiv-2303.12471]].
