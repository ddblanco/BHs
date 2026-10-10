# Extremal mass of rotating EGB black holes at fixed J

This folder covers asymptotically flat Einstein–Gauss–Bonnet black holes in D = 5 with two equal spins J₁ = J₂ = J. It collects what is known about the extremal mass M_ext(J, α), the exact extremal geometry order by order in α, and the code that produced them. It is meant to let anyone pick up the work.

The one-page summary is [`extremal-mass-report.pdf`](extremal-mass-report.pdf). Read it first.

## Conventions

- Action (16πG₅)⁻¹ ∫√−g (R + α L_GB), with G₅ = 1.
- The published coupling of arXiv:1010.0860 is 4α (see `docs/convenciones.md`).
- Metric ansatz (as in arXiv:1010.0860, eq. 2.8):

  ```
  ds² = dr²/f + r² dθ² + h sin²θ (dφ₁ − w dt)² + h cos²θ (dφ₂ − w dt)²
        + (r² − h) sin²θ cos²θ (dφ₁ − dφ₂)² − b dt²
  ```

- Scaling: M_ext = π^{1/3} J^{2/3} m(s), with s = π^{2/3} α / J^{2/3}.
- The code works in units where the α = 0 extremal Myers–Perry horizon sits at r = 1:
  - b₀ = (r² − 1)²/(r⁴ + 1);
  - f₀ = (1 − 1/r²)²;
  - h₀ = r² + 1/r²;
  - w₀ = √2/(r⁴ + 1);
  - J₀ = π√2/4.

  In these units the solver's per-order mass coefficient M_n relates to the series coefficient by d_n = M_n 2^{1−n}/π.

## Results and their status

| Statement | Status |
|---|---|
| m(s) = 3/2 + s + … (linear term) | published, arXiv:2009.00015 |
| d₂ = −236/105 | exact; obtained two independent ways (Euclidean action; direct exterior solution here) |
| d₃ = 60851752/33075 − 1536 ζ(3) | exact; two independent ways, as above |
| d₄ = −592541765354296/1489863375 + (227328/5) ζ(3) + (88064/25) π⁴ ≈ 66.1005220227 | exact, conditional on the zeta-value identification below; matches the numerical BVP value 66.10052 |
| Extremal horizon: r_H² = 1 + (16/3)α + (64/9)α² − (83968/81)α³ − 8192α⁴ + O(α⁵) | exact, conditional as above (from order 3 on) |
| Exterior metric (b, f, h, w) through α⁴ | exact closed form in multiple polylogarithms; files `code/pert_fields_order*.json` |
| M_ext = 3πα/4 + 4J/(3√α) + o(J/√α), Ω_H√α → 2/3, as α → ∞ | candidate; assumes uniform matching of three regions (see `reports/strong-coupling-endpoint.pdf`) |
| J → 0 endpoint at M = 3πα/4 | singular static solution; that the rotating extremal branch ends there is a conjecture of Kleihaus–Kunz–Radu (arXiv:2303.12471) |
| Closed form for m(s) at finite coupling | unknown; no proof that none exists |
| Convergence of the weak-coupling series | not studied |

**The condition on d₄.** Evaluating the polylogarithms at the horizon produces regularized constants C_w, one per word w.
- `code/c_constants.py` computes them to 110 digits and identifies them with multiple zeta values by PSLQ, through weight 6.
- The precision is 100 digits, with residual < 10⁻⁸⁰.
- Two of the identifications are checked independently: C₀₁ = ζ(2) to 10⁻⁹⁰, and C₁₁₁ = 0 by direct quadrature (`check_c111.py`).
- The rest are numerical identifications, not proofs.
- Through order α⁴ only ζ(2), ζ(3) and ζ(2)² appear.

## Method (code/pert.py)

1. **Reduce the equations.** Starting from the EGB field equations for the ansatz, the cyclic angular variable is eliminated exactly. At fixed J the angular first integral is a constant (C₂ = 32 in these units), and u = b′/b is solved from the constraint. This leaves a coupled system for f and h (`reduced_system.py`).
2. **Expand.** The system is expanded in α at fixed areal radius around the extremal Myers–Perry solution. At each order, h_n obeys a third-order linear equation whose operator factorizes into three first-order factors in x = r².
3. **Integrate exactly.** The source is a combination of hyperlogarithms G_w(x) on the letters {0, 1}, with rational coefficients. Three exact quadratures (Hermite reduction plus shuffle products, `hyperlog.py`) give h_n. Then f_n follows algebraically, b_n and w_n each follow from one more quadrature.
4. **Fix the constants.** The three homogeneous constants, the horizon shift X_n, and the normalizations of b and w are fixed by:
   - b → 1, w → 0 and h/r² → 1 at infinity;
   - h regular at the horizon;
   - f having a double zero at the shifted horizon. This is the extremality condition, imposed on the horizon-centred expansion in u = x − x_H and log u.
5. **Read the mass and angular velocity.**
   - M_n comes from the 1/r² tail of b.
   - Ω_n comes from w at the horizon.
   - The first law dM = 2Ω_H dJ at fixed α is checked as an identity at each order. It holds with zero residual at orders 2, 3 and 4.
6. **Control assertions.** At every order SymPy asserts that:
   - the lower orders satisfy the full reduced system exactly;
   - f_n satisfies its own equation;
   - the regularity and double-zero conditions hold.

## Reproducing

Python 3.11 with SymPy 1.14 and mpmath 1.3 (see `environment/requirements-lock.txt` at the repo root). The scripts import the generated field equations from `src/rotating_bh/`, so run them inside this repository.

```
cd extremal-mass/code
python -B extremal_perturbation.py   # order-1 extremal fixed-J profiles   -> extremal_perturbation.json
python -B master_k1.py               # alpha^2 master equation              -> master_k1.json
python -B second_order.py            # stand-alone exact alpha^2 exterior   -> second_order_fields.json
python -B constants.py               # its constants, M_2 and Omega_2       -> constants.json
python -B c_constants.py 6           # horizon constants C_w, PSLQ (~25 s)  -> c_constants.json
python -B check_c111.py              # quadrature control of one C_w
python -B pert.py 4 5                # orders 2..4 (~6 min)              -> pert_order4.json, pert_fields_order4.json
```

`pert.py N W` solves through order α^N, keeping hyperlog words up to weight W. Order n needs weight n, and a missing word raises `KeyError`. It prints, for each order:
- the constants;
- M_n, also with zeta values substituted;
- Ω_n;
- the first-law residual.

**Field files.** `pert_fields_orderN.json` has keys `b`, `f`, `h`, `v` (with v = w/√2). Each key holds a list indexed by the order in α. Each entry maps `"word|zeta-monomial"` to a rational function of r.
- The word is a comma-separated string of letters, outermost letter first. `""` means the rational part.
- G_(1) = log(1 − 1/x).
- G_(a, w′)(x) = ∫_∞^x G_{w′}(t) dt/(t − a), with x = r².
- In the zeta monomial, Z2 = ζ(2), Z3 = ζ(3), Z5 = ζ(5).

## How to continue

- **Order α⁵.** Run `python -B pert.py 5 6`. The C_w it needs (weight ≤ 6) are in `c_constants.json`. Expect much higher time and memory than order 4. If ζ(5) or ζ(3)² appear, they will be carried automatically.
- **Convergence.** With d₂…d₅ (or d₆), estimate the radius of convergence and the nearest singularity of m(s) (ratio test, Padé, Borel). The signs so far (+, −, −, +) suggest complex singularities.
- **Proving the zeta identifications.** Every C_w is a regularized value at x = 1 of a hyperlogarithm on {0, 1}. It can be reduced to multiple zeta values symbolically, for example with HyperInt or by shuffle regularization, which would remove the condition on d₄.
- **Finite coupling.** The numerical branch at fixed J should be extended beyond α ≈ 3 to test the strong-coupling candidate. Any proposed closed form must reproduce, at points not used to build it:
  - d₂ to d₄;
  - the strong-coupling limit;
  - the first law;
  - the numerical branch.
- **Known dead ends.** Do not repeat these without a new idea:
  - finite-Laurent Hamilton–Jacobi superpotentials;
  - Liouvillian solutions of the squashing variational equation (its Galois group is SL(2, ℂ));
  - the resonant candidate ODE for m(s), which an independent BVP point rules out.

## Files

- `extremal-mass-report.pdf` / `.tex`: one-page summary.
- `reports/strong-coupling-endpoint.pdf`: derivation of the α → ∞ candidate and the throat analysis.
- `code/`: scripts and their JSON outputs, as listed under Reproducing.
  - `probe.py`, `reduced_system.py` and `radial_series.py` are the equation-reduction layer.
  - `hyperlog.py` contains the exact hyperlogarithm algebra.
