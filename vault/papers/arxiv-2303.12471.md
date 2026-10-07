# arXiv:2303.12471 — Kleihaus, Kunz, Radu, *Phases of rotating black objects in d=5 EGB*

Source in `work/kkr-extremal/arxiv/` (paper.tex + 9 gnuplot EPS); PDF in `papers/`.

## Claims (equal spins, sec. 3) and status here

| claim | status |
|---|---|
| near-horizon closed forms (their at7–at9) | **reproduced** to round-off on 60 points; their at9 "J" is J1+J2 (twice the per-plane J of their eq. 3.3) |
| extremal solutions built directly at T=0 in their gauge, M = 3π/4 (a² − c_t) | **reproduced** with a new spectral solver ([[kkr-extremal-reproduction]]) |
| extremal boundary of fig. 1 (a_H, s vs x and vs j) | **reproduced** on x ≤ 0.81: our curve within ~1e-3 of the digitised one (a few device units) |
| j ≤ 1 and x < 1 on the branch | holds on what we construct |
| extremal set ends at the singular static point (x, j) = (1, 0) (conjecture, hand-drawn dotted line) | **not tested** (we stop at x ≈ 0.81; they reach ≈ 0.92) |
| "approximate form ... as a power series in r" at the horizon | cannot be exact: the horizon needs r^(2 gamma), gamma non-integer ([[non-smooth-extremal-horizon]]) |

## Gauge facts worth remembering

- Their gauge fixes b·g^{rr}; with a and alpha fixed it leaves **two zero modes**: the
  physical size relative to a, and the residual radial gauge r → r + c sqrt(b g^{rr})
  (∝ r^3 at the horizon, constant shift at infinity). Fix with g_H = 1 and "no 1/r in F2".
- In xi = r²/(1+r²) the residual gauge term is xi^(3/2): use r itself (compactified).

## Figures

EPS files: `*-MP2.eps` = equal spins (left panels), `*-MP.eps` = single plane,
`*-BR.eps` = black rings. Each has a main plot and an inset drawn twice; the paths drawn in
plot 0 after its second fill belong to the inset (`kkr_curves.py`).
