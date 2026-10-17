# BTZ universality of the strong-coupling extremal EGB core

D = 5 EGB (action R + α L_GB, G = 1) is the N = 2 "derivative-degenerate" Lovelock theory of
[[arxiv-1201.4994]]: ψ + 2αψ² = μ/r⁴, V = 1 + r²/(4α) − r²(1/(16α²) + μ/(2α r⁴))^(1/2).
For r ≪ √α it behaves like pure GB in its critical odd dimension, i.e. like 3D gravity.

- Static small holes (μ = r_h² + 2α): ΔM = 3π r_h²/8, S = 6π²α r_h, T = r_h/(8πα).
  This is BTZ with ℓ² = 4α and 1/(4G₃) = 3πα (first law checked by hand).
- Extremal BTZ with the same (ℓ, G₃): ΔM = S²/(48π³α²).
- Exact extremal throat at fixed J (algebraic, [[extremal-mass-strong-coupling]]):
  S_ext²/(48π³ α^(3/2) J) = 4/3 + (3/8) t² + O(t³), t = (2α/g_H)^(−1/2).
- Hence ΔM_ext → 4J/(3√α): the leading strong-coupling coefficient a₁ = 4/3 of
  m(s) = 3s/4 + a₁ s^(−1/2) + …, obtained independently of the matched asymptotics
  in `extremal-mass/reports/strong-coupling-endpoint.pdf`.
- The expansion has no O(t) term, so the next coefficient a₂ (≈ 0.46, numerics) does not
  come from the core.

Status: the BTZ dictionary is derived for static holes; applying it to the rotating extremal
core is an assumption, supported by the agreement with the matched result and with the
numerical branch (`extremal-mass/strong/analyse_sweep.py`).

Receipt: `extremal-mass/strong/btz_universality.py` (asserts the limit 4/3 symbolically),
2026-10-16, SymPy 1.14, output `S^2/(48 pi^3 alpha^(3/2) J) = 3*t**2/8 + 4/3`.
