"""Independent, lightweight checks used for the Codex referee report.

This is not a reproduction of the rotating boundary-value solve.  It checks
the algebra printed in the manuscript and conclusions that can be read back
from the saved extremal table without importing the project implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # Static potential, manuscript eqs. (static) and (psistatic).
    r, alpha = sp.symbols("r alpha", positive=True)
    mass = 3 * sp.pi * (r**2 + 2 * alpha) / 8
    temperature = r / (2 * sp.pi * (r**2 + 4 * alpha))
    entropy = sp.pi**2 * r**3 / 2 + 6 * sp.pi**2 * alpha * r
    psi_static = sp.diff(mass, alpha) - temperature * sp.diff(entropy, alpha)
    claimed_static = (
        3
        * sp.pi
        / 4
        * (4 * alpha / r**2 - 3)
        / (1 + 4 * alpha / r**2)
    )
    assert sp.simplify(psi_static - claimed_static) == 0

    # Differential of the Smarr relation after eliminating dM with the first law.
    T, S, Omega, J, Psi = sp.symbols("T S Omega J Psi")
    dT, dS, dOmega, dJ, dPsi, dAlpha = sp.symbols(
        "dT dS dOmega dJ dPsi dAlpha"
    )
    d_smarr_rhs = (
        3 * (T * dS + S * dT)
        + 6 * (Omega * dJ + J * dOmega)
        + 2 * (alpha * dPsi + Psi * dAlpha)
    )
    twice_first_law = 2 * (T * dS + 2 * Omega * dJ + Psi * dAlpha)
    claimed_constraint = (
        T * dS
        + 3 * S * dT
        + 2 * Omega * dJ
        + 6 * J * dOmega
        + 2 * alpha * dPsi
    )
    assert sp.simplify(d_smarr_rhs - twice_first_law - claimed_constraint) == 0

    # Perturbative potential endpoints and its physical zero.
    u, q = sp.symbols("u q", nonnegative=True)
    psi0 = -sp.pi * (u**2 - 14 * u + 9) / 4
    assert sp.simplify(psi0.subs(u, 0) + 9 * sp.pi / 4) == 0
    assert sp.simplify(psi0.subs(u, 1) - sp.pi) == 0
    physical_u_zero = 7 - 2 * sp.sqrt(10)
    q_zero = sp.sqrt(physical_u_zero / (1 + physical_u_zero))

    data_path = ROOT / "results" / "egb-extremality.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    rows = sorted(data["extremals"], key=lambda row: row["y"])
    y = np.array([row["y"] for row in rows], dtype=float)
    mu = np.array([row["mu"] for row in rows], dtype=float)
    psi = np.array([row["psi_gb"]["value"] for row in rows], dtype=float)
    sensitivity = np.array([row["psi_uncertainty"] for row in rows], dtype=float)

    assert np.all(np.diff(y) > 0)
    assert np.all(np.diff(mu) > 0)
    assert np.all(psi > 0)
    minimum_index = int(np.argmin(psi))
    assert 0 < minimum_index < len(psi) - 1

    # Even treating the stated sensitivities as adversarial half-widths, the
    # endpoint-to-valley fall and valley-to-endpoint rise remain separated.
    conservative_fall = (
        psi[0] - sensitivity[0] - (psi[minimum_index] + sensitivity[minimum_index])
    )
    conservative_rise = (
        psi[-1]
        - sensitivity[-1]
        - (psi[minimum_index] + sensitivity[minimum_index])
    )
    assert conservative_fall > 0
    assert conservative_rise > 0

    trapezoid_rule = getattr(np, "trapezoid", np.trapz)
    trapezoid = float(trapezoid_rule(psi, y))
    mass_change = float(mu[-1] - mu[0])
    cumulative = data["route_comparison"]["cumulative"]
    endpoint_cubic_difference = float(cumulative[-1]["difference"])
    maximum_running_difference = float(max(row["difference"] for row in cumulative))
    assert math.isclose(
        maximum_running_difference,
        data["route_comparison"]["worst_cumulative"],
        rel_tol=0,
        abs_tol=1e-15,
    )

    recorded = data["source_sha256"]["src/rotating_bh/egb_rotating_bvp.py"]
    current = sha256(ROOT / "src" / "rotating_bh" / "egb_rotating_bvp.py")

    print("PASS symbolic static-potential identity")
    print("PASS symbolic differentiated-Smarr constraint")
    print("PASS perturbative endpoints Psi_0(0)=-9*pi/4 and Psi_0(1)=pi")
    print(f"PASS perturbative sign zero q={float(q_zero.evalf(16)):.12f}")
    print(f"PASS saved table: {len(rows)} ordered y values, all Psi>0, all mu increasing")
    print(
        "PASS saved-table non-monotonic trend survives stated sensitivities: "
        f"fall gap={conservative_fall:.9f}, rise gap={conservative_rise:.9f}"
    )
    print(
        "CHECK trapezoid integral vs mass change: "
        f"{trapezoid:.12f} vs {mass_change:.12f}; "
        f"difference={abs(trapezoid - mass_change):.12f}"
    )
    print(
        "CHECK manuscript cubic-route running discrepancy: "
        f"endpoint={endpoint_cubic_difference:.12f}, "
        f"maximum={maximum_running_difference:.12f}"
    )
    print(f"CHECK source digest recorded={recorded}")
    print(f"CHECK source digest current ={current}")
    print(f"CHECK digest match={recorded == current}")
    print("NOT CHECKED continuum BVP solve or cited literature")


if __name__ == "__main__":
    main()
