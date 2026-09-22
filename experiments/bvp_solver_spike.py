"""Run the BVP comparison and update its provenance record in one command."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy

from rotating_bh.bvp_benchmark import (
    ADAPTIVE_MAX_NODES, ADAPTIVE_TOL, CHECK_POINT_COUNT,
    max_error, solve_adaptive, solve_spectral,
)
from rotating_bh.provenance import validate_manifest, validate_record

ROOT = Path(__file__).resolve().parents[1]


def measure(solver, name, n):
    started = perf_counter()
    solution = solver(n)
    elapsed = perf_counter() - started
    return {
        "solver": name, "initial_resolution": n,
        "effective_resolution": solution.effective_resolution,
        "elapsed_seconds": elapsed, "max_error": max_error(solution),
        "residual": solution.residual,
        "boundary_error": float(np.max(np.abs(solution.evaluate([0.0, 1.0])))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[16, 24, 32, 48])
    parser.add_argument("--output", type=Path, default=Path("results/bvp-spike.json"))
    args = parser.parse_args(argv)
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside the project")
    manifest_path = ROOT/"artifacts/manifest.json"
    if output == manifest_path:
        parser.error("output cannot overwrite the provenance manifest")
    validate_manifest(manifest_path)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_paths = ["experiments/bvp_solver_spike.py", "src/rotating_bh/bvp_benchmark.py",
                    "src/rotating_bh/provenance.py", "environment/requirements-lock.txt"]
    hashes = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
              for path in source_paths}
    parameters = {
        "resolutions": args.resolutions, "adaptive_tol": ADAPTIVE_TOL,
        "adaptive_max_nodes": ADAPTIVE_MAX_NODES,
        "check_point_count": CHECK_POINT_COUNT, "acceptance_threshold": 1e-7,
        "adaptive_representation": "integrated velocity cubic with linear boundary correction",
        "spectral_representation": "Chebyshev-Lobatto polynomial; DCT-I coefficients",
        "source_sha256": hashes,
    }
    runs = [measure(solver, name, n) for n in args.resolutions
            for name, solver in (("adaptive", solve_adaptive), ("spectral", solve_spectral))]
    convergence = [measure(solve_spectral, "spectral", n) for n in (8, 12)]
    checks = [
        {"name": "all_sampled_errors_below_1e-7",
         "passed": all(run["max_error"] < 1e-7 for run in runs)},
        {"name": "all_off_collocation_residuals_below_1e-7",
         "passed": all(run["residual"] < 1e-7 for run in runs)},
        {"name": "all_boundary_errors_below_1e-12",
         "passed": all(run["boundary_error"] < 1e-12 for run in runs)},
        {"name": "spectral_error_and_residual_converge_8_to_12",
         "passed": all(convergence[1][key] < convergence[0][key]/100
                       for key in ("max_error", "residual"))},
    ]
    scientifically_verified = all(check["passed"] for check in checks)
    spectral_not_worse = all(runs[i+1]["residual"] <= runs[i]["residual"]
                             for i in range(0, len(runs), 2))
    checks.append({"name": "spectral_residual_not_worse_than_baseline",
                   "passed": spectral_not_worse})
    decision = {
        "baseline": "scipy.integrate.solve_bvp",
        "spectral_candidate": scientifically_verified and spectral_not_worse,
        "scope": "smooth linear manufactured BVP only; no EGB performance certification",
    }
    payload = {"problem": "y'' + pi^2 sin(pi*x) = 0; y(0)=y(1)=0",
               "parameters": parameters,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "platform": platform.platform()},
               "runs": runs, "spectral_convergence": convergence,
               "checks": checks, "decision": decision}
    serialized = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    python = Path(sys.executable).resolve()
    python_command = python.relative_to(ROOT).as_posix() if python.is_relative_to(ROOT) else str(python)
    command = subprocess.list2cmdline([
        python_command, "experiments/bvp_solver_spike.py", "--resolutions",
        *map(str, args.resolutions), "--output", output.relative_to(ROOT).as_posix(),
    ])
    record = {
        "id": "bvp-solver-spike", "kind": "result", "path": output.relative_to(ROOT).as_posix(),
        "command": command, "commit": "working-tree", "inputs": source_paths,
        "parameters": parameters, "environment": "environment/requirements-lock.txt",
        "agent": "codex", "prompt_refs": ["plans/2026-09-04-hito-0.md",
                                               "reports/sdd/task-5-brief.md"],
        "decisions": [parameters["adaptive_representation"], parameters["spectral_representation"],
                      "1001 uniform midpoints, with collocation coincidences displaced",
                      "Residual differentiates the evaluated approximation; sampled maxima are not bounds",
                      "solve_bvp remains baseline; spectral candidacy requires convergence and no worse residual",
                      decision["scope"]],
        "checks": checks, "status": "verified" if scientifically_verified else "rejected",
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }
    errors = validate_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    for index, old in enumerate(records):
        if old["id"] == record["id"]:
            records[index] = record
            break
    else:
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)
    manifest_path.write_text(json.dumps(records, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": record["path"], "status": record["status"], **decision}))
    return 0 if scientifically_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
