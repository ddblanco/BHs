"""Scientific checks of actual off-grid approximations, not ODE identities."""

import numpy as np
import pytest
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from rotating_bh.bvp_benchmark import max_error, solve_adaptive, solve_spectral


@pytest.mark.parametrize("solver", [solve_adaptive, solve_spectral])
def test_solver_recovers_solution_and_has_small_independent_residual(solver):
    solution = solver(32)
    points = solution.check_points
    assert len(points) == 1001
    assert np.min(np.abs(points[:, None] - solution.nodes)) > 1e-12
    assert max_error(solution) < 1e-7
    measured = np.max(np.abs(solution.evaluate(points, derivative=2)
                             + np.pi**2 * np.sin(np.pi * points)))
    assert solution.residual == pytest.approx(measured, rel=1e-12, abs=1e-15)
    assert solution.residual < 1e-7
    assert np.max(np.abs(solution.evaluate(np.array([0.0, 1.0])))) < 1e-12


@pytest.mark.parametrize("solver", [solve_adaptive, solve_spectral])
def test_reported_second_derivative_belongs_to_evaluated_solution(solver):
    solution = solver(8)
    x, h = np.array([0.173, 0.413, 0.769]), 0.001
    y = solution.evaluate
    finite_difference = (-y(x+2*h)+16*y(x+h)-30*y(x)+16*y(x-h)-y(x-2*h))/(12*h*h)
    np.testing.assert_allclose(finite_difference, y(x, derivative=2), atol=2e-7, rtol=0)


def test_spectral_convergence_before_roundoff():
    coarse, fine = solve_spectral(8), solve_spectral(12)
    assert max_error(coarse) > 1e-8
    assert coarse.residual > 1e-5
    assert max_error(fine) < max_error(coarse) / 100
    assert fine.residual < coarse.residual / 100


@pytest.mark.parametrize("solver", [solve_adaptive, solve_spectral])
@pytest.mark.parametrize("n", [0, 1, 2, -3, 8.5, True, "32"])
def test_invalid_resolution_is_rejected(solver, n):
    with pytest.raises(ValueError, match="integer.*3"):
        solver(n)


def test_adaptive_nonconvergence_is_not_returned_as_solution():
    with pytest.raises(RuntimeError, match="converge"):
        solve_adaptive(8, max_nodes=8)


def test_experiment_repeated_run_keeps_result_hashes_and_manifest_consistent(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = Path("experiments/bvp_solver_spike.py")
    assert (root/script).is_file(), "experiment CLI is not implemented"
    sources = [script, Path("src/rotating_bh/bvp_benchmark.py"),
               Path("src/rotating_bh/provenance.py"),
               Path("environment/requirements-lock.txt")]
    for source in sources:
        target = tmp_path/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/source, target)
    manifest = tmp_path/"artifacts/manifest.json"
    manifest.parent.mkdir()
    foreign = {
        "id": "other-result", "kind": "report", "path": "other.json",
        "command": "other", "commit": "working-tree", "inputs": [],
        "parameters": {}, "environment": "environment/requirements-lock.txt",
        "agent": "other", "prompt_refs": [], "decisions": [], "checks": [],
        "status": "candidate",
    }
    manifest.write_text(json.dumps([foreign]), encoding="utf-8")
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(tmp_path/script), "--resolutions", "16", "32",
             "--output", "results/bvp-spike.json"],
            cwd=tmp_path, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        records = json.loads(manifest.read_text(encoding="utf-8"))
        assert len(records) == 2
        assert records[0] == foreign
        record = records[1]
        output = tmp_path/record["path"]
        assert record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        for source, digest in record["parameters"]["source_sha256"].items():
            assert hashlib.sha256((tmp_path/source).read_bytes()).hexdigest() == digest
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data["runs"]) == 4
        assert {run["solver"] for run in data["runs"]} == {"adaptive", "spectral"}
        assert all(run["max_error"] < 1e-7 and run["residual"] < 1e-7
                   and run["elapsed_seconds"] > 0 for run in data["runs"])
        assert all(check["passed"] for check in record["checks"])
