import json
import subprocess
import sys


def run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rotating_bh.cli", *(str(arg) for arg in arguments)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_provenance_accepts_empty_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")

    result = run_cli("validate-provenance", manifest)

    assert result.returncode == 0
    assert result.stderr == ""


def test_validate_provenance_rejects_duplicate_ids(tmp_path):
    record = {
        "id": "env-001",
        "kind": "report",
        "path": "reports/env.json",
        "command": "rbh environment --json",
        "commit": "working-tree",
        "inputs": [],
        "parameters": {},
        "environment": "environment/requirements-lock.txt",
        "agent": "codex",
        "prompt_refs": ["prompts/prompt-log.md"],
        "decisions": ["JSON uses the standard library"],
        "checks": [{"name": "schema", "passed": True}],
        "status": "verified",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([record, record]), encoding="utf-8")

    result = run_cli("validate-provenance", manifest)

    assert result.returncode == 2
    assert "duplicate artifact id: env-001" in result.stderr


def test_validate_provenance_rejects_missing_file(tmp_path):
    missing_manifest = tmp_path / "missing.json"

    result = run_cli("validate-provenance", missing_manifest)

    assert result.returncode == 2
    assert str(missing_manifest) in result.stderr


def test_environment_json_reports_required_versions():
    result = run_cli("environment", "--json")

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["python"].count(".") >= 1
    assert set(report["packages"]) == {"numpy", "scipy", "sympy", "matplotlib"}
    assert all(report["packages"].values())
