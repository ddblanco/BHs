"""Validation for reproducible artifact provenance records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "id",
    "kind",
    "path",
    "command",
    "commit",
    "inputs",
    "parameters",
    "environment",
    "agent",
    "prompt_refs",
    "decisions",
    "checks",
    "status",
)
VALID_STATUSES = ("candidate", "verified", "rejected")
STRING_FIELDS = (
    "id",
    "kind",
    "path",
    "command",
    "commit",
    "environment",
    "agent",
)
STRING_LIST_FIELDS = ("inputs", "prompt_refs", "decisions")


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return all schema errors found in one provenance record."""
    if not isinstance(record, dict):
        return ["record must be an object"]

    errors = [
        f"missing required field: {field}"
        for field in REQUIRED_FIELDS
        if field not in record
    ]

    for field in STRING_FIELDS:
        if field in record and (
            not isinstance(record[field], str) or not record[field].strip()
        ):
            errors.append(f"field {field} must be a non-empty string")

    for field in STRING_LIST_FIELDS:
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, list):
            errors.append(f"field {field} must be a list")
        elif any(not isinstance(item, str) for item in value):
            errors.append(f"field {field} must be a list of strings")

    if "parameters" in record and not isinstance(record["parameters"], dict):
        errors.append("field parameters must be an object")

    checks = record.get("checks")
    if "checks" in record and not isinstance(checks, list):
        errors.append("field checks must be a list")
    elif isinstance(checks, list):
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                errors.append(f"check {index} must be an object")
                continue
            if not isinstance(check.get("name"), str) or not check["name"].strip():
                errors.append(f"check {index} field name must be a non-empty string")
            if not isinstance(check.get("passed"), bool):
                errors.append(f"check {index} field passed must be a boolean")

    status = record.get("status")
    if "status" in record and status not in VALID_STATUSES:
        errors.append("status must be one of: candidate, verified, rejected")

    if status == "verified" and not (
        isinstance(checks, list)
        and any(
            isinstance(check, dict) and check.get("passed") is True
            for check in checks
        )
    ):
        errors.append("verified record requires a passing check")

    return errors


def validate_manifest(path: Path) -> None:
    """Raise ``ValueError`` when a JSON manifest violates the contract."""
    try:
        manifest_path = Path(path)
    except TypeError as error:
        raise ValueError("manifest path must be path-like") from error

    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest JSON: {error.msg}") from error

    if not isinstance(records, list):
        raise ValueError("manifest must be a list")

    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        errors.extend(
            f"record {index}: {error}" for error in validate_record(record)
        )
        if not isinstance(record, dict):
            continue
        artifact_id = record.get("id")
        if not isinstance(artifact_id, str):
            continue
        if artifact_id in seen_ids:
            errors.append(f"duplicate artifact id: {artifact_id}")
        else:
            seen_ids.add(artifact_id)

    if errors:
        raise ValueError("; ".join(errors))
