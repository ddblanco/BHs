import json
import tempfile
import unittest
from pathlib import Path

from rotating_bh.provenance import validate_manifest, validate_record


VALID = {
    "id": "env-001",
    "kind": "report",
    "path": "reports/env.json",
    "command": "python -m rotating_bh.environment",
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


class RecordValidationTests(unittest.TestCase):
    def test_valid_record_has_no_errors(self):
        self.assertEqual(validate_record(VALID), [])

    def test_missing_fields_are_reported(self):
        errors = validate_record({})

        self.assertEqual(len(errors), len(VALID))
        self.assertIn("missing required field: id", errors)
        self.assertIn("missing required field: status", errors)

    def test_fields_with_wrong_types_are_reported(self):
        record = VALID | {
            "id": 7,
            "inputs": {},
            "parameters": [],
            "prompt_refs": [3],
            "checks": [{"name": "schema", "passed": "yes"}],
        }

        errors = validate_record(record)

        self.assertIn("field id must be a non-empty string", errors)
        self.assertIn("field inputs must be a list", errors)
        self.assertIn("field parameters must be an object", errors)
        self.assertIn("field prompt_refs must be a list of strings", errors)
        self.assertIn("check 0 field passed must be a boolean", errors)

    def test_status_must_be_supported(self):
        self.assertIn(
            "status must be one of: candidate, verified, rejected",
            validate_record(VALID | {"status": "published"}),
        )

    def test_verified_record_requires_passing_check(self):
        self.assertIn(
            "verified record requires a passing check",
            validate_record(VALID | {"checks": []}),
        )
        self.assertIn(
            "verified record requires a passing check",
            validate_record(
                VALID | {"checks": [{"name": "schema", "passed": False}]}
            ),
        )

    def test_non_object_record_returns_validation_error(self):
        self.assertEqual(validate_record([]), ["record must be an object"])


class ManifestValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Path(".cache/tmp").mkdir(parents=True, exist_ok=True)

    def write_manifest(self, contents: object) -> Path:
        directory = tempfile.TemporaryDirectory(dir=".cache/tmp")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "manifest.json"
        path.write_text(json.dumps(contents), encoding="utf-8")
        return path

    def test_empty_manifest_is_valid(self):
        validate_manifest(self.write_manifest([]))

    def test_manifest_rejects_duplicate_ids(self):
        path = self.write_manifest([VALID, VALID])

        with self.assertRaisesRegex(ValueError, "duplicate artifact id: env-001"):
            validate_manifest(path)

    def test_manifest_rejects_non_list_root_with_value_error(self):
        with self.assertRaisesRegex(ValueError, "manifest must be a list"):
            validate_manifest(self.write_manifest({"record": VALID}))

    def test_manifest_rejects_malformed_records_with_value_error(self):
        path = self.write_manifest([None, VALID | {"checks": None}])

        with self.assertRaises(ValueError) as caught:
            validate_manifest(path)

        message = str(caught.exception)
        self.assertIn("record 0: record must be an object", message)
        self.assertIn("record 1: field checks must be a list", message)


if __name__ == "__main__":
    unittest.main()
