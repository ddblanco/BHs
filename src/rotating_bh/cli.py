"""Command-line entry points for reproducibility checks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from rotating_bh.environment import environment_report
from rotating_bh.provenance import validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbh")
    commands = parser.add_subparsers(dest="command", required=True)

    provenance = commands.add_parser(
        "validate-provenance", help="validate a provenance manifest"
    )
    provenance.add_argument("path", type=Path)

    environment = commands.add_parser(
        "environment", help="report the Python and package versions"
    )
    environment.add_argument("--json", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "validate-provenance":
        try:
            validate_manifest(args.path)
        except OSError as error:
            print(f"error: cannot read manifest {args.path}: {error.strerror}", file=sys.stderr)
            return 2
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        return 0

    print(json.dumps(environment_report(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
