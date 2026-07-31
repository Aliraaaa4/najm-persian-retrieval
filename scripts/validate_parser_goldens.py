"""Validate parser golden annotation files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from najm_retrieval.parsing.goldens import (
    validate_golden_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOLDEN_DIRECTORY = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "parser_goldens"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate parser golden annotation files."
        )
    )

    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Golden JSON paths. When omitted, all "
            "pilot_*.json files are validated."
        ),
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail when an annotation is not marked complete."
        ),
    )

    return parser.parse_args()


def resolve_paths(
    supplied_paths: list[Path],
) -> list[Path]:
    """Resolve explicit paths or discover pilot files."""

    if supplied_paths:
        return [
            (
                path
                if path.is_absolute()
                else PROJECT_ROOT / path
            )
            for path in supplied_paths
        ]

    return sorted(
        DEFAULT_GOLDEN_DIRECTORY.glob(
            "pilot_*.json"
        )
    )


def main() -> int:
    """Validate selected golden files."""

    args = parse_args()
    paths = resolve_paths(args.paths)

    if not paths:
        print(
            "No golden annotation files were found."
        )
        return 1

    failed = False

    for path in paths:
        try:
            result = validate_golden_file(
                path,
                require_complete=args.require_complete,
            )
        except (
            OSError,
            ValueError,
        ) as error:
            failed = True
            print(f"FAIL: {path}")
            print(f"  - {error}")
            continue

        status_label = (
            "PASS"
            if result.is_valid
            else "FAIL"
        )

        print(f"{status_label}: {path.name}")
        print(f"  sample: {result.sample_id}")
        print(f"  status: {result.status}")
        print(
            f"  coverage: "
            f"{result.covered_chars}/"
            f"{result.total_chars} "
            f"({result.coverage_ratio:.2%})"
        )
        print(
            f"  uncovered: "
            f"{result.uncovered_chars}"
        )
        print(
            f"  overlapping: "
            f"{result.overlapping_chars}"
        )
        print(
            "  reconstruction: "
            + (
                "PASS"
                if result.reconstruction_matches
                else "INCOMPLETE"
            )
        )

        for warning in result.warnings:
            print(f"  warning: {warning}")

        for error in result.errors:
            print(f"  error: {error}")

        if not result.is_valid:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())