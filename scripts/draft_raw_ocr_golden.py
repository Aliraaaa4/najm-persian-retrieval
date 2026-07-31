"""Create a conservative draft golden for a raw OCR sample."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from najm_retrieval.parsing.goldens import (
    validate_golden_data,
)
from najm_retrieval.parsing.raw_ocr_drafting import (
    apply_raw_ocr_draft,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_utf8_output() -> None:
    """Use UTF-8 for redirected Windows terminal output."""

    for stream_name in (
        "stdout",
        "stderr",
    ):
        stream = getattr(
            sys,
            stream_name,
        )

        reconfigure = getattr(
            stream,
            "reconfigure",
            None,
        )

        if callable(reconfigure):
            reconfigure(
                encoding="utf-8",
                errors="strict",
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a conservative raw OCR "
            "golden annotation draft."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to one golden sample JSON file.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace existing draft blocks. "
            "Complete annotations are never overwritten."
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve paths relative to the project root."""

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    """Load one golden sample JSON object."""

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise ValueError(
            f"Golden sample does not exist: {path}"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Golden sample must be a JSON object."
        )

    return data


def main() -> int:
    """Draft, validate, and save one raw OCR sample."""

    configure_utf8_output()

    args = parse_args()
    path = resolve_path(args.path)

    try:
        sample = load_json(path)

        annotations = sample.get(
            "annotations"
        )

        if not isinstance(annotations, dict):
            raise ValueError(
                "Sample annotations must be an object."
            )

        if annotations.get("status") == "complete":
            raise ValueError(
                "Refusing to overwrite a complete "
                "golden annotation."
            )

        updated = apply_raw_ocr_draft(
            sample,
            force=args.force,
        )

        result = validate_golden_data(
            updated,
            require_complete=False,
        )

        if not result.is_valid:
            print(
                "ERROR: generated draft is invalid "
                "and was not saved."
            )

            for error in result.errors:
                print("  -", error)

            return 1

        path.write_text(
            json.dumps(
                updated,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    blocks = updated[
        "annotations"
    ]["blocks"]

    counts = Counter(
        block["block_type"]
        for block in blocks
    )

    print("Updated:", path)
    print("Sample:", updated.get("sample_id"))
    print(
        "Status:",
        updated["annotations"]["status"],
    )
    print("Blocks:", len(blocks))

    for block_type in sorted(counts):
        print(
            f"  {block_type}: "
            f"{counts[block_type]}"
        )

    print(
        "Coverage:",
        f"{result.covered_chars}/"
        f"{result.total_chars} "
        f"({result.coverage_ratio:.2%})",
    )

    print(
        "Uncovered:",
        result.uncovered_chars,
    )

    print(
        "Overlapping:",
        result.overlapping_chars,
    )

    print(
        "Reconstruction:",
        (
            "PASS"
            if result.reconstruction_matches
            else "FAIL"
        ),
    )

    print(
        "Manual review is required before "
        "changing status to complete."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())