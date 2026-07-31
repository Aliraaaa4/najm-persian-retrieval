"""Create a reviewed-required draft for a structured-poetry golden."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from najm_retrieval.parsing.golden_drafting import (
    apply_structured_poetry_draft,
)
from najm_retrieval.parsing.goldens import (
    load_golden_file,
    validate_golden_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a draft annotation for a "
            "structured-poetry golden sample."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to the golden JSON file.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace existing annotation blocks."
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main() -> int:
    """Create and save the draft annotation."""

    args = parse_args()
    path = resolve_path(args.path)

    try:
        sample = load_golden_file(path)
    except (
        OSError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}")
        return 1

    annotations = sample.get(
        "annotations",
        {},
    )

    existing_blocks = (
        annotations.get("blocks", [])
        if isinstance(annotations, dict)
        else []
    )

    if existing_blocks and not args.force:
        print(
            "ERROR: Annotation blocks already exist."
        )
        print(
            "Use --force only after reviewing the "
            "current file."
        )
        return 1

    try:
        updated = apply_structured_poetry_draft(
            sample
        )
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    result = validate_golden_data(updated)

    if not result.is_valid:
        print(
            "ERROR: Generated draft did not pass "
            "golden validation."
        )

        for error in result.errors:
            print(f"  - {error}")

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

    blocks = updated[
        "annotations"
    ]["blocks"]

    block_counts = Counter(
        block["block_type"]
        for block in blocks
    )

    print(f"Updated: {path}")
    print(f"Sample: {result.sample_id}")
    print(f"Status: {result.status}")
    print(f"Blocks: {len(blocks)}")

    for block_type in sorted(block_counts):
        print(
            f"  {block_type}: "
            f"{block_counts[block_type]}"
        )

    print(
        f"Coverage: "
        f"{result.covered_chars}/"
        f"{result.total_chars} "
        f"({result.coverage_ratio:.2%})"
    )

    print(
        f"Uncovered: "
        f"{result.uncovered_chars}"
    )

    print(
        f"Overlapping: "
        f"{result.overlapping_chars}"
    )

    print(
        "Reconstruction: "
        + (
            "PASS"
            if result.reconstruction_matches
            else "FAIL"
        )
    )

    print(
        "Manual review is required before "
        "changing status to complete."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())