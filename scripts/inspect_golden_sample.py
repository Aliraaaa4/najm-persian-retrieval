"""Inspect structural patterns inside one parser golden sample."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

def configure_utf8_output() -> None:
    """Force UTF-8 output, including redirected Windows output."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAGE_PATTERN = re.compile(
    r"PageV(?P<volume>\d+)P(?P<page>\d+)",
    flags=re.IGNORECASE,
)

MILESTONE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"ms(?P<number>\d+)"
    r"(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)

IMAGE_PATTERN = re.compile(
    r"!\[[^\]]*\]\((?P<target>[^)]+)\)"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect the physical-line and marker structure "
            "of one golden sample."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to one golden JSON sample.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def visible_text(text: str, limit: int = 150) -> str:
    """Return a readable one-line preview."""

    visible = (
        text
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )

    if len(visible) > limit:
        visible = visible[: limit - 3] + "..."

    return visible


def classify_line(text: str) -> str:
    """Assign a preliminary physical-line category."""

    without_newline = text.rstrip("\r\n")
    stripped = without_newline.strip()

    if stripped == "":
        return "blank"

    if IMAGE_PATTERN.search(without_newline):
        return "image_reference"

    if stripped.startswith("###"):
        return "heading"

    if stripped.startswith("~~"):
        return "continuation"

    if stripped.startswith("#"):
        if "%~%" in without_newline:
            return "verse_like"

        return "paragraph"

    if PAGE_PATTERN.fullmatch(stripped):
        return "page_marker_only"

    return "unclassified"


def load_sample(path: Path) -> dict[str, Any]:
    """Load one golden sample."""

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise ValueError(
            f"Sample file does not exist: {path}"
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
    """Inspect one sample and print a structural report."""
    
    configure_utf8_output()
    args = parse_args()
    path = resolve_path(args.path)

    try:
        data = load_sample(path)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    lines = data.get("lines")
    annotations = data.get("annotations")

    if not isinstance(lines, list):
        print("ERROR: sample lines must be an array.")
        return 1

    if not isinstance(annotations, dict):
        print("ERROR: annotations must be an object.")
        return 1

    categories: Counter[str] = Counter()

    page_markers: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    inline_marker_lines: list[dict[str, Any]] = []
    unclassified_lines: list[dict[str, Any]] = []

    print("Sample information")
    print("------------------")
    print("File:", path)
    print("Sample ID:", data.get("sample_id"))
    print("Profile:", data.get("profile"))
    print("Split:", data.get("split"))
    print(
        "Lines:",
        data.get("line_start"),
        "-",
        data.get("line_end"),
    )
    print(
        "Characters:",
        data.get("char_start"),
        "-",
        data.get("char_end"),
    )
    print(
        "Annotation status:",
        annotations.get("status"),
    )
    print(
        "Existing blocks:",
        len(annotations.get("blocks", [])),
    )
    print()

    for record in lines:
        if not isinstance(record, dict):
            print(
                "ERROR: every line record must be an object."
            )
            return 1

        line_number = record.get("line_number")
        text = record.get("text")

        if not isinstance(line_number, int):
            print("ERROR: invalid line_number.")
            return 1

        if not isinstance(text, str):
            print(
                f"ERROR: line {line_number} has invalid text."
            )
            return 1

        category = classify_line(text)
        categories[category] += 1

        page_matches = list(
            PAGE_PATTERN.finditer(text)
        )

        milestone_matches = list(
            MILESTONE_PATTERN.finditer(text)
        )

        image_matches = list(
            IMAGE_PATTERN.finditer(text)
        )

        for match in page_matches:
            page_markers.append(
                {
                    "line": line_number,
                    "volume": int(
                        match.group("volume")
                    ),
                    "page": int(
                        match.group("page")
                    ),
                    "start": match.start(),
                    "end": match.end(),
                    "text": text,
                }
            )

        for match in milestone_matches:
            milestones.append(
                {
                    "line": line_number,
                    "number": int(
                        match.group("number")
                    ),
                    "start": match.start(),
                    "end": match.end(),
                    "text": text,
                }
            )

        for match in image_matches:
            images.append(
                {
                    "line": line_number,
                    "target": match.group("target"),
                    "start": match.start(),
                    "end": match.end(),
                    "text": text,
                }
            )

        marker_count = (
            len(page_matches)
            + len(milestone_matches)
            + len(image_matches)
        )

        stripped = text.strip()

        marker_only = (
            category
            in {
                "page_marker_only",
                "image_reference",
            }
        )

        if marker_count and not marker_only:
            inline_marker_lines.append(
                {
                    "line": line_number,
                    "category": category,
                    "text": text,
                }
            )

        if category == "unclassified":
            unclassified_lines.append(
                {
                    "line": line_number,
                    "text": text,
                }
            )

    print("Physical-line categories")
    print("------------------------")

    for category in sorted(categories):
        print(
            f"{category:<22}",
            categories[category],
        )

    print()
    print("Page markers")
    print("------------")

    if not page_markers:
        print("none")
    else:
        for item in page_markers:
            print(
                f"line {item['line']}: "
                f"V{item['volume']} P{item['page']} "
                f"span={item['start']}:{item['end']} "
                f"{visible_text(item['text'])!r}"
            )

    print()
    print("Milestones")
    print("----------")

    if not milestones:
        print("none")
    else:
        for item in milestones:
            print(
                f"line {item['line']}: "
                f"ms{item['number']} "
                f"span={item['start']}:{item['end']} "
                f"{visible_text(item['text'])!r}"
            )

    print()
    print("Image references")
    print("----------------")

    if not images:
        print("none")
    else:
        for item in images:
            print(
                f"line {item['line']}: "
                f"target={item['target']!r} "
                f"{visible_text(item['text'])!r}"
            )

    print()
    print("Lines containing inline markers")
    print("-------------------------------")

    if not inline_marker_lines:
        print("none")
    else:
        for item in inline_marker_lines:
            print(
                f"line {item['line']} "
                f"[{item['category']}]: "
                f"{visible_text(item['text'])!r}"
            )

    print()
    print("Unclassified lines")
    print("------------------")

    if not unclassified_lines:
        print("none")
    else:
        for item in unclassified_lines:
            print(
                f"line {item['line']}: "
                f"{visible_text(item['text'])!r}"
            )

    print()
    print("All physical lines")
    print("------------------")

    for record in lines:
        line_number = record["line_number"]
        text = record["text"]
        category = classify_line(text)

        print(
            f"{line_number} | "
            f"{category:<18} | "
            f"{visible_text(text)!r}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())