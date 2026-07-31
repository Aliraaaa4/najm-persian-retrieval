"""Validate the seven parser pilot samples.

This script checks both the internal JSON structure and the exact
relationship between each pilot sample and its original OpenITI source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_DIRECTORY = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "parser_goldens"
)

CORPUS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PER0675AH"
    / "data"
)


EXPECTED_SAMPLES: dict[str, dict[str, Any]] = {
    "pilot_baba_afzal_01": {
        "version_id": "0667BabaAfzal.Diwan.PDL00046-per1",
        "profile": "structured_poetry",
        "line_start": 557,
        "line_end": 617,
    },
    "pilot_masalik_aocp_01": {
        "version_id": (
            "0670IbnAscadHanati."
            "Masalik."
            "AOCP202605141788-per1"
        ),
        "profile": "mixed_prose_ocr",
        "line_start": 5284,
        "line_end": 5344,
    },
    "pilot_masalik_kraken_01": {
        "version_id": (
            "0670IbnAscadHanati."
            "Masalik."
            "Kraken220107010708-per1"
        ),
        "profile": "raw_ocr_reference",
        "line_start": 10206,
        "line_end": 10266,
    },
    "pilot_rumi_diwan_01": {
        "version_id": (
            "0672JalalDinRumi."
            "Diwan."
            "PDL00047-per1"
        ),
        "profile": "structured_poetry",
        "line_start": 45554,
        "line_end": 45625,
    },
    "pilot_majalis_01": {
        "version_id": (
            "0672JalalDinRumi."
            "MajalisSabica."
            "AOCP202502141236-per1"
        ),
        "profile": "mixed_prose_ocr",
        "line_start": 2315,
        "line_end": 2389,
    },
    "pilot_mathnawi_01": {
        "version_id": (
            "0672JalalDinRumi."
            "Mathnawi."
            "PDL00048-per1"
        ),
        "profile": "structured_poetry",
        "line_start": 12954,
        "line_end": 13014,
    },
    "pilot_akhlaq_01": {
        "version_id": (
            "0672NasirDinTusi."
            "AkhlaqMuhtashami."
            "AOCP202502141237-per1"
        ),
        "profile": "mixed_prose_ocr",
        "line_start": 171,
        "line_end": 231,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    """Read and validate the top-level JSON value."""

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level JSON value must be an object: {path}"
        )

    return data


def read_source_text(path: Path) -> str:
    """Read source text while preserving original newline characters."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Source file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return handle.read()


def build_line_offsets(
    lines: list[str],
) -> list[tuple[int, int]]:
    """Return absolute half-open character offsets for all source lines."""

    offsets: list[tuple[int, int]] = []
    cursor = 0

    for line in lines:
        char_start = cursor
        char_end = char_start + len(line)

        offsets.append(
            (
                char_start,
                char_end,
            )
        )

        cursor = char_end

    return offsets


def validate_required_fields(
    data: dict[str, Any],
    errors: list[str],
) -> None:
    """Check required top-level fields."""

    required_fields = {
        "schema_version",
        "sample_id",
        "split",
        "version_id",
        "profile",
        "source_path",
        "body_line_start",
        "line_start",
        "line_end",
        "char_start",
        "char_end",
        "raw_text",
        "lines",
        "annotations",
    }

    missing = sorted(
        required_fields - data.keys()
    )

    for field_name in missing:
        errors.append(
            f"missing required field: {field_name}"
        )


def validate_expected_metadata(
    data: dict[str, Any],
    errors: list[str],
) -> None:
    """Compare the sample with its expected pilot definition."""

    sample_id = data.get("sample_id")

    if not isinstance(sample_id, str):
        errors.append(
            "sample_id must be a string"
        )
        return

    expected = EXPECTED_SAMPLES.get(sample_id)

    if expected is None:
        errors.append(
            f"unexpected sample_id: {sample_id}"
        )
        return

    for field_name, expected_value in expected.items():
        actual_value = data.get(field_name)

        if actual_value != expected_value:
            errors.append(
                f"{field_name}: expected "
                f"{expected_value!r}, got {actual_value!r}"
            )

    if data.get("schema_version") != 1:
        errors.append(
            "schema_version must equal 1"
        )

    if data.get("split") != "development":
        errors.append(
            "pilot split must equal development"
        )


def validate_annotations(
    data: dict[str, Any],
    errors: list[str],
) -> None:
    """Confirm pilot annotations are still intentionally empty."""

    annotations = data.get("annotations")

    if not isinstance(annotations, dict):
        errors.append(
            "annotations must be an object"
        )
        return

    blocks = annotations.get("blocks")

    if blocks != []:
        errors.append(
            "annotations.blocks must still be empty"
        )

    notes = annotations.get("notes")

    if not isinstance(notes, str):
        errors.append(
            "annotations.notes must be a string"
        )


def validate_internal_lines(
    data: dict[str, Any],
    errors: list[str],
) -> None:
    """Check internal sample line continuity and reconstruction."""

    lines = data.get("lines")

    if not isinstance(lines, list):
        errors.append(
            "lines must be an array"
        )
        return

    line_start = data.get("line_start")
    line_end = data.get("line_end")

    if not isinstance(line_start, int):
        errors.append(
            "line_start must be an integer"
        )
        return

    if not isinstance(line_end, int):
        errors.append(
            "line_end must be an integer"
        )
        return

    expected_line_count = (
        line_end - line_start + 1
    )

    if len(lines) != expected_line_count:
        errors.append(
            f"expected {expected_line_count} line records, "
            f"found {len(lines)}"
        )

    expected_line_numbers = list(
        range(
            line_start,
            line_end + 1,
        )
    )

    actual_line_numbers: list[int] = []

    for line in lines:
        if not isinstance(line, dict):
            errors.append(
                "every line record must be an object"
            )
            continue

        line_number = line.get("line_number")

        if isinstance(line_number, int):
            actual_line_numbers.append(
                line_number
            )
        else:
            errors.append(
                "line_number must be an integer"
            )

        if not isinstance(
            line.get("char_start"),
            int,
        ):
            errors.append(
                "line char_start must be an integer"
            )

        if not isinstance(
            line.get("char_end"),
            int,
        ):
            errors.append(
                "line char_end must be an integer"
            )

        if not isinstance(
            line.get("text"),
            str,
        ):
            errors.append(
                "line text must be a string"
            )

    if actual_line_numbers != expected_line_numbers:
        errors.append(
            "line numbers are not contiguous "
            "or do not match the requested range"
        )

    if not lines:
        errors.append(
            "sample must contain at least one line"
        )
        return

    reconstructed = "".join(
        line.get("text", "")
        for line in lines
        if isinstance(line, dict)
    )

    if reconstructed != data.get("raw_text"):
        errors.append(
            "raw_text does not equal the joined line text"
        )

    first_line = lines[0]
    last_line = lines[-1]

    if data.get("char_start") != first_line.get(
        "char_start"
    ):
        errors.append(
            "sample char_start does not match "
            "the first line char_start"
        )

    if data.get("char_end") != last_line.get(
        "char_end"
    ):
        errors.append(
            "sample char_end does not match "
            "the last line char_end"
        )

    for previous, current in zip(
        lines,
        lines[1:],
    ):
        if previous.get("char_end") != current.get(
            "char_start"
        ):
            errors.append(
                "gap or overlap found between "
                "adjacent line offsets"
            )
            break

    char_start = data.get("char_start")
    char_end = data.get("char_end")
    raw_text = data.get("raw_text")

    if (
        isinstance(char_start, int)
        and isinstance(char_end, int)
        and isinstance(raw_text, str)
    ):
        if char_end - char_start != len(raw_text):
            errors.append(
                "sample character range does not "
                "match raw_text length"
            )


def validate_against_source(
    data: dict[str, Any],
    errors: list[str],
) -> None:
    """Compare sample contents and offsets with the original source."""

    source_label = data.get("source_path")

    if not isinstance(source_label, str):
        errors.append(
            "source_path must be a string"
        )
        return

    source_path = (
        CORPUS_ROOT
        / Path(source_label)
    ).resolve()

    corpus_root_resolved = CORPUS_ROOT.resolve()

    try:
        source_path.relative_to(
            corpus_root_resolved
        )
    except ValueError:
        errors.append(
            "source_path escapes the configured corpus root"
        )
        return

    if not source_path.is_file():
        errors.append(
            f"source file does not exist: {source_path}"
        )
        return

    source_text = read_source_text(source_path)
    source_lines = source_text.splitlines(
        keepends=True
    )

    line_start = data.get("line_start")
    line_end = data.get("line_end")

    if not isinstance(line_start, int):
        return

    if not isinstance(line_end, int):
        return

    if line_start < 1:
        errors.append(
            "line_start must be at least 1"
        )
        return

    if line_end < line_start:
        errors.append(
            "line_end must not be smaller than line_start"
        )
        return

    if line_end > len(source_lines):
        errors.append(
            f"line_end={line_end} exceeds source "
            f"line count {len(source_lines)}"
        )
        return

    body_line_start = data.get(
        "body_line_start"
    )

    if (
        isinstance(body_line_start, int)
        and line_start < body_line_start
    ):
        errors.append(
            "pilot sample starts inside the OpenITI header"
        )

    selected_source_lines = source_lines[
        line_start - 1 : line_end
    ]

    expected_raw_text = "".join(
        selected_source_lines
    )

    if data.get("raw_text") != expected_raw_text:
        errors.append(
            "sample raw_text does not exactly match "
            "the original source range"
        )

    source_offsets = build_line_offsets(
        source_lines
    )

    expected_char_start = source_offsets[
        line_start - 1
    ][0]

    expected_char_end = source_offsets[
        line_end - 1
    ][1]

    if data.get("char_start") != expected_char_start:
        errors.append(
            f"sample char_start should be "
            f"{expected_char_start}, got "
            f"{data.get('char_start')}"
        )

    if data.get("char_end") != expected_char_end:
        errors.append(
            f"sample char_end should be "
            f"{expected_char_end}, got "
            f"{data.get('char_end')}"
        )

    sample_lines = data.get("lines")

    if not isinstance(sample_lines, list):
        return

    for position, sample_line in enumerate(
        sample_lines,
    ):
        if not isinstance(sample_line, dict):
            continue

        source_index = (
            line_start - 1 + position
        )

        source_line_text = source_lines[
            source_index
        ]

        source_char_start, source_char_end = (
            source_offsets[source_index]
        )

        expected_line_number = (
            line_start + position
        )

        if sample_line.get(
            "line_number"
        ) != expected_line_number:
            errors.append(
                f"line record {position} has incorrect "
                "line_number"
            )

        if sample_line.get(
            "char_start"
        ) != source_char_start:
            errors.append(
                f"source offset mismatch at line "
                f"{expected_line_number}: char_start"
            )

        if sample_line.get(
            "char_end"
        ) != source_char_end:
            errors.append(
                f"source offset mismatch at line "
                f"{expected_line_number}: char_end"
            )

        if sample_line.get("text") != source_line_text:
            errors.append(
                f"source text mismatch at line "
                f"{expected_line_number}"
            )


def validate_sample(
    path: Path,
) -> list[str]:
    """Return all validation errors for one pilot file."""

    errors: list[str] = []

    try:
        data = read_json(path)
    except (
        OSError,
        ValueError,
    ) as error:
        return [str(error)]

    validate_required_fields(
        data,
        errors,
    )

    if errors:
        return errors

    validate_expected_metadata(
        data,
        errors,
    )

    validate_annotations(
        data,
        errors,
    )

    validate_internal_lines(
        data,
        errors,
    )

    validate_against_source(
        data,
        errors,
    )

    return errors


def print_sample_summary(
    path: Path,
) -> None:
    """Display useful first and last lines after validation."""

    data = read_json(path)
    lines = data["lines"]

    print(
        f"  lines: "
        f"{data['line_start']}-"
        f"{data['line_end']}"
    )

    print(
        f"  chars: "
        f"{data['char_start']}-"
        f"{data['char_end']}"
    )

    print("  first lines:")

    for line in lines[:3]:
        print(
            f"    {line['line_number']}: "
            f"{line['text'].rstrip()!r}"
        )

    print("  last lines:")

    for line in lines[-3:]:
        print(
            f"    {line['line_number']}: "
            f"{line['text'].rstrip()!r}"
        )


def main() -> int:
    """Validate exactly seven expected pilot samples."""

    paths = sorted(
        GOLDEN_DIRECTORY.glob(
            "pilot_*.json"
        )
    )

    expected_file_count = len(
        EXPECTED_SAMPLES
    )

    if len(paths) != expected_file_count:
        print(
            f"FAIL: expected "
            f"{expected_file_count} pilot files, "
            f"found {len(paths)}."
        )

        return 1

    expected_filenames = {
        f"{sample_id}.json"
        for sample_id in EXPECTED_SAMPLES
    }

    actual_filenames = {
        path.name
        for path in paths
    }

    missing_files = sorted(
        expected_filenames - actual_filenames
    )

    unexpected_files = sorted(
        actual_filenames - expected_filenames
    )

    if missing_files:
        print("Missing pilot files:")

        for filename in missing_files:
            print(f"  - {filename}")

    if unexpected_files:
        print("Unexpected pilot files:")

        for filename in unexpected_files:
            print(f"  - {filename}")

    if missing_files or unexpected_files:
        return 1

    failed = False

    for path in paths:
        errors = validate_sample(path)

        if errors:
            failed = True

            print(f"FAIL: {path.name}")

            for error in errors:
                print(f"  - {error}")

            continue

        print(f"PASS: {path.name}")
        print_sample_summary(path)

    if failed:
        print()
        print(
            "One or more pilot samples failed validation."
        )

        return 1

    print()
    print(
        "All seven pilot samples passed validation."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())