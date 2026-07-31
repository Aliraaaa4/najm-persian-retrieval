"""Loading and validation utilities for parser golden annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


ALLOWED_BLOCK_TYPES = {
    "page_marker",
    "image_reference",
    "milestone",
    "heading",
    "section",
    "verse",
    "paragraph",
    "blank",
    "raw",
}

ALLOWED_ANNOTATION_STATUSES = {
    "draft",
    "complete",
}

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


@dataclass(frozen=True)
class GoldenValidationResult:
    """Result of validating one golden annotation sample."""

    sample_id: str
    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    total_chars: int
    covered_chars: int
    uncovered_chars: int
    overlapping_chars: int

    reconstruction_matches: bool

    @property
    def is_valid(self) -> bool:
        """Return whether validation completed without errors."""

        return not self.errors

    @property
    def coverage_ratio(self) -> float:
        """Return the character coverage ratio."""

        if self.total_chars == 0:
            return 1.0

        return self.covered_chars / self.total_chars


def load_golden_file(
    path: Path,
) -> dict[str, Any]:
    """Load one golden JSON file."""

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Golden file does not exist: {path}"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Golden JSON top-level value must be an object."
        )

    return data


def validate_golden_file(
    path: Path,
    *,
    require_complete: bool = False,
) -> GoldenValidationResult:
    """Load and validate one golden annotation file."""

    data = load_golden_file(path)

    return validate_golden_data(
        data,
        require_complete=require_complete,
    )


def validate_golden_data(
    data: dict[str, Any],
    *,
    require_complete: bool = False,
) -> GoldenValidationResult:
    """Validate one loaded golden annotation object."""

    errors: list[str] = []
    warnings: list[str] = []

    sample_id_value = data.get("sample_id")
    sample_id = (
        sample_id_value
        if isinstance(sample_id_value, str)
        else "<unknown>"
    )

    raw_text_value = data.get("raw_text")
    raw_text = (
        raw_text_value
        if isinstance(raw_text_value, str)
        else ""
    )

    sample_char_start = data.get("char_start")
    sample_char_end = data.get("char_end")
    line_records = data.get("lines")
    annotations = data.get("annotations")

    if not isinstance(sample_id_value, str):
        errors.append(
            "sample_id must be a string"
        )

    if not isinstance(raw_text_value, str):
        errors.append(
            "raw_text must be a string"
        )

    if not isinstance(sample_char_start, int):
        errors.append(
            "char_start must be an integer"
        )

    if not isinstance(sample_char_end, int):
        errors.append(
            "char_end must be an integer"
        )

    if (
        isinstance(sample_char_start, int)
        and isinstance(sample_char_end, int)
    ):
        if sample_char_end < sample_char_start:
            errors.append(
                "sample char_end must not be smaller "
                "than char_start"
            )

        if (
            isinstance(raw_text_value, str)
            and sample_char_end - sample_char_start
            != len(raw_text_value)
        ):
            errors.append(
                "sample character range does not match "
                "raw_text length"
            )

    if not isinstance(line_records, list):
        errors.append(
            "lines must be an array"
        )
        line_records = []

    if not isinstance(annotations, dict):
        errors.append(
            "annotations must be an object"
        )
        annotations = {}

    annotation_schema_version = annotations.get(
        "schema_version"
    )

    if annotation_schema_version != 1:
        errors.append(
            "annotations.schema_version must equal 1"
        )

    status_value = annotations.get("status")
    status = (
        status_value
        if isinstance(status_value, str)
        else "<invalid>"
    )

    if status not in ALLOWED_ANNOTATION_STATUSES:
        errors.append(
            "annotations.status must be "
            "'draft' or 'complete'"
        )

    notes = annotations.get("notes")

    if not isinstance(notes, str):
        errors.append(
            "annotations.notes must be a string"
        )

    blocks_value = annotations.get("blocks")

    if not isinstance(blocks_value, list):
        errors.append(
            "annotations.blocks must be an array"
        )
        blocks: list[Any] = []
    else:
        blocks = blocks_value

    if require_complete and status != "complete":
        errors.append(
            "complete annotation is required"
        )

    if status == "draft":
        warnings.append(
            "annotation status is draft"
        )

    total_chars = len(raw_text)
    coverage = [0] * total_chars

    block_raw_texts: list[str] = []
    block_ids: set[str] = set()

    previous_char_start: int | None = None
    previous_char_end: int | None = None

    for block_index, block in enumerate(
        blocks,
        start=1,
    ):
        prefix = f"block {block_index}"

        if not isinstance(block, dict):
            errors.append(
                f"{prefix} must be an object"
            )
            continue

        block_id = block.get("block_id")
        block_type = block.get("block_type")
        char_start = block.get("char_start")
        char_end = block.get("char_end")
        line_start = block.get("line_start")
        line_end = block.get("line_end")
        block_raw_text = block.get("raw_text")
        group_id = block.get("group_id")
        attributes = block.get("attributes")

        if not isinstance(block_id, str):
            errors.append(
                f"{prefix}.block_id must be a string"
            )
        else:
            if block_id in block_ids:
                errors.append(
                    f"duplicate block_id: {block_id}"
                )

            block_ids.add(block_id)

        if block_type not in ALLOWED_BLOCK_TYPES:
            errors.append(
                f"{prefix}.block_type is invalid: "
                f"{block_type!r}"
            )

        if not isinstance(char_start, int):
            errors.append(
                f"{prefix}.char_start must be an integer"
            )

        if not isinstance(char_end, int):
            errors.append(
                f"{prefix}.char_end must be an integer"
            )

        if not isinstance(line_start, int):
            errors.append(
                f"{prefix}.line_start must be an integer"
            )

        if not isinstance(line_end, int):
            errors.append(
                f"{prefix}.line_end must be an integer"
            )

        if not isinstance(block_raw_text, str):
            errors.append(
                f"{prefix}.raw_text must be a string"
            )

        if group_id is not None and not isinstance(
            group_id,
            str,
        ):
            errors.append(
                f"{prefix}.group_id must be a string "
                "or null"
            )

        if not isinstance(attributes, dict):
            errors.append(
                f"{prefix}.attributes must be an object"
            )
            attributes = {}

        if not (
            isinstance(char_start, int)
            and isinstance(char_end, int)
            and isinstance(sample_char_start, int)
            and isinstance(sample_char_end, int)
        ):
            continue

        if char_end <= char_start:
            errors.append(
                f"{prefix} must have a positive "
                "character length"
            )
            continue

        if char_start < sample_char_start:
            errors.append(
                f"{prefix} starts before the sample"
            )

        if char_end > sample_char_end:
            errors.append(
                f"{prefix} ends after the sample"
            )

        if (
            previous_char_start is not None
            and char_start < previous_char_start
        ):
            errors.append(
                f"{prefix} is not stored in character "
                "offset order"
            )

        if (
            previous_char_end is not None
            and char_start < previous_char_end
        ):
            errors.append(
                f"{prefix} overlaps the previous block"
            )

        previous_char_start = char_start
        previous_char_end = char_end

        relative_start = (
            char_start - sample_char_start
        )
        relative_end = (
            char_end - sample_char_start
        )

        if (
            relative_start < 0
            or relative_end > total_chars
        ):
            continue

        expected_raw_text = raw_text[
            relative_start:relative_end
        ]

        if (
            isinstance(block_raw_text, str)
            and block_raw_text
            != expected_raw_text
        ):
            errors.append(
                f"{prefix}.raw_text does not match "
                "the sample character range"
            )

        if isinstance(block_raw_text, str):
            block_raw_texts.append(
                block_raw_text
            )

        for position in range(
            relative_start,
            relative_end,
        ):
            coverage[position] += 1

        if (
            isinstance(line_start, int)
            and isinstance(line_end, int)
        ):
            expected_line_start = (
                _line_number_for_character(
                    line_records,
                    char_start,
                )
            )

            expected_line_end = (
                _line_number_for_character(
                    line_records,
                    char_end - 1,
                )
            )

            if (
                expected_line_start is not None
                and line_start != expected_line_start
            ):
                errors.append(
                    f"{prefix}.line_start should be "
                    f"{expected_line_start}"
                )

            if (
                expected_line_end is not None
                and line_end != expected_line_end
            ):
                errors.append(
                    f"{prefix}.line_end should be "
                    f"{expected_line_end}"
                )

        _validate_block_attributes(
            block_index=block_index,
            block_type=block_type,
            raw_text=(
                block_raw_text
                if isinstance(block_raw_text, str)
                else ""
            ),
            attributes=attributes,
            errors=errors,
        )

    covered_chars = sum(
        value >= 1
        for value in coverage
    )

    uncovered_chars = sum(
        value == 0
        for value in coverage
    )

    overlapping_chars = sum(
        value > 1
        for value in coverage
    )

    reconstructed_text = "".join(
        block_raw_texts
    )

    reconstruction_matches = (
        reconstructed_text == raw_text
    )

    if status == "complete":
        if not blocks and total_chars > 0:
            errors.append(
                "complete annotation must contain blocks"
            )

        if uncovered_chars != 0:
            errors.append(
                f"complete annotation has "
                f"{uncovered_chars} uncovered characters"
            )

        if overlapping_chars != 0:
            errors.append(
                f"complete annotation has "
                f"{overlapping_chars} overlapping characters"
            )

        if not reconstruction_matches:
            errors.append(
                "complete annotation does not exactly "
                "reconstruct sample raw_text"
            )

        if blocks:
            first_block = blocks[0]
            last_block = blocks[-1]

            if (
                isinstance(first_block, dict)
                and first_block.get("char_start")
                != sample_char_start
            ):
                errors.append(
                    "first complete block must start "
                    "at sample char_start"
                )

            if (
                isinstance(last_block, dict)
                and last_block.get("char_end")
                != sample_char_end
            ):
                errors.append(
                    "last complete block must end "
                    "at sample char_end"
                )
    else:
        if uncovered_chars:
            warnings.append(
                f"draft annotation currently has "
                f"{uncovered_chars} uncovered characters"
            )

        if overlapping_chars:
            errors.append(
                f"draft annotation has "
                f"{overlapping_chars} overlapping characters"
            )

    return GoldenValidationResult(
        sample_id=sample_id,
        status=status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        total_chars=total_chars,
        covered_chars=covered_chars,
        uncovered_chars=uncovered_chars,
        overlapping_chars=overlapping_chars,
        reconstruction_matches=reconstruction_matches,
    )


def _line_number_for_character(
    lines: list[Any],
    character_offset: int,
) -> int | None:
    """Return the source line containing one character."""

    for line in lines:
        if not isinstance(line, dict):
            continue

        char_start = line.get("char_start")
        char_end = line.get("char_end")
        line_number = line.get("line_number")

        if not (
            isinstance(char_start, int)
            and isinstance(char_end, int)
            and isinstance(line_number, int)
        ):
            continue

        if char_start <= character_offset < char_end:
            return line_number

    return None


def _validate_block_attributes(
    *,
    block_index: int,
    block_type: Any,
    raw_text: str,
    attributes: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate type-specific marker attributes."""

    prefix = f"block {block_index}"

    if block_type == "page_marker":
        match = PAGE_PATTERN.search(raw_text)

        if match is None:
            errors.append(
                f"{prefix} is page_marker but contains "
                "no valid page marker"
            )
            return

        volume = attributes.get("volume")
        page = attributes.get("page")

        expected_volume = int(
            match.group("volume")
        )

        expected_page = int(
            match.group("page")
        )

        if volume != expected_volume:
            errors.append(
                f"{prefix}.attributes.volume should be "
                f"{expected_volume}"
            )

        if page != expected_page:
            errors.append(
                f"{prefix}.attributes.page should be "
                f"{expected_page}"
            )

    elif block_type == "milestone":
        match = MILESTONE_PATTERN.search(
            raw_text
        )

        if match is None:
            errors.append(
                f"{prefix} is milestone but contains "
                "no valid milestone"
            )
            return

        expected_number = int(
            match.group("number")
        )

        if attributes.get("number") != expected_number:
            errors.append(
                f"{prefix}.attributes.number should be "
                f"{expected_number}"
            )

    elif block_type == "image_reference":
        match = IMAGE_PATTERN.search(
            raw_text
        )

        if match is None:
            errors.append(
                f"{prefix} is image_reference but contains "
                "no valid Markdown image reference"
            )
            return

        expected_target = match.group(
            "target"
        )

        if attributes.get("target") != expected_target:
            errors.append(
                f"{prefix}.attributes.target should be "
                f"{expected_target!r}"
            )

    elif block_type == "blank":
        if raw_text.strip(" \t\r\n") != "":
            errors.append(
                f"{prefix} is blank but contains "
                "non-whitespace text"
            )