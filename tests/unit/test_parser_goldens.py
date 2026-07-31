"""Tests for parser golden annotation validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from najm_retrieval.parsing.goldens import (
    validate_golden_data,
)


def make_complete_sample() -> dict[str, Any]:
    """Create a minimal valid complete golden sample."""

    first_line = "PageV1P001\n"
    second_line = "متن آزمایشی\n"

    sample_char_start = 100

    first_char_start = sample_char_start
    first_char_end = (
        first_char_start + len(first_line)
    )

    second_char_start = first_char_end
    second_char_end = (
        second_char_start + len(second_line)
    )

    raw_text = first_line + second_line

    return {
        "schema_version": 1,
        "sample_id": "test_sample",
        "split": "development",
        "version_id": "test.version",
        "profile": "mixed_prose_ocr",
        "source_path": "test.version",
        "body_line_start": 1,
        "line_start": 1,
        "line_end": 2,
        "char_start": sample_char_start,
        "char_end": second_char_end,
        "raw_text": raw_text,
        "lines": [
            {
                "line_number": 1,
                "char_start": first_char_start,
                "char_end": first_char_end,
                "text": first_line,
            },
            {
                "line_number": 2,
                "char_start": second_char_start,
                "char_end": second_char_end,
                "text": second_line,
            },
        ],
        "annotations": {
            "schema_version": 1,
            "status": "complete",
            "blocks": [
                {
                    "block_id": "b0001",
                    "block_type": "page_marker",
                    "char_start": first_char_start,
                    "char_end": first_char_end,
                    "line_start": 1,
                    "line_end": 1,
                    "raw_text": first_line,
                    "group_id": None,
                    "attributes": {
                        "volume": 1,
                        "page": 1,
                    },
                },
                {
                    "block_id": "b0002",
                    "block_type": "paragraph",
                    "char_start": second_char_start,
                    "char_end": second_char_end,
                    "line_start": 2,
                    "line_end": 2,
                    "raw_text": second_line,
                    "group_id": None,
                    "attributes": {},
                },
            ],
            "notes": "",
        },
    }


def test_complete_golden_passes() -> None:
    sample = make_complete_sample()

    result = validate_golden_data(
        sample,
        require_complete=True,
    )

    assert result.is_valid is True
    assert result.coverage_ratio == 1.0
    assert result.uncovered_chars == 0
    assert result.overlapping_chars == 0
    assert result.reconstruction_matches is True


def test_draft_golden_may_have_no_blocks() -> None:
    sample = make_complete_sample()

    sample["annotations"] = {
        "schema_version": 1,
        "status": "draft",
        "blocks": [],
        "notes": "",
    }

    result = validate_golden_data(sample)

    assert result.is_valid is True
    assert result.covered_chars == 0
    assert result.uncovered_chars == len(
        sample["raw_text"]
    )
    assert result.reconstruction_matches is False
    assert result.warnings


def test_require_complete_rejects_draft() -> None:
    sample = make_complete_sample()

    sample["annotations"]["status"] = "draft"
    sample["annotations"]["blocks"] = []

    result = validate_golden_data(
        sample,
        require_complete=True,
    )

    assert result.is_valid is False
    assert "complete annotation is required" in (
        result.errors
    )


def test_complete_golden_detects_gap() -> None:
    sample = make_complete_sample()

    second_block = sample[
        "annotations"
    ]["blocks"][1]

    second_block["char_start"] += 1
    second_block["raw_text"] = second_block[
        "raw_text"
    ][1:]

    result = validate_golden_data(
        sample,
        require_complete=True,
    )

    assert result.is_valid is False
    assert result.uncovered_chars == 1


def test_complete_golden_detects_overlap() -> None:
    sample = make_complete_sample()

    second_block = sample[
        "annotations"
    ]["blocks"][1]

    second_block["char_start"] -= 1

    relative_start = (
        second_block["char_start"]
        - sample["char_start"]
    )

    relative_end = (
        second_block["char_end"]
        - sample["char_start"]
    )

    second_block["raw_text"] = sample[
        "raw_text"
    ][relative_start:relative_end]

    result = validate_golden_data(
        sample,
        require_complete=True,
    )

    assert result.is_valid is False
    assert result.overlapping_chars == 1


def test_golden_detects_raw_text_mismatch() -> None:
    sample = make_complete_sample()

    sample["annotations"]["blocks"][1][
        "raw_text"
    ] = "متن اشتباه\n"

    result = validate_golden_data(sample)

    assert result.is_valid is False

    assert any(
        "raw_text does not match" in error
        for error in result.errors
    )


def test_page_marker_attributes_are_validated() -> None:
    sample = make_complete_sample()

    sample["annotations"]["blocks"][0][
        "attributes"
    ]["page"] = 999

    result = validate_golden_data(sample)

    assert result.is_valid is False

    assert any(
        "attributes.page should be 1" in error
        for error in result.errors
    )


def test_duplicate_block_ids_are_rejected() -> None:
    sample = make_complete_sample()

    sample["annotations"]["blocks"][1][
        "block_id"
    ] = "b0001"

    result = validate_golden_data(sample)

    assert result.is_valid is False
    assert "duplicate block_id: b0001" in (
        result.errors
    )


def test_invalid_block_type_is_rejected() -> None:
    sample = make_complete_sample()

    sample["annotations"]["blocks"][1][
        "block_type"
    ] = "unknown_type"

    result = validate_golden_data(sample)

    assert result.is_valid is False