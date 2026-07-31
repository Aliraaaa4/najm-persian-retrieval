"""Tests for structured-poetry golden drafting."""

from __future__ import annotations

from collections import Counter
from typing import Any

from najm_retrieval.parsing.golden_drafting import (
    apply_structured_poetry_draft,
)
from najm_retrieval.parsing.goldens import (
    validate_golden_data,
)


def make_structured_poetry_sample() -> dict[str, Any]:
    """Create a small structured-poetry sample."""

    line_texts = [
        "~~PageV1P001\n",
        "### || [genre: ms16 R]\n",
        (
            "# 1 متن نخست ms17 ادامه متن "
            "%~% مصراع دوم PageV1P002\n"
        ),
        "\n",
    ]

    sample_char_start = 100
    cursor = sample_char_start
    lines: list[dict[str, Any]] = []

    for line_number, text in enumerate(
        line_texts,
        start=1,
    ):
        char_start = cursor
        char_end = char_start + len(text)

        lines.append(
            {
                "line_number": line_number,
                "char_start": char_start,
                "char_end": char_end,
                "text": text,
            }
        )

        cursor = char_end

    return {
        "schema_version": 1,
        "sample_id": "structured_test",
        "split": "development",
        "version_id": "test.poetry",
        "profile": "structured_poetry",
        "source_path": "test.poetry",
        "body_line_start": 1,
        "line_start": 1,
        "line_end": len(lines),
        "char_start": sample_char_start,
        "char_end": cursor,
        "raw_text": "".join(line_texts),
        "lines": lines,
        "annotations": {
            "schema_version": 1,
            "status": "draft",
            "blocks": [],
            "notes": "",
        },
    }


def test_structured_poetry_draft_is_lossless() -> None:
    sample = make_structured_poetry_sample()

    updated = apply_structured_poetry_draft(
        sample
    )

    result = validate_golden_data(updated)

    assert result.is_valid is True
    assert result.coverage_ratio == 1.0
    assert result.uncovered_chars == 0
    assert result.overlapping_chars == 0
    assert result.reconstruction_matches is True


def test_structured_poetry_draft_detects_markers() -> None:
    sample = make_structured_poetry_sample()

    updated = apply_structured_poetry_draft(
        sample
    )

    blocks = updated[
        "annotations"
    ]["blocks"]

    counts = Counter(
        block["block_type"]
        for block in blocks
    )

    assert counts["page_marker"] == 2
    assert counts["milestone"] == 2
    assert counts["blank"] == 1
    assert counts["heading"] >= 1
    assert counts["verse"] >= 1


def test_marker_only_prefix_is_preserved() -> None:
    sample = make_structured_poetry_sample()

    updated = apply_structured_poetry_draft(
        sample
    )

    first_block = updated[
        "annotations"
    ]["blocks"][0]

    assert (
        first_block["block_type"]
        == "page_marker"
    )

    assert (
        first_block["raw_text"]
        == "~~PageV1P001\n"
    )


def test_inline_marker_fragments_share_verse_group() -> None:
    sample = make_structured_poetry_sample()

    updated = apply_structured_poetry_draft(
        sample
    )

    verse_blocks = [
        block
        for block in updated[
            "annotations"
        ]["blocks"]
        if block["block_type"] == "verse"
    ]

    group_ids = {
        block["group_id"]
        for block in verse_blocks
    }

    assert len(group_ids) == 1
    assert None not in group_ids


def test_draft_remains_draft() -> None:
    sample = make_structured_poetry_sample()

    updated = apply_structured_poetry_draft(
        sample
    )

    annotations = updated[
        "annotations"
    ]

    assert annotations["schema_version"] == 1
    assert annotations["status"] == "draft"
    assert annotations["blocks"]
    assert "Manual review" in annotations["notes"]