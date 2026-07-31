"""Tests for conservative raw OCR golden drafting."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import pytest

from najm_retrieval.parsing.raw_ocr_drafting import (
    apply_raw_ocr_draft,
    draft_raw_ocr_blocks,
)


def make_raw_ocr_sample() -> dict[str, Any]:
    """Create a small lossless raw OCR sample."""

    line_texts = [
        "PageV00P001\r\n",
        "\r\n",
        "~~OCR text remains raw\r\n",
        "~~road ms007 one stage\r\n",
        "# heading-like OCR remains raw\r\n",
        "~~PageV00P002\r\n",
    ]

    lines: list[dict[str, Any]] = []

    offset = 1000

    for index, text in enumerate(
        line_texts
    ):
        line_number = 500 + index

        lines.append(
            {
                "line_number": line_number,
                "char_start": offset,
                "char_end": offset + len(text),
                "text": text,
            }
        )

        offset += len(text)

    return {
        "sample_id": "synthetic_raw_ocr",
        "profile": "raw_ocr_reference",
        "split": "development",
        "line_start": 500,
        "line_end": 505,
        "char_start": 1000,
        "char_end": offset,
        "lines": lines,
        "annotations": {
            "schema_version": 1,
            "status": "draft",
            "blocks": [],
            "notes": "",
        },
    }


def test_raw_ocr_draft_is_lossless() -> None:
    sample = make_raw_ocr_sample()

    blocks = draft_raw_ocr_blocks(
        sample
    )

    reconstructed = "".join(
        block["raw_text"]
        for block in blocks
    )

    expected = "".join(
        line["text"]
        for line in sample["lines"]
    )

    assert reconstructed == expected


def test_raw_ocr_draft_detects_types() -> None:
    sample = make_raw_ocr_sample()

    blocks = draft_raw_ocr_blocks(
        sample
    )

    counts = Counter(
        block["block_type"]
        for block in blocks
    )

    assert counts == {
        "page_marker": 2,
        "blank": 1,
        "raw": 4,
        "milestone": 1,
    }


def test_content_is_kept_conservatively_raw() -> None:
    sample = make_raw_ocr_sample()

    blocks = draft_raw_ocr_blocks(
        sample
    )

    heading_like = [
        block
        for block in blocks
        if (
            block["line_start"] == 504
            and block["block_type"] == "raw"
        )
    ]

    assert len(heading_like) == 1

    assert heading_like[0]["raw_text"] == (
        "# heading-like OCR remains raw\r\n"
    )


def test_inline_milestone_preserves_raw_group() -> None:
    sample = make_raw_ocr_sample()

    blocks = draft_raw_ocr_blocks(
        sample
    )

    milestone_index = next(
        index
        for index, block in enumerate(blocks)
        if block["block_type"] == "milestone"
    )

    before = blocks[milestone_index - 1]
    marker = blocks[milestone_index]
    after = blocks[milestone_index + 1]

    assert marker["attributes"] == {
        "number": 7,
    }

    assert before["block_type"] == "raw"
    assert after["block_type"] == "raw"

    assert (
        before["group_id"]
        == after["group_id"]
        == "raw_ocr_0002"
    )


def test_marker_only_pages_are_preserved() -> None:
    sample = make_raw_ocr_sample()

    blocks = draft_raw_ocr_blocks(
        sample
    )

    pages = [
        block
        for block in blocks
        if block["block_type"]
        == "page_marker"
    ]

    assert len(pages) == 2

    assert pages[0]["attributes"] == {
        "volume": 0,
        "page": 1,
    }

    assert pages[1]["attributes"] == {
        "volume": 0,
        "page": 2,
    }

    assert pages[1]["raw_text"] == (
        "~~PageV00P002\r\n"
    )


def test_apply_raw_ocr_draft_remains_draft() -> None:
    sample = make_raw_ocr_sample()
    original = deepcopy(sample)

    updated = apply_raw_ocr_draft(
        sample
    )

    assert sample == original
    assert updated is not sample

    assert (
        updated["annotations"]["status"]
        == "draft"
    )

    assert updated["annotations"]["blocks"]


def test_apply_refuses_existing_blocks() -> None:
    sample = make_raw_ocr_sample()

    sample["annotations"]["blocks"] = [
        {
            "block_id": "existing",
        }
    ]

    with pytest.raises(
        ValueError,
        match="already contains blocks",
    ):
        apply_raw_ocr_draft(sample)


def test_wrong_profile_is_rejected() -> None:
    sample = make_raw_ocr_sample()
    sample["profile"] = "mixed_prose_ocr"

    with pytest.raises(
        ValueError,
        match="raw_ocr_reference",
    ):
        draft_raw_ocr_blocks(sample)