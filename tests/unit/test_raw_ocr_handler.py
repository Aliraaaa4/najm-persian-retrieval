"""Tests for the conservative raw OCR production handler."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from najm_retrieval.parsing.core import (
    split_openiti_source,
)
from najm_retrieval.parsing.handlers.prose_ocr import (
    build_raw_ocr_records,
    parse_raw_ocr_reference,
)
from najm_retrieval.parsing.models import (
    BlockType,
)


VERSION_ID = (
    "0001Author.Work.Kraken-per1"
)


def make_source():
    """Create a small raw OCR OpenITI source."""

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Raw OCR Test\r\n"
        "#META#Header#End#\r\n"
        "PageV00P001\r\n"
        "\r\n"
        "~~OCR text remains raw\r\n"
        "~~road ms007 one stage\r\n"
        "# heading-like OCR remains raw\r\n"
        "~~PageV00P002\r\n"
    )

    return split_openiti_source(
        source_text,
        source_path=Path(VERSION_ID),
    )


def test_raw_ocr_records_reconstruct_body() -> None:
    source = make_source()

    records = build_raw_ocr_records(
        source
    )

    reconstructed = "".join(
        record["raw_text"]
        for record in records
    )

    assert reconstructed == source.body_text


def test_raw_ocr_handler_detects_types() -> None:
    source = make_source()

    records = build_raw_ocr_records(
        source
    )

    counts = Counter(
        record["block_type"]
        for record in records
    )

    assert counts == {
        "page_marker": 2,
        "blank": 1,
        "raw": 4,
        "milestone": 1,
    }


def test_heading_like_ocr_remains_raw() -> None:
    source = make_source()

    document, _ = parse_raw_ocr_reference(
        source
    )

    heading_like = [
        block
        for block in document.blocks
        if (
            block.span.line_start
            == source.body_line_start + 4
        )
    ]

    assert len(heading_like) == 1

    assert (
        heading_like[0].block_type
        == BlockType.RAW
    )

    assert heading_like[0].raw_text == (
        "# heading-like OCR remains raw\r\n"
    )


def test_inline_milestone_keeps_raw_group() -> None:
    source = make_source()

    document, _ = parse_raw_ocr_reference(
        source
    )

    milestone_index = next(
        index
        for index, block in enumerate(
            document.blocks
        )
        if block.block_type
        == BlockType.MILESTONE
    )

    before = document.blocks[
        milestone_index - 1
    ]

    marker = document.blocks[
        milestone_index
    ]

    after = document.blocks[
        milestone_index + 1
    ]

    assert marker.get_attribute(
        "number"
    ) == 7

    assert before.block_type == BlockType.RAW
    assert after.block_type == BlockType.RAW

    assert (
        before.get_attribute("group_id")
        == after.get_attribute("group_id")
    )

    assert (
        before.get_attribute("group_id")
        is not None
    )


def test_parse_raw_ocr_is_lossless() -> None:
    source = make_source()

    document, metrics = (
        parse_raw_ocr_reference(
            source
        )
    )

    assert document.version_id == VERSION_ID

    assert document.profile == (
        "raw_ocr_reference"
    )

    assert (
        document.reconstruct_body()
        == source.body_text
    )

    assert metrics.uncovered_chars == 0
    assert metrics.overlapping_chars == 0

    assert (
        metrics.reconstruction_matches_source
        is True
    )

    assert (
        metrics.passes_lossless_gate
        is True
    )
