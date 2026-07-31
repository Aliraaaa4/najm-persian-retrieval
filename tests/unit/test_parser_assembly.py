"""Tests for typed parser document assembly and metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from najm_retrieval.parsing.assembly import (
    build_parsed_document,
    compute_parse_metrics,
)
from najm_retrieval.parsing.core import (
    split_openiti_source,
)
from najm_retrieval.parsing.models import (
    BlockType,
)


VERSION_ID = "0001Author.Work.Version-per1"


def make_source():
    """Create one exact synthetic OpenITI source."""

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Test\r\n"
        "#META#Header#End#\r\n"
        "PageV01P001\r\n"
        "# متن نخست\r\n"
    )

    return split_openiti_source(
        source_text,
        source_path=Path(VERSION_ID),
    )


def make_complete_records(source):
    """Create records covering the complete body."""

    page_line = source.lines[0]
    paragraph_line = source.lines[1]

    return [
        {
            "block_id": "b0001",
            "block_type": "page_marker",
            "line_start": page_line.line_number,
            "line_end": page_line.line_number,
            "char_start": page_line.char_start,
            "char_end": page_line.char_end,
            "raw_text": page_line.text,
            "group_id": None,
            "attributes": {
                "volume": 1,
                "page": 1,
            },
        },
        {
            "block_id": "b0002",
            "block_type": "paragraph",
            "line_start": paragraph_line.line_number,
            "line_end": paragraph_line.line_number,
            "char_start": paragraph_line.char_start,
            "char_end": paragraph_line.char_end,
            "raw_text": paragraph_line.text,
            "group_id": "paragraph_0001",
            "attributes": {
                "continuation": False,
            },
        },
    ]


def test_build_document_creates_typed_blocks() -> None:
    source = make_source()
    records = make_complete_records(source)

    document = build_parsed_document(
        source=source,
        profile="mixed_prose_ocr",
        block_records=records,
        parser_name="test_parser",
        parser_version="0.1",
    )

    assert document.version_id == VERSION_ID
    assert document.profile == "mixed_prose_ocr"
    assert len(document.blocks) == 2

    page, paragraph = document.blocks

    assert page.block_id == (
        f"{VERSION_ID}:b0001"
    )

    assert page.block_type == (
        BlockType.PAGE_MARKER
    )

    assert page.page is not None
    assert page.page.volume == 1
    assert page.page.page == 1
    assert page.page.raw_marker == (
        "PageV01P001"
    )

    assert page.display_text == ""
    assert page.retrieval_text == ""

    assert paragraph.block_type == (
        BlockType.PARAGRAPH
    )

    assert paragraph.display_text == (
        "# متن نخست"
    )

    assert paragraph.retrieval_text == (
        "# متن نخست"
    )

    assert paragraph.get_attribute(
        "group_id"
    ) == "paragraph_0001"

    assert paragraph.get_attribute(
        "continuation"
    ) is False


def test_document_reconstructs_exact_body() -> None:
    source = make_source()

    document = build_parsed_document(
        source=source,
        profile="mixed_prose_ocr",
        block_records=make_complete_records(
            source
        ),
    )

    assert document.reconstruct_body() == (
        source.body_text
    )


def test_lossless_metrics_pass_for_complete_document() -> None:
    source = make_source()

    document = build_parsed_document(
        source=source,
        profile="mixed_prose_ocr",
        block_records=make_complete_records(
            source
        ),
    )

    metrics = compute_parse_metrics(
        source=source,
        document=document,
        runtime_seconds=0.05,
        peak_memory_bytes=2048,
    )

    assert metrics.total_body_lines == 2
    assert metrics.covered_lines == 2
    assert metrics.uncovered_lines == 0

    assert metrics.covered_chars == len(
        source.body_text
    )

    assert metrics.uncovered_chars == 0
    assert metrics.overlapping_chars == 0

    assert (
        metrics.reconstruction_matches_source
        is True
    )

    assert metrics.marker_count == 1
    assert metrics.raw_line_count == 0
    assert metrics.passes_lossless_gate is True


def test_metrics_detect_character_gap() -> None:
    source = make_source()

    records = make_complete_records(
        source
    )

    document = build_parsed_document(
        source=source,
        profile="mixed_prose_ocr",
        block_records=[records[0]],
    )

    metrics = compute_parse_metrics(
        source=source,
        document=document,
    )

    assert metrics.uncovered_chars == len(
        source.lines[1].text
    )

    assert metrics.passes_lossless_gate is False


def test_metrics_detect_overlap() -> None:
    source = make_source()

    records = make_complete_records(
        source
    )

    duplicate = dict(records[1])
    duplicate["block_id"] = "b0003"

    document = build_parsed_document(
        source=source,
        profile="mixed_prose_ocr",
        block_records=[
            *records,
            duplicate,
        ],
    )

    metrics = compute_parse_metrics(
        source=source,
        document=document,
    )

    assert metrics.overlapping_chars == len(
        source.lines[1].text
    )

    assert metrics.passes_lossless_gate is False


def test_document_rejects_raw_text_length_mismatch() -> None:
    """A block cannot claim text longer than its source span."""

    source = make_source()

    records = make_complete_records(
        source
    )

    records[1]["raw_text"] += "X"

    with pytest.raises(
        ValueError,
        match="length does not match its span",
    ):
        build_parsed_document(
            source=source,
            profile="mixed_prose_ocr",
            block_records=records,
        )

def test_document_rejects_raw_text_mismatch() -> None:
    source = make_source()

    records = make_complete_records(
        source
    )

    records[1]["raw_text"] = (
        "# متن جعلی\r\n"
    )

    with pytest.raises(
        ValueError,
        match="raw_text does not match",
    ):
        build_parsed_document(
            source=source,
            profile="mixed_prose_ocr",
            block_records=records,
        )


def test_document_rejects_unknown_profile() -> None:
    source = make_source()

    with pytest.raises(
        ValueError,
        match="Unsupported parser profile",
    ):
        build_parsed_document(
            source=source,
            profile="unknown_profile",
            block_records=[],
        )