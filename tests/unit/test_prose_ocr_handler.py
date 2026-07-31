"""Tests for the mixed-prose OCR production handler."""

from __future__ import annotations

from pathlib import Path

from najm_retrieval.parsing.core import (
    split_openiti_source,
)
from najm_retrieval.parsing.handlers.prose_ocr import (
    build_mixed_prose_records,
    parse_mixed_prose,
)
from najm_retrieval.parsing.models import (
    BlockType,
)


VERSION_ID = (
    "0001Author.Work.MixedProse-per1"
)


def make_source():
    """Create a small mixed-prose OpenITI source."""

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Mixed Prose Test\r\n"
        "#META#Header#End#\r\n"
        "PageV01P190\r\n"
        "\r\n"
        "![image filename](./page_249.png)\r\n"
        "# paragraph ms158 text\r\n"
        "~~continued ms159 text\r\n"
        "### | heading\r\n"
        "# embedded verse %~% second hemistich\r\n"
        "# ~~مجلس دوم\r\n"
    )

    return split_openiti_source(
        source_text,
        source_path=Path(VERSION_ID),
    )


def test_mixed_prose_records_reconstruct_body() -> None:
    source = make_source()

    records = build_mixed_prose_records(
        source
    )

    reconstructed = "".join(
        record["raw_text"]
        for record in records
    )

    assert reconstructed == source.body_text


def test_mixed_prose_handler_detects_types() -> None:
    source = make_source()

    records = build_mixed_prose_records(
        source
    )

    block_types = [
        record["block_type"]
        for record in records
    ]

    assert block_types.count(
        "page_marker"
    ) == 1

    assert block_types.count(
        "blank"
    ) == 1

    assert block_types.count(
        "image_reference"
    ) == 1

    assert block_types.count(
        "milestone"
    ) == 2

    assert block_types.count(
        "heading"
    ) == 1

    assert block_types.count(
        "verse"
    ) == 1

    assert block_types.count(
        "section"
    ) == 1

    assert block_types.count(
        "paragraph"
    ) >= 4


def test_parse_mixed_prose_is_lossless() -> None:
    source = make_source()

    document, metrics = parse_mixed_prose(
        source
    )

    assert document.version_id == VERSION_ID

    assert document.profile == (
        "mixed_prose_ocr"
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


def test_inline_paragraph_fragments_keep_group() -> None:
    source = make_source()

    document, _ = parse_mixed_prose(
        source
    )

    paragraph_blocks = [
        block
        for block in document.blocks
        if block.block_type
        == BlockType.PARAGRAPH
    ]

    group_ids = {
        block.get_attribute(
            "group_id"
        )
        for block in paragraph_blocks
    }

    assert len(paragraph_blocks) >= 4
    assert len(group_ids) == 1
    assert None not in group_ids


def test_image_reference_is_preserved() -> None:
    source = make_source()

    document, _ = parse_mixed_prose(
        source
    )

    image_blocks = [
        block
        for block in document.blocks
        if block.block_type
        == BlockType.IMAGE_REFERENCE
    ]

    assert len(image_blocks) == 1

    image_block = image_blocks[0]

    assert image_block.image is not None

    assert image_block.image.image_id == (
        "page_249.png"
    )

    assert image_block.display_text == ""
    assert image_block.retrieval_text == ""
