"""Tests for the structured-poetry production handler."""

from __future__ import annotations

from pathlib import Path

from najm_retrieval.parsing.core import (
    split_openiti_source,
)
from najm_retrieval.parsing.handlers.poetry import (
    build_structured_poetry_records,
    parse_structured_poetry,
)
from najm_retrieval.parsing.models import (
    BlockType,
)


VERSION_ID = (
    "0001Author.Work.Poetry-per1"
)

def make_source():
    """Create a small structured-poetry OpenITI source."""

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Poetry Test\r\n"
        "#META#Header#End#\r\n"
        "~~PageV1P001\r\n"
        "### | [ daftar 4 ]\r\n"
        "# 1 بیت نخست ms17 ادامه متن "
        "%~% بیت دوم PageV1P002\r\n"
        "\r\n"
    )

    return split_openiti_source(
        source_text,
        source_path=Path(VERSION_ID),
    )


def test_handler_records_reconstruct_body() -> None:
    source = make_source()

    records = (
        build_structured_poetry_records(
            source
        )
    )

    reconstructed = "".join(
        record["raw_text"]
        for record in records
    )

    assert reconstructed == source.body_text


def test_handler_detects_structural_types() -> None:
    source = make_source()

    records = (
        build_structured_poetry_records(
            source
        )
    )

    block_types = [
        record["block_type"]
        for record in records
    ]

    assert block_types.count(
        "page_marker"
    ) == 2

    assert block_types.count(
        "milestone"
    ) == 1

    assert block_types.count(
        "section"
    ) == 1

    assert block_types.count(
        "blank"
    ) == 1

    assert "verse" in block_types


def test_parse_structured_poetry_is_lossless() -> None:
    source = make_source()

    document, metrics = (
        parse_structured_poetry(
            source
        )
    )

    assert document.version_id == VERSION_ID

    assert document.profile == (
        "structured_poetry"
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


def test_inline_fragments_share_verse_group() -> None:
    source = make_source()

    document, _ = (
        parse_structured_poetry(
            source
        )
    )

    verse_blocks = [
        block
        for block in document.blocks
        if block.block_type
        == BlockType.VERSE
    ]

    group_ids = {
        block.get_attribute(
            "group_id"
        )
        for block in verse_blocks
    }

    assert len(verse_blocks) >= 2
    assert len(group_ids) == 1
    assert None not in group_ids