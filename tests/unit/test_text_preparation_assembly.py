"""Tests for logical-unit assembly."""

from __future__ import annotations

import pytest

from najm_retrieval.parsing.models import (
    BlockType,
)
from najm_retrieval.text_preparation.assembly import (
    assemble_logical_units,
)


def make_block(
    *,
    block_id: str,
    block_type: BlockType,
    raw_text: str,
    char_start: int,
    group_id: str | None = None,
    line_start: int = 1,
    extra_attributes: list[list[object]]
        | None = None,
) -> dict[str, object]:
    """Build one JSON-compatible parsed block."""

    attributes: list[
        list[object]
    ] = []

    if group_id is not None:
        attributes.append(
            [
                "group_id",
                group_id,
            ]
        )

    if extra_attributes:
        attributes.extend(
            extra_attributes
        )

    return {
        "block_id": block_id,
        "block_type": block_type.value,
        "span": {
            "line_start": line_start,
            "line_end": line_start,
            "char_start": char_start,
            "char_end": (
                char_start
                + len(raw_text)
            ),
        },
        "raw_text": raw_text,
        "attributes": attributes,
    }


def test_assembles_single_content_block() -> None:
    blocks = [
        make_block(
            block_id="version-1:b0001",
            block_type=BlockType.VERSE,
            raw_text="# 1 بیت\n",
            char_start=0,
            group_id="verse_0001",
            extra_attributes=[
                [
                    "verse_number",
                    1,
                ],
            ],
        ),
    ]

    units = assemble_logical_units(
        version_id="version-1",
        blocks=blocks,
    )

    assert len(units) == 1

    unit = units[0]

    assert unit.unit_id == (
        "version-1:verse_0001"
    )
    assert unit.raw_text == "# 1 بیت\n"
    assert unit.attributes == (
        (
            "verse_number",
            1,
        ),
    )


def test_assembles_group_separated_by_marker() -> None:
    first = make_block(
        block_id="version-1:b0001",
        block_type=BlockType.VERSE,
        raw_text="# نیمه نخست\n",
        char_start=0,
        group_id="verse_0001",
    )

    marker = make_block(
        block_id="version-1:b0002",
        block_type=BlockType.MILESTONE,
        raw_text="ms1\n",
        char_start=len(
            first["raw_text"]
        ),
    )

    second_start = (
        marker["span"]["char_end"]
    )

    second = make_block(
        block_id="version-1:b0003",
        block_type=BlockType.VERSE,
        raw_text="نیمه دوم\n",
        char_start=second_start,
        group_id="verse_0001",
    )

    units = assemble_logical_units(
        version_id="version-1",
        blocks=[
            first,
            marker,
            second,
        ],
    )

    assert len(units) == 1

    unit = units[0]

    assert unit.source_block_ids == (
        "version-1:b0001",
        "version-1:b0003",
    )
    assert unit.has_source_gaps is True
    assert [
        issue.code
        for issue in unit.issues
    ] == [
        "source_gap",
    ]


def test_skips_non_content_blocks() -> None:
    blocks = [
        make_block(
            block_id="version-1:b0001",
            block_type=BlockType.PAGE_MARKER,
            raw_text="PageV01P001\n",
            char_start=0,
        ),
        make_block(
            block_id="version-1:b0002",
            block_type=BlockType.PARAGRAPH,
            raw_text="# متن\n",
            char_start=12,
            group_id="paragraph_0001",
        ),
    ]

    units = assemble_logical_units(
        version_id="version-1",
        blocks=blocks,
    )

    assert len(units) == 1
    assert units[0].unit_type == (
        BlockType.PARAGRAPH
    )


def test_orders_components_by_source_span() -> None:
    first = make_block(
        block_id="version-1:b0001",
        block_type=BlockType.PARAGRAPH,
        raw_text="first",
        char_start=0,
        group_id="paragraph_0001",
    )

    second = make_block(
        block_id="version-1:b0002",
        block_type=BlockType.PARAGRAPH,
        raw_text="second",
        char_start=5,
        group_id="paragraph_0001",
    )

    units = assemble_logical_units(
        version_id="version-1",
        blocks=[
            second,
            first,
        ],
    )

    unit = units[0]

    assert unit.source_block_ids == (
        "version-1:b0001",
        "version-1:b0002",
    )
    assert unit.raw_text == "firstsecond"
    assert any(
        issue.code
        == "input_order_corrected"
        for issue in unit.issues
    )


def test_orders_units_by_source_position() -> None:
    later = make_block(
        block_id="version-1:b0002",
        block_type=BlockType.PARAGRAPH,
        raw_text="later",
        char_start=10,
        group_id="paragraph_0002",
    )

    earlier = make_block(
        block_id="version-1:b0001",
        block_type=BlockType.PARAGRAPH,
        raw_text="early",
        char_start=0,
        group_id="paragraph_0001",
    )

    units = assemble_logical_units(
        version_id="version-1",
        blocks=[
            later,
            earlier,
        ],
    )

    assert [
        unit.group_id
        for unit in units
    ] == [
        "paragraph_0001",
        "paragraph_0002",
    ]


def test_supports_top_level_group_id() -> None:
    block = make_block(
        block_id="version-1:b0001",
        block_type=BlockType.RAW,
        raw_text="raw text",
        char_start=0,
    )

    block["group_id"] = "raw_0001"

    units = assemble_logical_units(
        version_id="version-1",
        blocks=[
            block,
        ],
    )

    assert units[0].group_id == "raw_0001"


def test_rejects_content_block_without_group_id() -> None:
    block = make_block(
        block_id="version-1:b0001",
        block_type=BlockType.VERSE,
        raw_text="text",
        char_start=0,
    )

    with pytest.raises(
        ValueError,
        match="valid group_id",
    ):
        assemble_logical_units(
            version_id="version-1",
            blocks=[
                block,
            ],
        )


def test_rejects_mixed_block_types_in_group() -> None:
    blocks = [
        make_block(
            block_id="version-1:b0001",
            block_type=BlockType.VERSE,
            raw_text="verse",
            char_start=0,
            group_id="group_0001",
        ),
        make_block(
            block_id="version-1:b0002",
            block_type=BlockType.PARAGRAPH,
            raw_text="paragraph",
            char_start=5,
            group_id="group_0001",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="multiple block types",
    ):
        assemble_logical_units(
            version_id="version-1",
            blocks=blocks,
        )


def test_rejects_duplicate_content_block_ids() -> None:
    blocks = [
        make_block(
            block_id="version-1:b0001",
            block_type=BlockType.VERSE,
            raw_text="first",
            char_start=0,
            group_id="verse_0001",
        ),
        make_block(
            block_id="version-1:b0001",
            block_type=BlockType.VERSE,
            raw_text="second",
            char_start=5,
            group_id="verse_0002",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate content block ID",
    ):
        assemble_logical_units(
            version_id="version-1",
            blocks=blocks,
        )


def test_records_conflicting_semantic_attribute() -> None:
    blocks = [
        make_block(
            block_id="version-1:b0001",
            block_type=BlockType.VERSE,
            raw_text="first",
            char_start=0,
            group_id="verse_0001",
            extra_attributes=[
                [
                    "verse_number",
                    1,
                ],
            ],
        ),
        make_block(
            block_id="version-1:b0002",
            block_type=BlockType.VERSE,
            raw_text="second",
            char_start=5,
            group_id="verse_0001",
            extra_attributes=[
                [
                    "verse_number",
                    2,
                ],
            ],
        ),
    ]

    unit = assemble_logical_units(
        version_id="version-1",
        blocks=blocks,
    )[0]

    assert unit.attributes == ()
    assert any(
        issue.code
        == "attribute_conflict"
        for issue in unit.issues
    )


def test_rejects_raw_text_span_mismatch() -> None:
    block = make_block(
        block_id="version-1:b0001",
        block_type=BlockType.VERSE,
        raw_text="text",
        char_start=0,
        group_id="verse_0001",
    )

    block["span"]["char_end"] = 20

    with pytest.raises(
        ValueError,
        match="length does not match",
    ):
        assemble_logical_units(
            version_id="version-1",
            blocks=[
                block,
            ],
        )
