"""Tests for logical-unit context resolution."""

from __future__ import annotations

import pytest

from najm_retrieval.parsing.models import (
    BlockType,
    SourceSpan,
)
from najm_retrieval.text_preparation.context_resolver import (
    resolve_logical_unit_contexts,
)
from najm_retrieval.text_preparation.models import (
    LogicalUnit,
)


def make_unit(
    *,
    group_id: str,
    block_type: BlockType,
    char_start: int,
    char_end: int,
    level: int | None = None,
    second_span: tuple[int, int] | None = None,
) -> LogicalUnit:
    """Create one logical unit."""

    attributes: tuple[
        tuple[str, object],
        ...,
    ] = ()

    if level is not None:
        attributes = (
            ("level", level),
        )

    if second_span is None:
        source_block_ids = (
            f"version-1:{group_id}:b1",
        )

        source_spans = (
            SourceSpan(
                line_start=1,
                line_end=1,
                char_start=char_start,
                char_end=char_end,
            ),
        )

        raw_parts = (
            "text",
        )

    else:
        source_block_ids = (
            f"version-1:{group_id}:b1",
            f"version-1:{group_id}:b2",
        )

        source_spans = (
            SourceSpan(
                line_start=1,
                line_end=1,
                char_start=char_start,
                char_end=char_end,
            ),
            SourceSpan(
                line_start=3,
                line_end=3,
                char_start=second_span[0],
                char_end=second_span[1],
            ),
        )

        raw_parts = (
            "first",
            "second",
        )

    return LogicalUnit(
        unit_id=f"version-1:{group_id}",
        version_id="version-1",
        group_id=group_id,
        unit_type=block_type,
        source_block_ids=source_block_ids,
        source_spans=source_spans,
        raw_parts=raw_parts,
        attributes=attributes,
    )


def make_page_block(
    *,
    block_id: str,
    char_start: int,
    page: int,
) -> dict[str, object]:
    """Create one JSON-compatible page marker."""

    raw_text = f"PageV01P{page:03d}\n"

    return {
        "block_id": block_id,
        "block_type": "page_marker",
        "span": {
            "line_start": 2,
            "line_end": 2,
            "char_start": char_start,
            "char_end": (
                char_start
                + len(raw_text)
            ),
        },
        "raw_text": raw_text,
        "page": {
            "raw_marker": (
                f"PageV01P{page:03d}"
            ),
            "volume": 1,
            "page": page,
            "source_line": 2,
        },
        "attributes": [
            ["page", page],
            ["volume", 1],
        ],
    }


def test_resolves_page_marker_between_units() -> None:
    first = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=0,
        char_end=10,
    )

    marker = make_page_block(
        block_id="version-1:b0002",
        char_start=10,
        page=1,
    )

    marker_end = marker[
        "span"
    ][
        "char_end"
    ]

    second = make_unit(
        group_id="paragraph_0002",
        block_type=BlockType.PARAGRAPH,
        char_start=marker_end,
        char_end=marker_end + 10,
    )

    results = resolve_logical_unit_contexts(
        units=[
            first,
            second,
        ],
        blocks=[
            marker,
        ],
    )

    assert (
        results[0]
        .context
        .following_page_marker
        is not None
    )

    assert (
        results[1]
        .context
        .preceding_page_marker
        is not None
    )

    assert (
        results[1]
        .context
        .preceding_page_marker
        .page
        .page
        == 1
    )


def test_resolves_page_marker_inside_unit() -> None:
    marker = make_page_block(
        block_id="version-1:b0002",
        char_start=10,
        page=4,
    )

    marker_end = marker[
        "span"
    ][
        "char_end"
    ]

    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=0,
        char_end=10,
        second_span=(
            marker_end,
            marker_end + 10,
        ),
    )

    result = resolve_logical_unit_contexts(
        units=[
            unit,
        ],
        blocks=[
            marker,
        ],
    )[0]

    assert (
        result.context.spans_page_boundary
        is True
    )

    assert len(
        result.context.embedded_page_markers
    ) == 1


def test_builds_nested_heading_path() -> None:
    heading_one = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        char_start=0,
        char_end=10,
        level=1,
    )

    paragraph_one = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=10,
        char_end=20,
    )

    heading_two = make_unit(
        group_id="heading_0002",
        block_type=BlockType.HEADING,
        char_start=20,
        char_end=30,
        level=2,
    )

    paragraph_two = make_unit(
        group_id="paragraph_0002",
        block_type=BlockType.PARAGRAPH,
        char_start=30,
        char_end=40,
    )

    results = resolve_logical_unit_contexts(
        units=[
            paragraph_two,
            heading_two,
            paragraph_one,
            heading_one,
        ],
        blocks=[],
    )

    by_id = {
        item.unit.unit_id: item
        for item in results
    }

    assert (
        by_id[paragraph_one.unit_id]
        .context
        .heading_path
        == (
            heading_one.unit_id,
        )
    )

    assert (
        by_id[heading_two.unit_id]
        .context
        .heading_path
        == (
            heading_one.unit_id,
        )
    )

    assert (
        by_id[paragraph_two.unit_id]
        .context
        .heading_path
        == (
            heading_one.unit_id,
            heading_two.unit_id,
        )
    )


def test_same_level_heading_replaces_previous_heading() -> None:
    heading_one = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        char_start=0,
        char_end=10,
        level=1,
    )

    heading_two = make_unit(
        group_id="heading_0002",
        block_type=BlockType.HEADING,
        char_start=10,
        char_end=20,
        level=1,
    )

    paragraph = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=20,
        char_end=30,
    )

    results = resolve_logical_unit_contexts(
        units=[
            heading_one,
            heading_two,
            paragraph,
        ],
        blocks=[],
    )

    assert (
        results[2]
        .context
        .heading_path
        == (
            heading_two.unit_id,
        )
    )


def test_section_creates_scope_and_resets_headings() -> None:
    section = make_unit(
        group_id="section_0001",
        block_type=BlockType.SECTION,
        char_start=0,
        char_end=10,
    )

    heading = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        char_start=10,
        char_end=20,
        level=1,
    )

    paragraph = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=20,
        char_end=30,
    )

    new_section = make_unit(
        group_id="section_0002",
        block_type=BlockType.SECTION,
        char_start=30,
        char_end=40,
    )

    final_paragraph = make_unit(
        group_id="paragraph_0002",
        block_type=BlockType.PARAGRAPH,
        char_start=40,
        char_end=50,
    )

    results = resolve_logical_unit_contexts(
        units=[
            section,
            heading,
            paragraph,
            new_section,
            final_paragraph,
        ],
        blocks=[],
    )

    assert (
        results[2]
        .context
        .section_path
        == (
            section.unit_id,
        )
    )

    assert (
        results[2]
        .context
        .heading_path
        == (
            heading.unit_id,
        )
    )

    assert (
        results[4]
        .context
        .section_path
        == (
            new_section.unit_id,
        )
    )

    assert (
        results[4]
        .context
        .heading_path
        == ()
    )


def test_invalid_heading_level_records_issue() -> None:
    heading = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        char_start=0,
        char_end=10,
    )

    result = resolve_logical_unit_contexts(
        units=[
            heading,
        ],
        blocks=[],
    )[0]

    assert [
        issue.code
        for issue in result.context.issues
    ] == [
        "invalid_heading_level",
    ]


def test_skips_non_structural_blocks() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=0,
        char_end=10,
    )

    content_block = {
        "block_id": "version-1:b0001",
        "block_type": "paragraph",
        "span": {
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 10,
        },
        "raw_text": "paragraph",
        "attributes": [],
    }

    result = resolve_logical_unit_contexts(
        units=[
            unit,
        ],
        blocks=[
            content_block,
        ],
    )[0]

    assert result.context.embedded_markers == ()


def test_rejects_duplicate_structural_block_ids() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=30,
        char_end=40,
    )

    first = make_page_block(
        block_id="version-1:b0001",
        char_start=0,
        page=1,
    )

    second = make_page_block(
        block_id="version-1:b0001",
        char_start=15,
        page=2,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate structural block ID",
    ):
        resolve_logical_unit_contexts(
            units=[
                unit,
            ],
            blocks=[
                first,
                second,
            ],
        )


def test_rejects_overlapping_unit_envelopes() -> None:
    first = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        char_start=0,
        char_end=20,
    )

    second = make_unit(
        group_id="paragraph_0002",
        block_type=BlockType.PARAGRAPH,
        char_start=10,
        char_end=30,
    )

    with pytest.raises(
        ValueError,
        match="non-overlapping",
    ):
        resolve_logical_unit_contexts(
            units=[
                first,
                second,
            ],
            blocks=[],
        )
