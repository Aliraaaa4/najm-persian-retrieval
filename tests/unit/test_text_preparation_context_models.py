"""Tests for logical-unit context models."""

from __future__ import annotations

import pytest

from najm_retrieval.parsing.models import (
    BlockType,
    ImageReference,
    PageReference,
    SourceSpan,
)
from najm_retrieval.text_preparation.context_models import (
    ContextMarker,
    ContextualLogicalUnit,
    LogicalUnitContext,
)
from najm_retrieval.text_preparation.models import (
    LogicalUnit,
)


def make_page_marker(
    *,
    block_id: str = "version-1:b0002",
    char_start: int = 10,
) -> ContextMarker:
    """Build a representative page marker."""

    raw_text = "PageV01P001\n"

    return ContextMarker(
        block_id=block_id,
        marker_type=BlockType.PAGE_MARKER,
        span=SourceSpan(
            line_start=2,
            line_end=2,
            char_start=char_start,
            char_end=(
                char_start
                + len(raw_text)
            ),
        ),
        raw_text=raw_text,
        page=PageReference(
            raw_marker="PageV01P001",
            volume=1,
            page=1,
            source_line=2,
        ),
        attributes=(
            ("page", 1),
            ("volume", 1),
        ),
    )


def make_unit() -> LogicalUnit:
    """Build a logical unit spanning a marker position."""

    return LogicalUnit(
        unit_id="version-1:paragraph_0001",
        version_id="version-1",
        group_id="paragraph_0001",
        unit_type=BlockType.PARAGRAPH,
        source_block_ids=(
            "version-1:b0001",
            "version-1:b0003",
        ),
        source_spans=(
            SourceSpan(
                line_start=1,
                line_end=1,
                char_start=0,
                char_end=10,
            ),
            SourceSpan(
                line_start=3,
                line_end=3,
                char_start=24,
                char_end=34,
            ),
        ),
        raw_parts=(
            "first part",
            "secondpart",
        ),
    )


def test_builds_page_context_marker() -> None:
    marker = make_page_marker()

    assert marker.page is not None
    assert marker.page.page == 1


def test_rejects_content_context_marker() -> None:
    with pytest.raises(
        ValueError,
        match="structural block type",
    ):
        ContextMarker(
            block_id="version-1:b0001",
            marker_type=BlockType.PARAGRAPH,
            span=SourceSpan(
                line_start=1,
                line_end=1,
                char_start=0,
                char_end=4,
            ),
            raw_text="text",
        )


def test_page_marker_requires_page_reference() -> None:
    with pytest.raises(
        ValueError,
        match="must include a PageReference",
    ):
        ContextMarker(
            block_id="version-1:b0001",
            marker_type=BlockType.PAGE_MARKER,
            span=SourceSpan(
                line_start=1,
                line_end=1,
                char_start=0,
                char_end=12,
            ),
            raw_text="PageV01P001",
        )


def test_builds_image_context_marker() -> None:
    marker = ContextMarker(
        block_id="version-1:b0002",
        marker_type=BlockType.IMAGE_REFERENCE,
        span=SourceSpan(
            line_start=2,
            line_end=2,
            char_start=10,
            char_end=20,
        ),
        raw_text="image.jpg",
        image=ImageReference(
            raw_marker="image.jpg",
            image_id="image.jpg",
            image_url=None,
            source_line=2,
        ),
    )

    assert marker.image is not None
    assert marker.image.image_id == "image.jpg"


def test_context_detects_embedded_page_boundary() -> None:
    context = LogicalUnitContext(
        unit_id="version-1:paragraph_0001",
        embedded_markers=(
            make_page_marker(),
        ),
    )

    assert context.spans_page_boundary is True
    assert len(
        context.embedded_page_markers
    ) == 1


def test_filters_embedded_milestones() -> None:
    milestone = ContextMarker(
        block_id="version-1:b0002",
        marker_type=BlockType.MILESTONE,
        span=SourceSpan(
            line_start=2,
            line_end=2,
            char_start=10,
            char_end=14,
        ),
        raw_text="ms1\n",
    )

    context = LogicalUnitContext(
        unit_id="version-1:paragraph_0001",
        embedded_markers=(
            milestone,
        ),
    )

    assert context.embedded_milestones == (
        milestone,
    )


def test_rejects_non_page_boundary_marker() -> None:
    milestone = ContextMarker(
        block_id="version-1:b0002",
        marker_type=BlockType.MILESTONE,
        span=SourceSpan(
            line_start=2,
            line_end=2,
            char_start=10,
            char_end=14,
        ),
        raw_text="ms1\n",
    )

    with pytest.raises(
        ValueError,
        match="must reference a page marker",
    ):
        LogicalUnitContext(
            unit_id="version-1:paragraph_0001",
            preceding_page_marker=milestone,
        )


def test_pairs_unit_with_matching_context() -> None:
    unit = make_unit()

    context = LogicalUnitContext(
        unit_id=unit.unit_id,
        embedded_markers=(
            make_page_marker(),
        ),
    )

    result = ContextualLogicalUnit(
        unit=unit,
        context=context,
    )

    assert result.unit is unit
    assert result.context is context


def test_rejects_mismatched_unit_context_ids() -> None:
    with pytest.raises(
        ValueError,
        match="IDs must match",
    ):
        ContextualLogicalUnit(
            unit=make_unit(),
            context=LogicalUnitContext(
                unit_id="version-1:paragraph_9999",
            ),
        )


def test_rejects_embedded_marker_outside_unit() -> None:
    unit = make_unit()

    marker = make_page_marker(
        char_start=100,
    )

    with pytest.raises(
        ValueError,
        match="outside the logical-unit envelope",
    ):
        ContextualLogicalUnit(
            unit=unit,
            context=LogicalUnitContext(
                unit_id=unit.unit_id,
                embedded_markers=(
                    marker,
                ),
            ),
        )
