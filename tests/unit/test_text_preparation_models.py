"""Tests for logical text preparation models."""

from __future__ import annotations

import pytest

from najm_retrieval.parsing.models import (
    BlockType,
    SourceSpan,
)
from najm_retrieval.text_preparation.models import (
    AssemblyIssue,
    LogicalUnit,
)


def make_unit() -> LogicalUnit:
    """Build a representative multi-block logical unit."""

    return LogicalUnit(
        unit_id="version-1:verse_0001",
        version_id="version-1",
        group_id="verse_0001",
        unit_type=BlockType.VERSE,
        source_block_ids=(
            "version-1:b0001",
            "version-1:b0003",
        ),
        source_spans=(
            SourceSpan(
                line_start=10,
                line_end=10,
                char_start=100,
                char_end=120,
            ),
            SourceSpan(
                line_start=12,
                line_end=12,
                char_start=130,
                char_end=150,
            ),
        ),
        raw_parts=(
            "# 1 بخش نخست\r\n",
            "~~بخش دوم\r\n",
        ),
        attributes=(
            ("verse_number", 1),
        ),
    )


def test_builds_valid_logical_unit() -> None:
    unit = make_unit()

    assert unit.unit_type == BlockType.VERSE
    assert unit.component_count == 2
    assert unit.source_block_ids == (
        "version-1:b0001",
        "version-1:b0003",
    )


def test_raw_text_joins_source_parts_in_order() -> None:
    unit = make_unit()

    assert unit.raw_text == (
        "# 1 بخش نخست\r\n"
        "~~بخش دوم\r\n"
    )


def test_envelope_span_contains_all_components() -> None:
    unit = make_unit()

    assert unit.envelope_span == SourceSpan(
        line_start=10,
        line_end=12,
        char_start=100,
        char_end=150,
    )


def test_reports_source_gaps() -> None:
    assert make_unit().has_source_gaps is True


def test_reports_contiguous_components() -> None:
    unit = LogicalUnit(
        unit_id="version-1:paragraph_0001",
        version_id="version-1",
        group_id="paragraph_0001",
        unit_type=BlockType.PARAGRAPH,
        source_block_ids=(
            "version-1:b0001",
            "version-1:b0002",
        ),
        source_spans=(
            SourceSpan(
                line_start=1,
                line_end=1,
                char_start=0,
                char_end=10,
            ),
            SourceSpan(
                line_start=2,
                line_end=2,
                char_start=10,
                char_end=20,
            ),
        ),
        raw_parts=(
            "first",
            "second",
        ),
    )

    assert unit.has_source_gaps is False


def test_rejects_non_content_block_type() -> None:
    with pytest.raises(
        ValueError,
        match="content block types",
    ):
        LogicalUnit(
            unit_id="version-1:page_1",
            version_id="version-1",
            group_id="page_1",
            unit_type=BlockType.PAGE_MARKER,
            source_block_ids=(
                "version-1:b0001",
            ),
            source_spans=(
                SourceSpan(
                    line_start=1,
                    line_end=1,
                    char_start=0,
                    char_end=10,
                ),
            ),
            raw_parts=(
                "PageV01P001",
            ),
        )


def test_rejects_mismatched_component_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="source_spans must have",
    ):
        LogicalUnit(
            unit_id="version-1:verse_0001",
            version_id="version-1",
            group_id="verse_0001",
            unit_type=BlockType.VERSE,
            source_block_ids=(
                "version-1:b0001",
                "version-1:b0002",
            ),
            source_spans=(
                SourceSpan(
                    line_start=1,
                    line_end=1,
                    char_start=0,
                    char_end=10,
                ),
            ),
            raw_parts=(
                "first",
                "second",
            ),
        )


def test_rejects_overlapping_source_spans() -> None:
    with pytest.raises(
        ValueError,
        match="ordered and non-overlapping",
    ):
        LogicalUnit(
            unit_id="version-1:verse_0001",
            version_id="version-1",
            group_id="verse_0001",
            unit_type=BlockType.VERSE,
            source_block_ids=(
                "version-1:b0001",
                "version-1:b0002",
            ),
            source_spans=(
                SourceSpan(
                    line_start=1,
                    line_end=1,
                    char_start=0,
                    char_end=12,
                ),
                SourceSpan(
                    line_start=2,
                    line_end=2,
                    char_start=10,
                    char_end=20,
                ),
            ),
            raw_parts=(
                "first",
                "second",
            ),
        )


def test_rejects_duplicate_attribute_keys() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate logical unit attribute",
    ):
        LogicalUnit(
            unit_id="version-1:verse_0001",
            version_id="version-1",
            group_id="verse_0001",
            unit_type=BlockType.VERSE,
            source_block_ids=(
                "version-1:b0001",
            ),
            source_spans=(
                SourceSpan(
                    line_start=1,
                    line_end=1,
                    char_start=0,
                    char_end=10,
                ),
            ),
            raw_parts=(
                "text",
            ),
            attributes=(
                ("verse_number", 1),
                ("verse_number", 2),
            ),
        )


def test_builds_valid_assembly_issue() -> None:
    issue = AssemblyIssue(
        code="source_gap",
        message="A marker occurs between source blocks.",
        source_block_ids=(
            "version-1:b0001",
            "version-1:b0003",
        ),
    )

    assert issue.code == "source_gap"
    assert len(issue.source_block_ids) == 2


def test_rejects_empty_assembly_issue_code() -> None:
    with pytest.raises(
        ValueError,
        match="code must be",
    ):
        AssemblyIssue(
            code="",
            message="Invalid issue.",
        )
