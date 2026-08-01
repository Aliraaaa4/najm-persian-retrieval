"""Tests for OpenITI structural-text cleanup."""

from __future__ import annotations

import pytest

from najm_retrieval.parsing.models import (
    BlockType,
    SourceSpan,
)
from najm_retrieval.text_preparation.models import (
    LogicalUnit,
)
from najm_retrieval.text_preparation.structural_cleanup import (
    clean_logical_unit_text,
    clean_logical_units,
)


def make_unit(
    *,
    group_id: str,
    block_type: BlockType,
    raw_text: str,
    attributes: tuple[
        tuple[str, object],
        ...,
    ] = (),
) -> LogicalUnit:
    """Create one representative logical unit."""

    return LogicalUnit(
        unit_id=f"version-1:{group_id}",
        version_id="version-1",
        group_id=group_id,
        unit_type=block_type,
        source_block_ids=(
            f"version-1:{group_id}:b1",
        ),
        source_spans=(
            SourceSpan(
                line_start=1,
                line_end=max(
                    1,
                    len(
                        raw_text.splitlines()
                    ),
                ),
                char_start=0,
                char_end=max(
                    1,
                    len(raw_text),
                ),
            ),
        ),
        raw_parts=(
            raw_text,
        ),
        attributes=attributes,
    )


def test_cleans_numbered_verse() -> None:
    raw_text = (
        "# 12 مصرع اول %~% مصرع دوم\r\n"
        "~~ادامه مصرع دوم\r\n"
    )

    unit = make_unit(
        group_id="verse_0001",
        block_type=BlockType.VERSE,
        raw_text=raw_text,
        attributes=(
            ("verse_number", 12),
            (
                "has_hemistich_separator",
                True,
            ),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.raw_text == raw_text

    assert result.display_text == (
        "مصرع اول | مصرع دوم ادامه مصرع دوم"
    )

    assert result.retrieval_text == (
        "مصرع اول مصرع دوم ادامه مصرع دوم"
    )

    assert result.metadata == (
        ("verse_number", 12),
    )

    assert result.issues == ()


def test_cleans_prose_source_wrapping() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        raw_text=(
            "# آغاز پاراگراف\r\n"
            "~~ادامه پاراگراف\r\n"
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "آغاز پاراگراف ادامه پاراگراف"
    )

    assert result.retrieval_text == (
        "آغاز پاراگراف ادامه پاراگراف"
    )


def test_keeps_paragraph_leading_number() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        raw_text="# 15 متن پاراگراف\n",
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "15 متن پاراگراف"
    )


def test_cleans_heading_title() -> None:
    unit = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        raw_text="### | عنوان فصل\n",
        attributes=(
            ("level", 1),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "عنوان فصل"
    )

    assert result.metadata == ()


def test_extracts_metadata_only_heading() -> None:
    unit = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        raw_text=(
            "### || [genre: Gh]\n"
        ),
        attributes=(
            ("level", 1),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == ""
    assert result.retrieval_text == ""
    assert result.is_empty is True

    assert result.metadata == (
        ("genre", "Gh"),
    )


def test_keeps_title_and_extracts_metadata() -> None:
    unit = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        raw_text=(
            "### | غزل‌ها [genre: Gh]\n"
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "غزل‌ها"
    )

    assert result.metadata == (
        ("genre", "Gh"),
    )


def test_does_not_remove_non_metadata_brackets() -> None:
    unit = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        raw_text=(
            "### | عنوان [نسخه دوم]\n"
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "عنوان [نسخه دوم]"
    )

    assert result.metadata == ()


def test_cleans_section_pipe_and_continuation() -> None:
    unit = make_unit(
        group_id="section_0001",
        block_type=BlockType.SECTION,
        raw_text=(
            "### | ~~دفتر اول\n"
        ),
        attributes=(
            ("section_type", "daftar"),
            ("number", 1),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "دفتر اول"
    )


def test_preserves_hash_inside_text() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        raw_text=(
            "# نشانه # داخل متن باقی می‌ماند\n"
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "نشانه # داخل متن باقی می‌ماند"
    )


def test_reports_verse_number_mismatch() -> None:
    unit = make_unit(
        group_id="verse_0001",
        block_type=BlockType.VERSE,
        raw_text="# 9 متن بیت %~% ادامه\n",
        attributes=(
            ("verse_number", 8),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "متن بیت | ادامه"
    )

    assert [
        issue.code
        for issue in result.issues
    ] == [
        "verse_number_mismatch",
    ]


def test_reports_missing_verse_number_prefix() -> None:
    unit = make_unit(
        group_id="verse_0001",
        block_type=BlockType.VERSE,
        raw_text="# متن بیت %~% ادامه\n",
        attributes=(
            ("verse_number", 1),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert [
        issue.code
        for issue in result.issues
    ] == [
        "missing_verse_number_prefix",
    ]


def test_batch_cleanup_preserves_order() -> None:
    first = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        raw_text="# متن اول\n",
    )

    second = make_unit(
        group_id="paragraph_0002",
        block_type=BlockType.PARAGRAPH,
        raw_text="# متن دوم\n",
    )

    results = clean_logical_units(
        [
            first,
            second,
        ]
    )

    assert [
        result.unit_id
        for result in results
    ] == [
        first.unit_id,
        second.unit_id,
    ]


def test_batch_cleanup_rejects_duplicate_ids() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        raw_text="# متن\n",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate logical unit ID",
    ):
        clean_logical_units(
            [
                unit,
                unit,
            ]
        )

def test_removes_repeated_hash_prefixes() -> None:
    unit = make_unit(
        group_id="verse_0001",
        block_type=BlockType.VERSE,
        raw_text=(
            "# # متن بیت %~% ادامه بیت\n"
        ),
        attributes=(
            (
                "has_hemistich_separator",
                True,
            ),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "متن بیت | ادامه بیت"
    )

    assert result.retrieval_text == (
        "متن بیت ادامه بیت"
    )


def test_removes_repeated_continuation_prefixes() -> None:
    unit = make_unit(
        group_id="raw_0001",
        block_type=BlockType.RAW,
        raw_text="~~~~متن OCR\n",
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == "متن OCR"
    assert result.retrieval_text == "متن OCR"


def test_records_structural_only_unit_issue() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        raw_text="# \r\n",
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.is_empty is True

    assert [
        issue.code
        for issue in result.issues
    ] == [
        "structural_only_unit",
    ]


def test_metadata_only_heading_is_not_structural_only() -> None:
    unit = make_unit(
        group_id="heading_0001",
        block_type=BlockType.HEADING,
        raw_text=(
            "### || [genre: Gh]\n"
        ),
        attributes=(
            ("level", 1),
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.is_empty is True

    assert result.metadata == (
        ("genre", "Gh"),
    )

    assert result.issues == ()


def test_preserves_literal_pipe_inside_paragraph() -> None:
    unit = make_unit(
        group_id="paragraph_0001",
        block_type=BlockType.PARAGRAPH,
        raw_text=(
            "# متن سمت اول | متن سمت دوم\n"
        ),
    )

    result = clean_logical_unit_text(
        unit
    )

    assert result.display_text == (
        "متن سمت اول | متن سمت دوم"
    )

    assert result.retrieval_text == (
        "متن سمت اول | متن سمت دوم"
    )

