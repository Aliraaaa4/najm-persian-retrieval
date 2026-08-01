"""Tests for passage models and configuration."""

from __future__ import annotations

import pytest

from najm_retrieval.parsing.models import (
    BlockType,
)
from najm_retrieval.passages import (
    PASSAGE_SCHEMA_VERSION,
    Passage,
    PassageBoundary,
    PassageBuildResult,
    PassageConfig,
    PassageKind,
    PassageMember,
)


def make_member(
    unit_id: str,
    *,
    unit_type: BlockType = BlockType.PARAGRAPH,
    text: str = "متن آزمایشی",
    segment_index: int = 0,
    segment_count: int = 1,
) -> PassageMember:
    """Create a compact valid passage member."""

    return PassageMember(
        unit_id=unit_id,
        unit_type=unit_type,
        display_text=text,
        retrieval_text=text,
        search_alias_text=text,
        segment_index=segment_index,
        segment_count=segment_count,
    )


def make_passage(
    *,
    passage_id: str = "book:passage:001",
    previous_passage_id: str | None = None,
    next_passage_id: str | None = None,
) -> Passage:
    """Create a compact valid passage."""

    return Passage(
        passage_id=passage_id,
        version_id="book",
        profile="mixed_prose_ocr",
        kind=PassageKind.MIXED_PROSE,
        ordinal=0,
        include_in_index=True,
        members=(
            make_member(
                "book:paragraph_0001"
            ),
        ),
        previous_passage_id=(
            previous_passage_id
        ),
        next_passage_id=(
            next_passage_id
        ),
    )


def test_passage_config_defaults() -> None:
    """Defaults match the audited corpus configuration."""

    config = PassageConfig()

    assert config.diwan_target_verses == 8
    assert config.diwan_overlap_verses == 1
    assert config.diwan_minimum_tail_verses == 2

    assert config.mathnawi_target_verses == 8
    assert config.mathnawi_overlap_verses == 1
    assert config.mathnawi_minimum_tail_verses == 6

    assert config.prose_target_words == 180
    assert config.prose_soft_min_words == 80
    assert config.prose_hard_max_words == 300

    assert (
        config.schema_version
        == PASSAGE_SCHEMA_VERSION
    )


def test_passage_config_rejects_invalid_diwan_overlap() -> None:
    """Poetry overlap must remain below target."""

    with pytest.raises(
        ValueError,
        match="diwan overlap",
    ):
        PassageConfig(
            diwan_target_verses=8,
            diwan_overlap_verses=8,
        )


def test_passage_config_rejects_invalid_mathnawi_tail() -> None:
    """Mathnawi tail cannot exceed target size."""

    with pytest.raises(
        ValueError,
        match="mathnawi minimum tail",
    ):
        PassageConfig(
            mathnawi_target_verses=8,
            mathnawi_minimum_tail_verses=9,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "prose_soft_min_words": 181,
            "prose_target_words": 180,
        },
        {
            "prose_target_words": 301,
            "prose_hard_max_words": 300,
        },
        {
            "prose_hard_max_words": 0,
        },
    ],
)
def test_passage_config_rejects_invalid_prose_limits(
    overrides: dict[str, int],
) -> None:
    """Prose limits must be positive and ordered."""

    with pytest.raises(ValueError):
        PassageConfig(
            **overrides
        )


def test_passage_member_reports_fragment_state() -> None:
    """Split source units expose fragment metadata."""

    member = make_member(
        "book:paragraph_0001",
        segment_index=1,
        segment_count=3,
    )

    assert member.is_fragment
    assert member.word_count == 2
    assert member.char_count == len(
        "متن آزمایشی"
    )


def test_passage_member_rejects_empty_retrieval_text() -> None:
    """Empty normalized content cannot enter a passage."""

    with pytest.raises(
        ValueError,
        match="retrieval_text",
    ):
        PassageMember(
            unit_id="book:paragraph_0001",
            unit_type=BlockType.PARAGRAPH,
            display_text="متن",
            retrieval_text="",
            search_alias_text="متن",
        )


def test_boundary_rejects_content_unit_type() -> None:
    """Only headings and sections are passage boundaries."""

    with pytest.raises(
        ValueError,
        match="heading or section",
    ):
        PassageBoundary(
            unit_id="book:verse_0001",
            unit_type=BlockType.VERSE,
        )


def test_poetry_passage_joins_verses_with_line_breaks() -> None:
    """Consecutive verses remain visually distinct."""

    passage = Passage(
        passage_id="book:passage:poetry",
        version_id="book",
        profile="structured_poetry",
        kind=PassageKind.DIWAN,
        ordinal=0,
        include_in_index=True,
        members=(
            make_member(
                "book:verse_0001",
                unit_type=BlockType.VERSE,
                text="بیت اول",
            ),
            make_member(
                "book:verse_0002",
                unit_type=BlockType.VERSE,
                text="بیت دوم",
            ),
        ),
    )

    assert passage.display_text == (
        "بیت اول\nبیت دوم"
    )

    assert passage.retrieval_text == (
        "بیت اول\nبیت دوم"
    )

    assert passage.unit_count == 2
    assert passage.word_count == 4


def test_mixed_passage_joins_paragraphs_with_blank_lines() -> None:
    """Paragraph boundaries remain visible."""

    passage = Passage(
        passage_id="book:passage:prose",
        version_id="book",
        profile="mixed_prose_ocr",
        kind=PassageKind.MIXED_PROSE,
        ordinal=0,
        include_in_index=True,
        members=(
            make_member(
                "book:paragraph_0001",
                text="پاراگراف اول",
            ),
            make_member(
                "book:paragraph_0002",
                text="پاراگراف دوم",
            ),
        ),
    )

    assert passage.display_text == (
        "پاراگراف اول\n\nپاراگراف دوم"
    )


def test_passage_deduplicates_fragment_source_ids() -> None:
    """Multiple fragments retain one logical source ID."""

    passage = Passage(
        passage_id="book:passage:fragments",
        version_id="book",
        profile="mixed_prose_ocr",
        kind=PassageKind.MIXED_PROSE,
        ordinal=0,
        include_in_index=True,
        members=(
            make_member(
                "book:paragraph_0001",
                text="بخش اول",
                segment_index=0,
                segment_count=2,
            ),
            make_member(
                "book:paragraph_0001",
                text="بخش دوم",
                segment_index=1,
                segment_count=2,
            ),
            make_member(
                "book:paragraph_0002",
                text="پاراگراف بعد",
            ),
        ),
    )

    assert passage.source_unit_ids == (
        "book:paragraph_0001",
        "book:paragraph_0002",
    )

    assert passage.unit_count == 2
    assert passage.member_count == 3
    assert passage.has_split_member


@pytest.mark.parametrize(
    "neighbor_field",
    [
        "previous_passage_id",
        "next_passage_id",
    ],
)
def test_passage_rejects_itself_as_neighbor(
    neighbor_field: str,
) -> None:
    """A passage cannot link to itself."""

    kwargs = {
        neighbor_field: "book:passage:001"
    }

    with pytest.raises(
        ValueError,
        match="cannot be its own",
    ):
        make_passage(
            **kwargs
        )


def test_passage_build_result_reports_coverage() -> None:
    """Build results expose passage and unit coverage."""

    passage = make_passage()

    result = PassageBuildResult(
        config=PassageConfig(),
        passages=(
            passage,
        ),
        skipped_unit_ids=(
            "book:paragraph_empty",
        ),
    )

    assert result.passage_count == 1

    assert result.covered_unit_ids == (
        "book:paragraph_0001",
    )

    assert result.indexable_passages == (
        passage,
    )
