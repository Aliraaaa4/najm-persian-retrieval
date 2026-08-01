"""Tests for structure-aware passage construction."""

from __future__ import annotations

from types import SimpleNamespace

from najm_retrieval.parsing.models import (
    BlockType,
)
from najm_retrieval.passage_builder import (
    build_passages,
)
from najm_retrieval.passages import (
    PassageConfig,
    PassageKind,
)


def block_type(
    value: str,
) -> BlockType:
    """Resolve a block type by its serialized value."""

    return next(
        item
        for item in BlockType
        if item.value == value
    )


VERSE = block_type("verse")
PARAGRAPH = block_type("paragraph")
RAW = block_type("raw")
HEADING = block_type("heading")
SECTION = block_type("section")


def make_pair(
    unit_id: str,
    *,
    unit_type: BlockType,
    text: str,
    version_id: str = "book",
    heading_path: tuple[str, ...] = (),
    section_path: tuple[str, ...] = (),
    metadata: tuple[
        tuple[str, object],
        ...,
    ] = (),
):
    """Create aligned contextual and cleaned test objects."""

    unit = SimpleNamespace(
        unit_id=unit_id,
        version_id=version_id,
        unit_type=unit_type,
        source_spans=(),
        issues=(),
    )

    context = SimpleNamespace(
        heading_path=heading_path,
        section_path=section_path,
        issues=(),
    )

    contextual = SimpleNamespace(
        unit=unit,
        context=context,
    )

    cleaned = SimpleNamespace(
        unit_id=unit_id,
        display_text=text,
        metadata=metadata,
        issues=(),
    )

    return contextual, cleaned


def build_from_pairs(
    pairs,
    *,
    version_id: str = "book",
    profile: str,
    include_in_index: bool = True,
    config: PassageConfig | None = None,
):
    """Build passages from compact aligned fixtures."""

    contextual_units = tuple(
        pair[0]
        for pair in pairs
    )

    cleaned_units = tuple(
        pair[1]
        for pair in pairs
    )

    return build_passages(
        version_id=version_id,
        profile=profile,
        include_in_index=(
            include_in_index
        ),
        contextual_units=(
            contextual_units
        ),
        cleaned_units=cleaned_units,
        config=config,
    )


def test_diwan_uses_eight_verse_windows_with_one_overlap() -> None:
    """Diwan windows preserve one overlapping verse."""

    heading_id = "book:heading_0001"

    pairs = [
        make_pair(
            heading_id,
            unit_type=HEADING,
            text="",
            metadata=(
                ("genre", "Gh"),
            ),
        )
    ]

    pairs.extend(
        make_pair(
            f"book:verse_{index:04d}",
            unit_type=VERSE,
            text=f"بیت شماره {index}",
            heading_path=(
                heading_id,
            ),
        )
        for index in range(
            1,
            11,
        )
    )

    result = build_from_pairs(
        pairs,
        profile="structured_poetry",
    )

    assert result.passage_count == 2

    first, second = result.passages

    assert first.kind is PassageKind.DIWAN
    assert first.unit_count == 8
    assert second.unit_count == 3

    assert (
        first.source_unit_ids[-1]
        == second.source_unit_ids[0]
    )

    assert first.next_passage_id == (
        second.passage_id
    )

    assert second.previous_passage_id == (
        first.passage_id
    )

    assert first.boundaries[0].metadata_dict == {
        "genre": "Gh"
    }


def test_mathnawi_rebalances_a_short_final_window() -> None:
    """A Mathnawi tail is shifted to meet the audited minimum."""

    version_id = (
        "0672JalalDinRumi."
        "Mathnawi.PDL00048-per1"
    )

    section_id = (
        f"{version_id}:section_0001"
    )

    pairs = [
        make_pair(
            section_id,
            unit_type=SECTION,
            text="دفتر اول",
            version_id=version_id,
        )
    ]

    pairs.extend(
        make_pair(
            f"{version_id}:verse_{index:04d}",
            unit_type=VERSE,
            text=f"بیت {index}",
            version_id=version_id,
            section_path=(
                section_id,
            ),
        )
        for index in range(
            1,
            11,
        )
    )

    result = build_from_pairs(
        pairs,
        version_id=version_id,
        profile="structured_poetry",
    )

    assert result.passage_count == 2

    first, second = result.passages

    assert first.kind is PassageKind.MATHNAWI
    assert first.unit_count == 8
    assert second.unit_count == 8

    assert second.source_unit_ids[0].endswith(
        "verse_0003"
    )


def test_mixed_prose_packs_without_overlap() -> None:
    """Prose members are packed without duplicate source units."""

    pairs = [
        make_pair(
            f"book:paragraph_{index:04d}",
            unit_type=PARAGRAPH,
            text=" ".join(
                ["واژه"] * 100
            ),
        )
        for index in range(
            1,
            4,
        )
    ]

    result = build_from_pairs(
        pairs,
        profile="mixed_prose_ocr",
    )

    assert result.passage_count == 2

    assert result.passages[0].word_count == 200
    assert result.passages[1].word_count == 100

    covered = [
        unit_id
        for passage in result.passages
        for unit_id in passage.source_unit_ids
    ]

    assert len(covered) == len(
        set(covered)
    )


def test_oversized_paragraph_uses_sentence_splitting() -> None:
    """Oversized prose is split at sentence boundaries."""

    sentence = (
        " ".join(
            ["واژه"] * 170
        )
        + "؟"
    )

    text = " ".join(
        [sentence] * 4
    )

    pairs = [
        make_pair(
            "book:paragraph_0001",
            unit_type=PARAGRAPH,
            text=text,
        )
    ]

    result = build_from_pairs(
        pairs,
        profile="mixed_prose_ocr",
    )

    assert result.passage_count == 4

    assert all(
        passage.word_count <= 300
        for passage in result.passages
    )

    assert all(
        passage.has_split_member
        for passage in result.passages
    )

    assert result.covered_unit_ids == (
        "book:paragraph_0001",
    )

    issue_codes = {
        issue.code
        for passage in result.passages
        for issue in passage.issues
    }

    assert (
        "oversized_unit_sentence_split"
        in issue_codes
    )


def test_oversized_paragraph_uses_word_fallback() -> None:
    """A paragraph without punctuation uses fixed-word fallback."""

    text = " ".join(
        ["واژه"] * 311
    )

    pairs = [
        make_pair(
            "book:paragraph_0001",
            unit_type=PARAGRAPH,
            text=text,
        )
    ]

    result = build_from_pairs(
        pairs,
        profile="mixed_prose_ocr",
    )

    assert result.passage_count == 2

    assert all(
        passage.word_count <= 300
        for passage in result.passages
    )

    issue_codes = {
        issue.code
        for passage in result.passages
        for issue in passage.issues
    }

    assert (
        "oversized_unit_fallback_split"
        in issue_codes
    )


def test_empty_content_unit_is_skipped() -> None:
    """Structural-only content is excluded and reported."""

    pairs = [
        make_pair(
            "book:paragraph_0001",
            unit_type=PARAGRAPH,
            text="",
        ),
        make_pair(
            "book:paragraph_0002",
            unit_type=PARAGRAPH,
            text="متن معتبر",
        ),
    ]

    result = build_from_pairs(
        pairs,
        profile="mixed_prose_ocr",
    )

    assert result.passage_count == 1

    assert result.skipped_unit_ids == (
        "book:paragraph_0001",
    )

    assert result.covered_unit_ids == (
        "book:paragraph_0002",
    )


def test_mixed_profile_preserves_embedded_verse_order() -> None:
    """Embedded verses stay in source order with prose."""

    pairs = [
        make_pair(
            "book:paragraph_0001",
            unit_type=PARAGRAPH,
            text="پاراگراف نخست",
        ),
        make_pair(
            "book:verse_0001",
            unit_type=VERSE,
            text="بیت میانی",
        ),
        make_pair(
            "book:paragraph_0002",
            unit_type=PARAGRAPH,
            text="پاراگراف دوم",
        ),
    ]

    result = build_from_pairs(
        pairs,
        profile="mixed_prose_ocr",
    )

    assert result.passage_count == 1

    assert result.passages[0].source_unit_ids == (
        "book:paragraph_0001",
        "book:verse_0001",
        "book:paragraph_0002",
    )


def test_neighbor_links_do_not_cross_heading_boundary() -> None:
    """Previous and next links remain inside one structural run."""

    first_heading = "book:heading_0001"
    second_heading = "book:heading_0002"

    pairs = [
        make_pair(
            first_heading,
            unit_type=HEADING,
            text="عنوان اول",
        ),
        make_pair(
            "book:paragraph_0001",
            unit_type=PARAGRAPH,
            text="متن اول",
            heading_path=(
                first_heading,
            ),
        ),
        make_pair(
            second_heading,
            unit_type=HEADING,
            text="عنوان دوم",
        ),
        make_pair(
            "book:paragraph_0002",
            unit_type=PARAGRAPH,
            text="متن دوم",
            heading_path=(
                second_heading,
            ),
        ),
    ]

    result = build_from_pairs(
        pairs,
        profile="mixed_prose_ocr",
    )

    assert result.passage_count == 2

    first, second = result.passages

    assert first.next_passage_id is None
    assert second.previous_passage_id is None


def test_reference_version_builds_no_index_passages() -> None:
    """Kraken-style reference versions remain outside retrieval."""

    pairs = [
        make_pair(
            "book:raw_0001",
            unit_type=RAW,
            text="متن مرجع",
        )
    ]

    result = build_from_pairs(
        pairs,
        profile="raw_ocr_reference",
        include_in_index=False,
    )

    assert result.passage_count == 0

    assert result.skipped_unit_ids == (
        "book:raw_0001",
    )

    assert {
        issue.code
        for issue in result.issues
    } == {
        "version_excluded_from_index"
    }
