"""Tests for mixed-prose golden drafting."""

from __future__ import annotations

from collections import Counter

import pytest

from najm_retrieval.parsing.mixed_prose_drafting import (
    apply_mixed_prose_draft,
    draft_mixed_prose_blocks,
)


def make_mixed_prose_sample() -> dict:
    """Create a small lossless mixed-prose fixture."""

    line_texts = [
        "PageV01P190\r\n",
        "\r\n",
        (
            "![image filename]"
            "(./page_249.png)\r\n"
        ),
        "\r\n",
        "~~orphan continuation\r\n",
        "# paragraph ms158 text\r\n",
        "~~continued ms159 text\r\n",
        "### | heading\r\n",
        "# embedded verse %~% second hemistich\r\n",
        "# ~~مجلس دوم\r\n",
    ]

    line_records = []
    cursor = 1000

    for offset, text in enumerate(
        line_texts
    ):
        line_records.append(
            {
                "line_number": 500 + offset,
                "char_start": cursor,
                "char_end": cursor + len(text),
                "text": text,
            }
        )

        cursor += len(text)

    return {
        "schema_version": 1,
        "sample_id": "mixed_prose_test",
        "profile": "mixed_prose_ocr",
        "split": "development",
        "line_start": 500,
        "line_end": 500 + len(line_texts) - 1,
        "char_start": 1000,
        "char_end": cursor,
        "source_path": "example.txt",
        "source_sha256": "test",
        "lines": line_records,
        "source_text": "".join(line_texts),
        "annotations": {
            "schema_version": 1,
            "status": "draft",
            "blocks": [],
            "notes": "",
        },
    }


def test_mixed_prose_draft_is_lossless() -> None:
    sample = make_mixed_prose_sample()

    blocks = draft_mixed_prose_blocks(
        sample
    )

    reconstructed = "".join(
        block["raw_text"]
        for block in blocks
    )

    assert reconstructed == sample["source_text"]

    assert blocks[0]["char_start"] == (
        sample["char_start"]
    )

    assert blocks[-1]["char_end"] == (
        sample["char_end"]
    )


def test_mixed_prose_draft_detects_types() -> None:
    sample = make_mixed_prose_sample()

    blocks = draft_mixed_prose_blocks(
        sample
    )

    counts = Counter(
        block["block_type"]
        for block in blocks
    )

    assert counts["page_marker"] == 1
    assert counts["blank"] == 2
    assert counts["image_reference"] == 1
    assert counts["milestone"] == 2
    assert counts["heading"] == 1
    assert counts["verse"] == 1
    assert counts["section"] == 1
    assert counts["paragraph"] >= 4


def test_orphan_continuation_is_marked() -> None:
    sample = make_mixed_prose_sample()

    blocks = draft_mixed_prose_blocks(
        sample
    )

    orphan_blocks = [
        block
        for block in blocks
        if (
            block["block_type"]
            == "paragraph"
            and block["line_start"] == 504
        )
    ]

    assert len(orphan_blocks) == 1

    orphan = orphan_blocks[0]

    assert orphan["attributes"] == {
        "continuation": True,
        "orphan_continuation": True,
    }

    assert orphan["group_id"] == (
        "paragraph_0001"
    )


def test_inline_milestone_preserves_group() -> None:
    sample = make_mixed_prose_sample()

    blocks = draft_mixed_prose_blocks(
        sample
    )

    milestone_index = next(
        index
        for index, block in enumerate(blocks)
        if (
            block["block_type"]
            == "milestone"
            and block["attributes"]["number"]
            == 158
        )
    )

    previous_block = blocks[
        milestone_index - 1
    ]
    next_block = blocks[
        milestone_index + 1
    ]

    assert previous_block["block_type"] == (
        "paragraph"
    )
    assert next_block["block_type"] == (
        "paragraph"
    )

    assert (
        previous_block["group_id"]
        == next_block["group_id"]
    )

    assert (
        previous_block["attributes"][
            "continuation"
        ]
        is False
    )


def test_image_reference_attributes() -> None:
    sample = make_mixed_prose_sample()

    blocks = draft_mixed_prose_blocks(
        sample
    )

    image_blocks = [
        block
        for block in blocks
        if (
            block["block_type"]
            == "image_reference"
        )
    ]

    assert len(image_blocks) == 1

    assert image_blocks[0]["attributes"] == {
        "target": "./page_249.png",
        "alt_text": "image filename",
    }


def test_embedded_verse_is_detected() -> None:
    sample = make_mixed_prose_sample()

    blocks = draft_mixed_prose_blocks(
        sample
    )

    verse_blocks = [
        block
        for block in blocks
        if block["block_type"] == "verse"
    ]

    assert len(verse_blocks) == 1

    verse = verse_blocks[0]

    assert verse["line_start"] == 508
    assert verse["group_id"] == "verse_0001"

    assert verse["attributes"] == {
        "has_hemistich_separator": True,
        "continuation": False,
        "embedded_in_prose": True,
    }


def test_majlis_section_is_detected() -> None:
    sample = make_mixed_prose_sample()

    blocks = draft_mixed_prose_blocks(
        sample
    )

    sections = [
        block
        for block in blocks
        if block["block_type"] == "section"
    ]

    assert len(sections) == 1

    section = sections[0]

    assert section["line_start"] == 509
    assert section["group_id"] == "section_0001"

    assert section["attributes"] == {
        "section_type": "majlis",
        "title": "مجلس دوم",
    }


def test_apply_draft_remains_draft() -> None:
    sample = make_mixed_prose_sample()

    updated = apply_mixed_prose_draft(
        sample
    )

    assert updated["annotations"]["status"] == (
        "draft"
    )

    assert updated["annotations"]["blocks"]

    assert sample["annotations"]["blocks"] == []


def test_apply_refuses_existing_blocks() -> None:
    sample = make_mixed_prose_sample()

    first = apply_mixed_prose_draft(
        sample
    )

    with pytest.raises(
        ValueError,
        match="already contains blocks",
    ):
        apply_mixed_prose_draft(
            first
        )


def test_wrong_profile_is_rejected() -> None:
    sample = make_mixed_prose_sample()
    sample["profile"] = "structured_poetry"

    with pytest.raises(
        ValueError,
        match="profile='mixed_prose_ocr'",
    ):
        draft_mixed_prose_blocks(
            sample
        )