"""Tests for shared parser output models."""

from pathlib import Path

import pytest

from najm_retrieval.parsing.models import (
    BlockType,
    ImageReference,
    IssueSeverity,
    PageReference,
    ParseIssue,
    ParseMetrics,
    ParsedBlock,
    ParsedDocument,
    SourceSpan,
)


def test_source_span_counts_lines_and_characters() -> None:
    span = SourceSpan(
        line_start=10,
        line_end=12,
        char_start=100,
        char_end=160,
    )

    assert span.line_count == 3
    assert span.char_count == 60


def test_source_span_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="line_start"):
        SourceSpan(
            line_start=0,
            line_end=1,
            char_start=0,
            char_end=1,
        )

    with pytest.raises(ValueError, match="line_end"):
        SourceSpan(
            line_start=5,
            line_end=4,
            char_start=0,
            char_end=1,
        )

    with pytest.raises(ValueError, match="char_end"):
        SourceSpan(
            line_start=1,
            line_end=1,
            char_start=10,
            char_end=9,
        )


def test_marker_block_is_not_searchable_content() -> None:
    page = PageReference(
        raw_marker="PageV01P012",
        volume=1,
        page=12,
        source_line=4,
    )

    block = ParsedBlock(
        block_id="version:000001",
        block_type=BlockType.PAGE_MARKER,
        span=SourceSpan(
            line_start=4,
            line_end=4,
            char_start=40,
            char_end=53,
        ),
        raw_text="PageV01P012\n",
        display_text="",
        retrieval_text="",
        page=page,
    )

    assert block.is_marker is True
    assert block.is_content is False
    assert block.retrieval_text == ""


def test_image_reference_can_be_attached_to_content() -> None:
    image = ImageReference(
        raw_marker="ImageRef: image_0012",
        image_id="image_0012",
        image_url=None,
        source_line=5,
    )

    block = ParsedBlock(
        block_id="version:000002",
        block_type=BlockType.PARAGRAPH,
        span=SourceSpan(
            line_start=6,
            line_end=6,
            char_start=60,
            char_end=75,
        ),
        raw_text="متن پاراگراف\n",
        display_text="متن پاراگراف",
        retrieval_text="متن پاراگراف",
        image=image,
    )

    assert block.image is image
    assert block.is_content is True


def test_parsed_document_reconstructs_exact_raw_body() -> None:
    first = ParsedBlock(
        block_id="version:000001",
        block_type=BlockType.HEADING,
        span=SourceSpan(
            line_start=4,
            line_end=4,
            char_start=20,
            char_end=28,
        ),
        raw_text="عنوان\n",
        display_text="عنوان",
        retrieval_text="عنوان",
    )

    second = ParsedBlock(
        block_id="version:000002",
        block_type=BlockType.PARAGRAPH,
        span=SourceSpan(
            line_start=5,
            line_end=5,
            char_start=28,
            char_end=37,
        ),
        raw_text="متن اصلی\n",
        display_text="متن اصلی",
        retrieval_text="متن اصلی",
    )

    document = ParsedDocument(
        version_id="0001Author.Work.Version-per1",
        profile="mixed_prose_ocr",
        parser_name="test_parser",
        parser_version="0.1",
        source_path=Path("test-per1"),
        body_char_start=20,
        body_line_start=4,
        blocks=(first, second),
        issues=(),
    )

    assert document.reconstruct_body() == (
        "عنوان\n"
        "متن اصلی\n"
    )

    assert document.block_counts == {
        "heading": 1,
        "paragraph": 1,
    }

    assert document.content_blocks == (
        first,
        second,
    )


def test_document_collects_block_issues() -> None:
    issue = ParseIssue(
        code="unrecognized_line",
        message="Line could not be classified.",
        severity=IssueSeverity.WARNING,
        span=SourceSpan(
            line_start=4,
            line_end=4,
            char_start=20,
            char_end=25,
        ),
        raw_text="???\n",
    )

    block = ParsedBlock(
        block_id="version:000001",
        block_type=BlockType.RAW,
        span=issue.span,
        raw_text="???\n",
        display_text="???",
        retrieval_text="???",
        issues=(issue,),
    )

    document = ParsedDocument(
        version_id="0001Author.Work.Version-per1",
        profile="raw_ocr_reference",
        parser_name="test_parser",
        parser_version="0.1",
        source_path=Path("test-per1"),
        body_char_start=20,
        body_line_start=4,
        blocks=(block,),
        issues=(),
    )

    assert document.all_issues == (issue,)

def test_parse_metrics_lossless_gate() -> None:
    metrics = ParseMetrics(
        total_body_lines=100,
        covered_lines=100,
        uncovered_lines=0,
        total_body_chars=1000,
        covered_chars=1000,
        uncovered_chars=0,
        overlapping_chars=0,
        reconstructed_chars=1000,
        reconstruction_matches_source=True,
        marker_count=10,
        raw_line_count=2,
        issue_count=1,
        runtime_seconds=0.1,
        peak_memory_bytes=1024,
    )

    assert metrics.line_coverage == 1.0
    assert metrics.char_coverage == 1.0
    assert metrics.reconstruction_ratio == 1.0
    assert metrics.raw_line_rate == 0.02
    assert metrics.passes_lossless_gate is True
    
    
    
    
    
    
def test_parse_metrics_rejects_overlapping_characters() -> None:
    metrics = ParseMetrics(
        total_body_lines=10,
        covered_lines=10,
        uncovered_lines=0,
        total_body_chars=100,
        covered_chars=100,
        uncovered_chars=0,
        overlapping_chars=4,
        reconstructed_chars=104,
        reconstruction_matches_source=False,
        marker_count=1,
        raw_line_count=0,
        issue_count=1,
        runtime_seconds=0.01,
        peak_memory_bytes=100,
    )

    assert metrics.passes_lossless_gate is False    