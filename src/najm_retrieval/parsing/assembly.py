"""Build typed parser outputs and calculate lossless metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from najm_retrieval.parsing.core import (
    OpenITISource,
)
from najm_retrieval.parsing.models import (
    BlockType,
    ImageReference,
    PageReference,
    ParseMetrics,
    ParsedBlock,
    ParsedDocument,
    SourceSpan,
)
from najm_retrieval.parsing.profiles import (
    SUPPORTED_PROFILES,
)


DEFAULT_PARSER_NAME = "najm_lossless_parser"
DEFAULT_PARSER_VERSION = "0.1.0"

PAGE_TOKEN_PATTERN = re.compile(
    r"PageV\d+P\d+"
)


def build_parsed_document(
    *,
    source: OpenITISource,
    profile: str,
    block_records: Iterable[
        Mapping[str, Any]
    ],
    parser_name: str = DEFAULT_PARSER_NAME,
    parser_version: str = DEFAULT_PARSER_VERSION,
) -> ParsedDocument:
    """Convert raw parser block records into a typed document."""

    if profile not in SUPPORTED_PROFILES:
        supported = ", ".join(
            sorted(SUPPORTED_PROFILES)
        )

        raise ValueError(
            f"Unsupported parser profile "
            f"{profile!r}. Supported profiles: "
            f"{supported}"
        )

    blocks = tuple(
        build_parsed_block(
            source=source,
            version_id=source.version_id,
            record=record,
        )
        for record in block_records
    )

    return ParsedDocument(
        version_id=source.version_id,
        profile=profile,
        parser_name=parser_name,
        parser_version=parser_version,
        source_path=source.source_path,
        body_char_start=source.body_char_start,
        body_line_start=source.body_line_start,
        blocks=blocks,
        issues=(),
    )


def build_parsed_block(
    *,
    source: OpenITISource,
    version_id: str,
    record: Mapping[str, Any],
) -> ParsedBlock:
    """Convert one raw block record to ParsedBlock."""

    source_block_id = _require_text(
        record,
        "block_id",
    )

    block_type_text = _require_text(
        record,
        "block_type",
    )

    try:
        block_type = BlockType(
            block_type_text
        )
    except ValueError as error:
        raise ValueError(
            f"Unsupported block_type: "
            f"{block_type_text!r}"
        ) from error

    line_start = _require_integer(
        record,
        "line_start",
    )

    line_end = _require_integer(
        record,
        "line_end",
    )

    char_start = _require_integer(
        record,
        "char_start",
    )

    char_end = _require_integer(
        record,
        "char_end",
    )

    raw_text = record.get("raw_text")

    if not isinstance(raw_text, str):
        raise ValueError(
            "Block raw_text must be a string."
        )

    if char_end <= char_start:
        raise ValueError(
            "Block character span must have "
            "positive length."
        )

    if len(raw_text) != (
        char_end - char_start
    ):
        raise ValueError(
            f"Block {source_block_id!r} raw_text "
            "length does not match its span."
        )

    body_char_end = (
        source.body_char_start
        + len(source.body_text)
    )

    if (
        char_start < source.body_char_start
        or char_end > body_char_end
    ):
        raise ValueError(
            f"Block {source_block_id!r} lies "
            "outside the OpenITI body."
        )

    source_slice = source.source_text[
        char_start:char_end
    ]

    if source_slice != raw_text:
        raise ValueError(
            f"Block {source_block_id!r} "
            "raw_text does not match the "
            "source character range."
        )

    raw_attributes = record.get(
        "attributes",
        {},
    )

    if not isinstance(
        raw_attributes,
        Mapping,
    ):
        raise ValueError(
            "Block attributes must be a mapping."
        )

    attribute_values = dict(
        raw_attributes
    )

    group_id = record.get(
        "group_id"
    )

    if group_id is not None:
        if not isinstance(group_id, str):
            raise ValueError(
                "Block group_id must be a "
                "string or null."
            )

        attribute_values[
            "group_id"
        ] = group_id

    attributes = tuple(
        sorted(
            attribute_values.items(),
            key=lambda item: item[0],
        )
    )

    page = _build_page_reference(
        block_type=block_type,
        raw_text=raw_text,
        attributes=raw_attributes,
        source_line=line_start,
    )

    image = _build_image_reference(
        block_type=block_type,
        raw_text=raw_text,
        attributes=raw_attributes,
        source_line=line_start,
    )

    display_text, retrieval_text = (
        _build_content_texts(
            block_type=block_type,
            raw_text=raw_text,
        )
    )

    return ParsedBlock(
        block_id=(
            f"{version_id}:"
            f"{source_block_id}"
        ),
        block_type=block_type,
        span=SourceSpan(
            line_start=line_start,
            line_end=line_end,
            char_start=char_start,
            char_end=char_end,
        ),
        raw_text=raw_text,
        display_text=display_text,
        retrieval_text=retrieval_text,
        page=page,
        image=image,
        section_path=(),
        attributes=attributes,
        issues=(),
    )


def compute_parse_metrics(
    *,
    source: OpenITISource,
    document: ParsedDocument,
    runtime_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
) -> ParseMetrics:
    """Calculate strict coverage and reconstruction metrics."""

    body_char_start = (
        source.body_char_start
    )

    body_char_end = (
        body_char_start
        + len(source.body_text)
    )

    body_line_start = (
        source.body_line_start
    )

    body_line_end = (
        body_line_start
        + len(source.lines)
    )

    char_intervals = [
        (
            max(
                block.span.char_start,
                body_char_start,
            ),
            min(
                block.span.char_end,
                body_char_end,
            ),
        )
        for block in document.blocks
    ]

    line_intervals = [
        (
            max(
                block.span.line_start,
                body_line_start,
            ),
            min(
                block.span.line_end + 1,
                body_line_end,
            ),
        )
        for block in document.blocks
    ]

    raw_line_intervals = [
        (
            max(
                block.span.line_start,
                body_line_start,
            ),
            min(
                block.span.line_end + 1,
                body_line_end,
            ),
        )
        for block in document.blocks
        if block.block_type == BlockType.RAW
    ]

    covered_chars, overlapping_chars = (
        _measure_intervals(
            char_intervals
        )
    )

    covered_lines, _ = (
        _measure_intervals(
            line_intervals
        )
    )

    raw_line_count, _ = (
        _measure_intervals(
            raw_line_intervals
        )
    )

    total_body_chars = len(
        source.body_text
    )

    total_body_lines = len(
        source.lines
    )

    reconstructed_body = (
        document.reconstruct_body()
    )

    return ParseMetrics(
        total_body_lines=total_body_lines,
        covered_lines=covered_lines,
        uncovered_lines=(
            total_body_lines
            - covered_lines
        ),
        total_body_chars=total_body_chars,
        covered_chars=covered_chars,
        uncovered_chars=(
            total_body_chars
            - covered_chars
        ),
        overlapping_chars=overlapping_chars,
        reconstructed_chars=len(
            reconstructed_body
        ),
        reconstruction_matches_source=(
            reconstructed_body
            == source.body_text
        ),
        marker_count=len(
            document.marker_blocks
        ),
        raw_line_count=raw_line_count,
        issue_count=len(
            document.all_issues
        ),
        runtime_seconds=runtime_seconds,
        peak_memory_bytes=peak_memory_bytes,
    )


def _build_page_reference(
    *,
    block_type: BlockType,
    raw_text: str,
    attributes: Mapping[str, Any],
    source_line: int,
) -> PageReference | None:
    """Create a page reference for page-marker blocks."""

    if block_type != BlockType.PAGE_MARKER:
        return None

    marker_match = PAGE_TOKEN_PATTERN.search(
        raw_text
    )

    raw_marker = (
        marker_match.group(0)
        if marker_match is not None
        else raw_text.strip()
    )

    volume = attributes.get(
        "volume"
    )

    page = attributes.get(
        "page"
    )

    if not isinstance(volume, int):
        volume = None

    if not isinstance(page, int):
        page = None

    return PageReference(
        raw_marker=raw_marker,
        volume=volume,
        page=page,
        source_line=source_line,
    )


def _build_image_reference(
    *,
    block_type: BlockType,
    raw_text: str,
    attributes: Mapping[str, Any],
    source_line: int,
) -> ImageReference | None:
    """Create an image reference for image blocks."""

    if (
        block_type
        != BlockType.IMAGE_REFERENCE
    ):
        return None

    target = attributes.get(
        "target"
    )

    image_id: str | None = None
    image_url: str | None = None

    if isinstance(target, str):
        image_id = Path(
            target
        ).name

        if target.startswith(
            ("http://", "https://")
        ):
            image_url = target

    return ImageReference(
        raw_marker=raw_text.rstrip(
            "\r\n"
        ),
        image_id=image_id,
        image_url=image_url,
        source_line=source_line,
    )


def _build_content_texts(
    *,
    block_type: BlockType,
    raw_text: str,
) -> tuple[str, str]:
    """Build conservative pre-normalization text values."""

    if block_type in {
        BlockType.PAGE_MARKER,
        BlockType.IMAGE_REFERENCE,
        BlockType.MILESTONE,
        BlockType.BLANK,
    }:
        return "", ""

    content = raw_text.strip(
        " \t\r\n"
    )

    return content, content


def _measure_intervals(
    intervals: Iterable[
        tuple[int, int]
    ],
) -> tuple[int, int]:
    """Measure unique and multiply covered integer positions."""

    events: dict[int, int] = (
        defaultdict(int)
    )

    for start, end in intervals:
        if end <= start:
            continue

        events[start] += 1
        events[end] -= 1

    if not events:
        return 0, 0

    covered = 0
    overlapping = 0

    active = 0
    previous_position: int | None = None

    for position in sorted(events):
        if previous_position is not None:
            length = (
                position
                - previous_position
            )

            if active > 0:
                covered += length

            if active > 1:
                overlapping += length

        active += events[position]
        previous_position = position

    return covered, overlapping


def _require_text(
    mapping: Mapping[str, Any],
    field_name: str,
) -> str:
    """Read one mandatory non-empty string."""

    value = mapping.get(
        field_name
    )

    if (
        not isinstance(value, str)
        or not value
    ):
        raise ValueError(
            f"Block {field_name} must be "
            "a non-empty string."
        )

    return value


def _require_integer(
    mapping: Mapping[str, Any],
    field_name: str,
) -> int:
    """Read one mandatory integer."""

    value = mapping.get(
        field_name
    )

    if not isinstance(value, int):
        raise ValueError(
            f"Block {field_name} must be "
            "an integer."
        )

    return value