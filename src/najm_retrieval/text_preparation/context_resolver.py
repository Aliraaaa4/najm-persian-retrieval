"""Resolve structural context for assembled logical units."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from typing import Any

from najm_retrieval.parsing.models import (
    BlockType,
    ImageReference,
    PageReference,
    SourceSpan,
)
from najm_retrieval.text_preparation.attributes import (
    attributes_to_dict,
)
from najm_retrieval.text_preparation.context_models import (
    ContextIssue,
    ContextMarker,
    ContextualLogicalUnit,
    LogicalUnitContext,
    STRUCTURAL_CONTEXT_TYPES,
)
from najm_retrieval.text_preparation.models import (
    LogicalUnit,
)


def resolve_logical_unit_contexts(
    *,
    units: Iterable[LogicalUnit],
    blocks: Iterable[Mapping[str, Any]],
) -> tuple[ContextualLogicalUnit, ...]:
    """Resolve structural context for logical units.

    Logical units are ordered by source character position.

    Page markers are retained as observations before, after, or
    inside a unit. No assumption is made here about whether an
    OpenITI page marker labels preceding or following text.

    Heading paths are inferred from heading levels. Sections are
    currently treated as flat scopes because the parsed corpus does
    not provide an explicit hierarchical section level.
    """

    ordered_units = _prepare_units(
        units
    )

    markers = _prepare_context_markers(
        blocks
    )

    marker_starts = [
        marker.span.char_start
        for marker in markers
    ]

    page_markers = tuple(
        marker
        for marker in markers
        if marker.marker_type
        == BlockType.PAGE_MARKER
    )

    page_starts = [
        marker.span.char_start
        for marker in page_markers
    ]

    page_ends = [
        marker.span.char_end
        for marker in page_markers
    ]

    heading_stack: list[
        tuple[int, str]
    ] = []

    active_section_path: tuple[
        str,
        ...,
    ] = ()

    results: list[
        ContextualLogicalUnit
    ] = []

    for unit in ordered_units:
        envelope = unit.envelope_span

        issues: list[
            ContextIssue
        ] = []

        embedded_markers = (
            _find_embedded_markers(
                markers=markers,
                marker_starts=marker_starts,
                envelope=envelope,
            )
        )

        preceding_page_marker = (
            _find_preceding_page_marker(
                page_markers=page_markers,
                page_ends=page_ends,
                envelope=envelope,
            )
        )

        following_page_marker = (
            _find_following_page_marker(
                page_markers=page_markers,
                page_starts=page_starts,
                envelope=envelope,
            )
        )

        if unit.unit_type == BlockType.SECTION:
            context_heading_path: tuple[
                str,
                ...,
            ] = ()

            context_section_path: tuple[
                str,
                ...,
            ] = ()

            heading_level: int | None = None

        elif unit.unit_type == BlockType.HEADING:
            heading_level, heading_issue = (
                _read_heading_level(
                    unit
                )
            )

            if heading_issue is not None:
                issues.append(
                    heading_issue
                )

            parent_heading_stack = [
                item
                for item in heading_stack
                if item[0] < heading_level
            ]

            context_heading_path = tuple(
                unit_id
                for _, unit_id
                in parent_heading_stack
            )

            context_section_path = (
                active_section_path
            )

        else:
            heading_level = None

            context_heading_path = tuple(
                unit_id
                for _, unit_id
                in heading_stack
            )

            context_section_path = (
                active_section_path
            )

        context = LogicalUnitContext(
            unit_id=unit.unit_id,
            preceding_page_marker=(
                preceding_page_marker
            ),
            following_page_marker=(
                following_page_marker
            ),
            embedded_markers=(
                embedded_markers
            ),
            heading_path=(
                context_heading_path
            ),
            section_path=(
                context_section_path
            ),
            issues=tuple(issues),
        )

        results.append(
            ContextualLogicalUnit(
                unit=unit,
                context=context,
            )
        )

        if unit.unit_type == BlockType.SECTION:
            active_section_path = (
                unit.unit_id,
            )

            heading_stack = []

        elif unit.unit_type == BlockType.HEADING:
            assert heading_level is not None

            heading_stack = [
                item
                for item in heading_stack
                if item[0] < heading_level
            ]

            heading_stack.append(
                (
                    heading_level,
                    unit.unit_id,
                )
            )

    return tuple(results)


def _prepare_units(
    units: Iterable[LogicalUnit],
) -> tuple[LogicalUnit, ...]:
    """Validate and order logical units."""

    materialized = list(units)

    seen_unit_ids: set[str] = set()

    for unit in materialized:
        if not isinstance(
            unit,
            LogicalUnit,
        ):
            raise TypeError(
                "Every unit must be a LogicalUnit."
            )

        if unit.unit_id in seen_unit_ids:
            raise ValueError(
                f"Duplicate logical unit ID: "
                f"{unit.unit_id!r}."
            )

        seen_unit_ids.add(
            unit.unit_id
        )

    materialized.sort(
        key=lambda unit: (
            unit.envelope_span.char_start,
            unit.envelope_span.char_end,
            unit.unit_id,
        )
    )

    previous: LogicalUnit | None = None

    for unit in materialized:
        if (
            previous is not None
            and unit.envelope_span.char_start
            < previous.envelope_span.char_end
        ):
            raise ValueError(
                "Logical-unit envelopes must be "
                "ordered and non-overlapping."
            )

        previous = unit

    return tuple(materialized)


def _prepare_context_markers(
    blocks: Iterable[Mapping[str, Any]],
) -> tuple[ContextMarker, ...]:
    """Convert structural parsed blocks to context markers."""

    markers: list[
        ContextMarker
    ] = []

    seen_block_ids: set[str] = set()

    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError(
                "Every parsed block must be a mapping."
            )

        block_type = _read_block_type(
            block
        )

        if block_type not in (
            STRUCTURAL_CONTEXT_TYPES
        ):
            continue

        block_id = _require_text(
            block,
            "block_id",
        )

        if block_id in seen_block_ids:
            raise ValueError(
                f"Duplicate structural block ID: "
                f"{block_id!r}."
            )

        seen_block_ids.add(
            block_id
        )

        span = _read_span(
            block.get("span"),
            block_id=block_id,
        )

        raw_text = block.get(
            "raw_text"
        )

        if not isinstance(raw_text, str):
            raise TypeError(
                f"Structural block {block_id!r} "
                "raw_text must be a string."
            )

        attributes = attributes_to_dict(
            block.get("attributes")
        )

        page = (
            _read_page_reference(
                block.get("page"),
                block_id=block_id,
            )
            if block_type
            == BlockType.PAGE_MARKER
            else None
        )

        image = (
            _read_image_reference(
                block.get("image"),
                block_id=block_id,
            )
            if block_type
            == BlockType.IMAGE_REFERENCE
            else None
        )

        markers.append(
            ContextMarker(
                block_id=block_id,
                marker_type=block_type,
                span=span,
                raw_text=raw_text,
                page=page,
                image=image,
                attributes=tuple(
                    sorted(
                        attributes.items(),
                        key=lambda item: item[0],
                    )
                ),
            )
        )

    markers.sort(
        key=lambda marker: (
            marker.span.char_start,
            marker.span.char_end,
            marker.block_id,
        )
    )

    previous: ContextMarker | None = None

    for marker in markers:
        if (
            previous is not None
            and marker.span.char_start
            < previous.span.char_end
        ):
            raise ValueError(
                "Structural marker spans must be "
                "ordered and non-overlapping."
            )

        previous = marker

    return tuple(markers)


def _find_embedded_markers(
    *,
    markers: tuple[ContextMarker, ...],
    marker_starts: list[int],
    envelope: SourceSpan,
) -> tuple[ContextMarker, ...]:
    """Return structural markers inside one unit envelope."""

    index = bisect_left(
        marker_starts,
        envelope.char_start,
    )

    result: list[
        ContextMarker
    ] = []

    while index < len(markers):
        marker = markers[index]

        if (
            marker.span.char_start
            >= envelope.char_end
        ):
            break

        if (
            marker.span.char_end
            <= envelope.char_end
        ):
            result.append(
                marker
            )

        index += 1

    return tuple(result)


def _find_preceding_page_marker(
    *,
    page_markers: tuple[
        ContextMarker,
        ...,
    ],
    page_ends: list[int],
    envelope: SourceSpan,
) -> ContextMarker | None:
    """Return the closest page marker ending before a unit."""

    index = (
        bisect_right(
            page_ends,
            envelope.char_start,
        )
        - 1
    )

    if index < 0:
        return None

    return page_markers[index]


def _find_following_page_marker(
    *,
    page_markers: tuple[
        ContextMarker,
        ...,
    ],
    page_starts: list[int],
    envelope: SourceSpan,
) -> ContextMarker | None:
    """Return the closest page marker starting after a unit."""

    index = bisect_left(
        page_starts,
        envelope.char_end,
    )

    if index >= len(page_markers):
        return None

    return page_markers[index]


def _read_heading_level(
    unit: LogicalUnit,
) -> tuple[
    int,
    ContextIssue | None,
]:
    """Read heading level, using a safe fallback when invalid."""

    attributes = dict(
        unit.attributes
    )

    value = attributes.get(
        "level"
    )

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
    ):
        return value, None

    return (
        1,
        ContextIssue(
            code="invalid_heading_level",
            message=(
                "Heading level was missing or "
                "invalid; level 1 was used."
            ),
        ),
    )


def _read_block_type(
    block: Mapping[str, Any],
) -> BlockType:
    """Read one parsed block type."""

    value = block.get(
        "block_type"
    )

    if isinstance(value, BlockType):
        return value

    if not isinstance(value, str):
        raise TypeError(
            "block_type must be a string "
            "or BlockType."
        )

    try:
        return BlockType(value)
    except ValueError as error:
        raise ValueError(
            f"Unsupported block_type: {value!r}."
        ) from error


def _read_span(
    value: object,
    *,
    block_id: str,
) -> SourceSpan:
    """Read a source span from Python or JSON data."""

    if isinstance(value, SourceSpan):
        span = value

    else:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Block {block_id!r} span must "
                "be a mapping or SourceSpan."
            )

        span = SourceSpan(
            line_start=_require_integer(
                value,
                "line_start",
            ),
            line_end=_require_integer(
                value,
                "line_end",
            ),
            char_start=_require_integer(
                value,
                "char_start",
            ),
            char_end=_require_integer(
                value,
                "char_end",
            ),
        )

    if span.line_start < 1:
        raise ValueError(
            f"Block {block_id!r} line_start "
            "must be at least one."
        )

    if span.line_end < span.line_start:
        raise ValueError(
            f"Block {block_id!r} has an "
            "invalid line span."
        )

    if span.char_start < 0:
        raise ValueError(
            f"Block {block_id!r} char_start "
            "cannot be negative."
        )

    if span.char_end <= span.char_start:
        raise ValueError(
            f"Block {block_id!r} has an "
            "invalid character span."
        )

    return span


def _read_page_reference(
    value: object,
    *,
    block_id: str,
) -> PageReference:
    """Read a page reference from JSON-compatible data."""

    if isinstance(value, PageReference):
        return value

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Page marker {block_id!r} does not "
            "contain a valid page reference."
        )

    raw_marker = _require_text(
        value,
        "raw_marker",
    )

    source_line = _require_integer(
        value,
        "source_line",
    )

    volume = _optional_integer(
        value.get("volume"),
        field_name="volume",
    )

    page = _optional_integer(
        value.get("page"),
        field_name="page",
    )

    return PageReference(
        raw_marker=raw_marker,
        volume=volume,
        page=page,
        source_line=source_line,
    )


def _read_image_reference(
    value: object,
    *,
    block_id: str,
) -> ImageReference:
    """Read an image reference from JSON-compatible data."""

    if isinstance(value, ImageReference):
        return value

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Image marker {block_id!r} does not "
            "contain a valid image reference."
        )

    raw_marker = _require_text(
        value,
        "raw_marker",
    )

    source_line = _require_integer(
        value,
        "source_line",
    )

    image_id = _optional_text(
        value.get("image_id"),
        field_name="image_id",
    )

    image_url = _optional_text(
        value.get("image_url"),
        field_name="image_url",
    )

    return ImageReference(
        raw_marker=raw_marker,
        image_id=image_id,
        image_url=image_url,
        source_line=source_line,
    )


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
            f"{field_name} must be a "
            "non-empty string."
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

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"{field_name} must be an integer."
        )

    return value


def _optional_integer(
    value: object,
    *,
    field_name: str,
) -> int | None:
    """Read one optional integer."""

    if value is None:
        return None

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"{field_name} must be an integer "
            "or None."
        )

    return value


def _optional_text(
    value: object,
    *,
    field_name: str,
) -> str | None:
    """Read one optional string."""

    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string "
            "or None."
        )

    return value
