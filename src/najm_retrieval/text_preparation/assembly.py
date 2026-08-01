"""Assemble logical text units from parsed block mappings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from najm_retrieval.parsing.models import (
    BlockType,
    SourceSpan,
)
from najm_retrieval.text_preparation.attributes import (
    attributes_to_dict,
)
from najm_retrieval.text_preparation.models import (
    AssemblyIssue,
    CONTENT_BLOCK_TYPES,
    LogicalUnit,
)


CONTROL_ATTRIBUTE_KEYS = frozenset(
    {
        "group_id",
        "continuation",
        "orphan_continuation",
    }
)


@dataclass(frozen=True)
class _PreparedBlock:
    """Validated content block used during assembly."""

    input_index: int
    block_id: str
    block_type: BlockType
    span: SourceSpan
    raw_text: str
    attributes: dict[str, Any]


def assemble_logical_units(
    *,
    version_id: str,
    blocks: Iterable[Mapping[str, Any]],
) -> tuple[LogicalUnit, ...]:
    """Assemble content blocks into logical units.

    Content blocks are grouped by ``group_id``. Structural blocks
    such as page markers, image references, milestones, and blank
    lines do not become logical-unit components.

    Components are sorted by their source character offsets, so
    structural markers may occur between members of one group.
    """

    if (
        not isinstance(version_id, str)
        or not version_id
    ):
        raise ValueError(
            "version_id must be a non-empty string."
        )

    groups: dict[
        str,
        list[_PreparedBlock],
    ] = {}

    seen_block_ids: set[str] = set()

    for input_index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise TypeError(
                "Every parsed block must be a mapping."
            )

        block_type = _read_block_type(
            block
        )

        if block_type not in CONTENT_BLOCK_TYPES:
            continue

        block_id = _require_text(
            block,
            "block_id",
        )

        if block_id in seen_block_ids:
            raise ValueError(
                f"Duplicate content block ID: "
                f"{block_id!r}."
            )

        seen_block_ids.add(block_id)

        span = _read_span(
            block.get("span"),
            block_id=block_id,
        )

        raw_text = block.get(
            "raw_text"
        )

        if not isinstance(raw_text, str):
            raise TypeError(
                f"Block {block_id!r} raw_text "
                "must be a string."
            )

        if len(raw_text) != (
            span.char_end - span.char_start
        ):
            raise ValueError(
                f"Block {block_id!r} raw_text "
                "length does not match its span."
            )

        attributes = attributes_to_dict(
            block.get("attributes")
        )

        _merge_top_level_group_id(
            block=block,
            attributes=attributes,
            block_id=block_id,
        )

        group_id_value = attributes.get(
            "group_id"
        )

        if (
            not isinstance(group_id_value, str)
            or not group_id_value
        ):
            raise ValueError(
                f"Content block {block_id!r} "
                "does not have a valid group_id."
            )

        prepared_block = _PreparedBlock(
            input_index=input_index,
            block_id=block_id,
            block_type=block_type,
            span=span,
            raw_text=raw_text,
            attributes=attributes,
        )

        groups.setdefault(
            group_id_value,
            [],
        ).append(
            prepared_block
        )

    units: list[LogicalUnit] = []

    for group_id, components in groups.items():
        units.append(
            _assemble_group(
                version_id=version_id,
                group_id=group_id,
                components=components,
            )
        )

    units.sort(
        key=lambda unit: (
            unit.envelope_span.char_start,
            unit.envelope_span.char_end,
            unit.unit_id,
        )
    )

    return tuple(units)


def _assemble_group(
    *,
    version_id: str,
    group_id: str,
    components: list[_PreparedBlock],
) -> LogicalUnit:
    """Assemble one group of prepared content blocks."""

    ordered_components = sorted(
        components,
        key=lambda component: (
            component.span.char_start,
            component.span.char_end,
            component.input_index,
        ),
    )

    block_types = {
        component.block_type
        for component in ordered_components
    }

    if len(block_types) != 1:
        readable_types = ", ".join(
            sorted(
                block_type.value
                for block_type in block_types
            )
        )

        raise ValueError(
            f"Logical group {group_id!r} "
            "contains multiple block types: "
            f"{readable_types}."
        )

    issues: list[AssemblyIssue] = []

    original_block_ids = tuple(
        component.block_id
        for component in components
    )

    ordered_block_ids = tuple(
        component.block_id
        for component in ordered_components
    )

    if ordered_block_ids != original_block_ids:
        issues.append(
            AssemblyIssue(
                code="input_order_corrected",
                message=(
                    "Source components were reordered "
                    "using character offsets."
                ),
                source_block_ids=ordered_block_ids,
            )
        )

    gap_block_ids = _find_gap_block_ids(
        ordered_components
    )

    if gap_block_ids:
        issues.append(
            AssemblyIssue(
                code="source_gap",
                message=(
                    "One or more structural source "
                    "ranges occur between components."
                ),
                source_block_ids=gap_block_ids,
            )
        )

    attributes, attribute_issues = (
        _merge_component_attributes(
            ordered_components
        )
    )

    issues.extend(
        attribute_issues
    )

    unit_type = ordered_components[
        0
    ].block_type

    return LogicalUnit(
        unit_id=(
            f"{version_id}:{group_id}"
        ),
        version_id=version_id,
        group_id=group_id,
        unit_type=unit_type,
        source_block_ids=ordered_block_ids,
        source_spans=tuple(
            component.span
            for component in ordered_components
        ),
        raw_parts=tuple(
            component.raw_text
            for component in ordered_components
        ),
        attributes=attributes,
        issues=tuple(issues),
    )


def _read_block_type(
    block: Mapping[str, Any],
) -> BlockType:
    """Read and validate one block type."""

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
    """Read one source span from JSON-compatible data."""

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


def _merge_top_level_group_id(
    *,
    block: Mapping[str, Any],
    attributes: dict[str, Any],
    block_id: str,
) -> None:
    """Support parser records with a top-level group_id."""

    top_level_group_id = block.get(
        "group_id"
    )

    if top_level_group_id is None:
        return

    if (
        not isinstance(top_level_group_id, str)
        or not top_level_group_id
    ):
        raise ValueError(
            f"Block {block_id!r} top-level "
            "group_id must be a non-empty string."
        )

    attribute_group_id = attributes.get(
        "group_id"
    )

    if (
        attribute_group_id is not None
        and attribute_group_id
        != top_level_group_id
    ):
        raise ValueError(
            f"Block {block_id!r} has conflicting "
            "group_id values."
        )

    attributes[
        "group_id"
    ] = top_level_group_id


def _merge_component_attributes(
    components: list[_PreparedBlock],
) -> tuple[
    tuple[tuple[str, Any], ...],
    tuple[AssemblyIssue, ...],
]:
    """Merge stable semantic attributes from group components."""

    values_by_key: dict[
        str,
        list[Any],
    ] = defaultdict(list)

    blocks_by_key: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for component in components:
        for key, value in (
            component.attributes.items()
        ):
            if key in CONTROL_ATTRIBUTE_KEYS:
                continue

            blocks_by_key[key].append(
                component.block_id
            )

            values = values_by_key[key]

            if not any(
                existing == value
                for existing in values
            ):
                values.append(value)

    merged_attributes: list[
        tuple[str, Any]
    ] = []

    issues: list[
        AssemblyIssue
    ] = []

    for key in sorted(values_by_key):
        values = values_by_key[key]

        if len(values) == 1:
            merged_attributes.append(
                (
                    key,
                    values[0],
                )
            )
            continue

        issues.append(
            AssemblyIssue(
                code="attribute_conflict",
                message=(
                    f"Attribute {key!r} has "
                    "multiple values inside the "
                    "logical group and was omitted."
                ),
                source_block_ids=tuple(
                    blocks_by_key[key]
                ),
            )
        )

    return (
        tuple(merged_attributes),
        tuple(issues),
    )


def _find_gap_block_ids(
    components: list[_PreparedBlock],
) -> tuple[str, ...]:
    """Return component IDs adjacent to source gaps."""

    result: list[str] = []

    for previous, current in zip(
        components,
        components[1:],
    ):
        if (
            current.span.char_start
            > previous.span.char_end
        ):
            result.extend(
                [
                    previous.block_id,
                    current.block_id,
                ]
            )

    return tuple(
        dict.fromkeys(result)
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
    """Read one mandatory integer value."""

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
