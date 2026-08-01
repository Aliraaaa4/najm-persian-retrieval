"""Models for logical units assembled from parsed blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from najm_retrieval.parsing.models import (
    BlockType,
    SourceSpan,
)


CONTENT_BLOCK_TYPES = frozenset(
    {
        BlockType.HEADING,
        BlockType.SECTION,
        BlockType.VERSE,
        BlockType.PARAGRAPH,
        BlockType.RAW,
    }
)


@dataclass(frozen=True)
class AssemblyIssue:
    """One issue detected while assembling logical units."""

    code: str
    message: str
    source_block_ids: tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate issue fields."""

        if (
            not isinstance(self.code, str)
            or not self.code.strip()
        ):
            raise ValueError(
                "Assembly issue code must be "
                "a non-empty string."
            )

        if (
            not isinstance(self.message, str)
            or not self.message.strip()
        ):
            raise ValueError(
                "Assembly issue message must be "
                "a non-empty string."
            )

        for block_id in self.source_block_ids:
            if (
                not isinstance(block_id, str)
                or not block_id
            ):
                raise ValueError(
                    "Assembly issue block IDs must "
                    "be non-empty strings."
                )


@dataclass(frozen=True)
class LogicalUnit:
    """One logical textual unit assembled from parsed blocks.

    A logical unit can be a verse, paragraph, heading, section,
    or raw OCR unit. Its source blocks do not need to be physically
    adjacent because structural markers may occur between them.
    """

    unit_id: str
    version_id: str
    group_id: str
    unit_type: BlockType

    source_block_ids: tuple[str, ...]
    source_spans: tuple[SourceSpan, ...]
    raw_parts: tuple[str, ...]

    attributes: tuple[
        tuple[str, Any],
        ...,
    ] = field(
        default_factory=tuple
    )

    issues: tuple[
        AssemblyIssue,
        ...,
    ] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate structural invariants."""

        self._validate_identifier(
            self.unit_id,
            "unit_id",
        )

        self._validate_identifier(
            self.version_id,
            "version_id",
        )

        self._validate_identifier(
            self.group_id,
            "group_id",
        )

        if not isinstance(
            self.unit_type,
            BlockType,
        ):
            raise TypeError(
                "unit_type must be a BlockType."
            )

        if self.unit_type not in (
            CONTENT_BLOCK_TYPES
        ):
            raise ValueError(
                "Logical units can only use "
                "content block types."
            )

        component_count = len(
            self.source_block_ids
        )

        if component_count == 0:
            raise ValueError(
                "A logical unit must contain "
                "at least one source block."
            )

        if len(self.source_spans) != (
            component_count
        ):
            raise ValueError(
                "source_spans must have the same "
                "length as source_block_ids."
            )

        if len(self.raw_parts) != (
            component_count
        ):
            raise ValueError(
                "raw_parts must have the same "
                "length as source_block_ids."
            )

        for block_id in self.source_block_ids:
            self._validate_identifier(
                block_id,
                "source block ID",
            )

        for raw_part in self.raw_parts:
            if not isinstance(raw_part, str):
                raise TypeError(
                    "Every raw part must be "
                    "a string."
                )

        self._validate_source_spans()
        self._validate_attributes()

        for issue in self.issues:
            if not isinstance(
                issue,
                AssemblyIssue,
            ):
                raise TypeError(
                    "Every issue must be an "
                    "AssemblyIssue."
                )

    @staticmethod
    def _validate_identifier(
        value: object,
        field_name: str,
    ) -> None:
        """Validate one identifier."""

        if (
            not isinstance(value, str)
            or not value
        ):
            raise ValueError(
                f"{field_name} must be a "
                "non-empty string."
            )

    def _validate_source_spans(self) -> None:
        """Validate source span order and overlap."""

        previous_span: SourceSpan | None = None

        for span in self.source_spans:
            if not isinstance(
                span,
                SourceSpan,
            ):
                raise TypeError(
                    "Every source span must be "
                    "a SourceSpan."
                )

            if span.line_start < 1:
                raise ValueError(
                    "Source line numbers must "
                    "start at one."
                )

            if span.line_end < span.line_start:
                raise ValueError(
                    "Source line span is invalid."
                )

            if span.char_start < 0:
                raise ValueError(
                    "Source character offsets "
                    "cannot be negative."
                )

            if span.char_end <= span.char_start:
                raise ValueError(
                    "Source character span must "
                    "have positive length."
                )

            if (
                previous_span is not None
                and span.char_start
                < previous_span.char_end
            ):
                raise ValueError(
                    "Source spans must be ordered "
                    "and non-overlapping."
                )

            previous_span = span

    def _validate_attributes(self) -> None:
        """Validate attribute pairs and duplicate keys."""

        seen_keys: set[str] = set()

        for item in self.attributes:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
            ):
                raise ValueError(
                    "Every attribute must be a "
                    "two-item tuple."
                )

            key, _ = item

            if (
                not isinstance(key, str)
                or not key
            ):
                raise ValueError(
                    "Attribute keys must be "
                    "non-empty strings."
                )

            if key in seen_keys:
                raise ValueError(
                    f"Duplicate logical unit "
                    f"attribute: {key!r}."
                )

            seen_keys.add(key)

    @property
    def component_count(self) -> int:
        """Return the number of contributing source blocks."""

        return len(
            self.source_block_ids
        )

    @property
    def raw_text(self) -> str:
        """Join exact source block texts in source order."""

        return "".join(
            self.raw_parts
        )

    @property
    def envelope_span(self) -> SourceSpan:
        """Return the outer boundary containing all components.

        The envelope may include structural markers located between
        source components. Exact provenance remains available through
        source_spans.
        """

        first_span = self.source_spans[0]
        last_span = self.source_spans[-1]

        return SourceSpan(
            line_start=first_span.line_start,
            line_end=last_span.line_end,
            char_start=first_span.char_start,
            char_end=last_span.char_end,
        )

    @property
    def has_source_gaps(self) -> bool:
        """Return whether source components are non-contiguous."""

        return any(
            current.char_start
            > previous.char_end
            for previous, current in zip(
                self.source_spans,
                self.source_spans[1:],
            )
        )
