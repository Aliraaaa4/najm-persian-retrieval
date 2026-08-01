"""Models for structural context around logical text units."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from najm_retrieval.parsing.models import (
    BlockType,
    ImageReference,
    PageReference,
    SourceSpan,
)
from najm_retrieval.text_preparation.models import (
    LogicalUnit,
)


STRUCTURAL_CONTEXT_TYPES = frozenset(
    {
        BlockType.PAGE_MARKER,
        BlockType.IMAGE_REFERENCE,
        BlockType.MILESTONE,
        BlockType.BLANK,
    }
)


@dataclass(frozen=True)
class ContextIssue:
    """One issue detected while resolving unit context."""

    code: str
    message: str
    marker_block_ids: tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate issue fields."""

        _validate_identifier(
            self.code,
            "Context issue code",
        )

        _validate_identifier(
            self.message,
            "Context issue message",
        )

        for block_id in self.marker_block_ids:
            _validate_identifier(
                block_id,
                "Context marker block ID",
            )


@dataclass(frozen=True)
class ContextMarker:
    """One structural marker relevant to a logical unit."""

    block_id: str
    marker_type: BlockType
    span: SourceSpan
    raw_text: str

    page: PageReference | None = None
    image: ImageReference | None = None

    attributes: tuple[
        tuple[str, Any],
        ...,
    ] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate marker invariants."""

        _validate_identifier(
            self.block_id,
            "Context marker block_id",
        )

        if not isinstance(
            self.marker_type,
            BlockType,
        ):
            raise TypeError(
                "marker_type must be a BlockType."
            )

        if self.marker_type not in (
            STRUCTURAL_CONTEXT_TYPES
        ):
            raise ValueError(
                "Context markers must use a "
                "structural block type."
            )

        _validate_span(
            self.span
        )

        if not isinstance(
            self.raw_text,
            str,
        ):
            raise TypeError(
                "Context marker raw_text must "
                "be a string."
            )

        if (
            self.page is not None
            and not isinstance(
                self.page,
                PageReference,
            )
        ):
            raise TypeError(
                "page must be a PageReference "
                "or None."
            )

        if (
            self.image is not None
            and not isinstance(
                self.image,
                ImageReference,
            )
        ):
            raise TypeError(
                "image must be an ImageReference "
                "or None."
            )

        if (
            self.marker_type
            == BlockType.PAGE_MARKER
            and self.page is None
        ):
            raise ValueError(
                "Page-marker context must include "
                "a PageReference."
            )

        if (
            self.marker_type
            != BlockType.PAGE_MARKER
            and self.page is not None
        ):
            raise ValueError(
                "Only page markers can carry "
                "a PageReference."
            )

        if (
            self.marker_type
            == BlockType.IMAGE_REFERENCE
            and self.image is None
        ):
            raise ValueError(
                "Image-marker context must include "
                "an ImageReference."
            )

        if (
            self.marker_type
            != BlockType.IMAGE_REFERENCE
            and self.image is not None
        ):
            raise ValueError(
                "Only image markers can carry "
                "an ImageReference."
            )

        _validate_attributes(
            self.attributes
        )


@dataclass(frozen=True)
class LogicalUnitContext:
    """Resolved structural context for one logical unit.

    Page markers are retained as boundary observations. This model
    does not assume whether a marker labels preceding or following
    text.
    """

    unit_id: str

    preceding_page_marker: (
        ContextMarker | None
    ) = None

    following_page_marker: (
        ContextMarker | None
    ) = None

    embedded_markers: tuple[
        ContextMarker,
        ...,
    ] = field(
        default_factory=tuple
    )

    heading_path: tuple[
        str,
        ...,
    ] = field(
        default_factory=tuple
    )

    section_path: tuple[
        str,
        ...,
    ] = field(
        default_factory=tuple
    )

    issues: tuple[
        ContextIssue,
        ...,
    ] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate context fields."""

        _validate_identifier(
            self.unit_id,
            "Context unit_id",
        )

        for marker, field_name in (
            (
                self.preceding_page_marker,
                "preceding_page_marker",
            ),
            (
                self.following_page_marker,
                "following_page_marker",
            ),
        ):
            if marker is None:
                continue

            if not isinstance(
                marker,
                ContextMarker,
            ):
                raise TypeError(
                    f"{field_name} must be a "
                    "ContextMarker or None."
                )

            if (
                marker.marker_type
                != BlockType.PAGE_MARKER
            ):
                raise ValueError(
                    f"{field_name} must reference "
                    "a page marker."
                )

        previous_marker: ContextMarker | None = None

        for marker in self.embedded_markers:
            if not isinstance(
                marker,
                ContextMarker,
            ):
                raise TypeError(
                    "Every embedded marker must "
                    "be a ContextMarker."
                )

            if (
                previous_marker is not None
                and marker.span.char_start
                < previous_marker.span.char_end
            ):
                raise ValueError(
                    "Embedded markers must be "
                    "ordered and non-overlapping."
                )

            previous_marker = marker

        for identifier in self.heading_path:
            _validate_identifier(
                identifier,
                "Heading path unit ID",
            )

        for identifier in self.section_path:
            _validate_identifier(
                identifier,
                "Section path unit ID",
            )

        for issue in self.issues:
            if not isinstance(
                issue,
                ContextIssue,
            ):
                raise TypeError(
                    "Every context issue must be "
                    "a ContextIssue."
                )

    @property
    def embedded_page_markers(
        self,
    ) -> tuple[ContextMarker, ...]:
        """Return page markers inside the logical unit."""

        return tuple(
            marker
            for marker in self.embedded_markers
            if marker.marker_type
            == BlockType.PAGE_MARKER
        )

    @property
    def embedded_image_markers(
        self,
    ) -> tuple[ContextMarker, ...]:
        """Return image references inside the logical unit."""

        return tuple(
            marker
            for marker in self.embedded_markers
            if marker.marker_type
            == BlockType.IMAGE_REFERENCE
        )

    @property
    def embedded_milestones(
        self,
    ) -> tuple[ContextMarker, ...]:
        """Return milestones inside the logical unit."""

        return tuple(
            marker
            for marker in self.embedded_markers
            if marker.marker_type
            == BlockType.MILESTONE
        )

    @property
    def spans_page_boundary(
        self,
    ) -> bool:
        """Return whether a page marker occurs inside the unit."""

        return bool(
            self.embedded_page_markers
        )


@dataclass(frozen=True)
class ContextualLogicalUnit:
    """A logical unit paired with its resolved context."""

    unit: LogicalUnit
    context: LogicalUnitContext

    def __post_init__(self) -> None:
        """Validate unit-context consistency."""

        if not isinstance(
            self.unit,
            LogicalUnit,
        ):
            raise TypeError(
                "unit must be a LogicalUnit."
            )

        if not isinstance(
            self.context,
            LogicalUnitContext,
        ):
            raise TypeError(
                "context must be a "
                "LogicalUnitContext."
            )

        if (
            self.unit.unit_id
            != self.context.unit_id
        ):
            raise ValueError(
                "Logical unit and context IDs "
                "must match."
            )

        envelope = self.unit.envelope_span

        for marker in (
            self.context.embedded_markers
        ):
            if (
                marker.span.char_start
                < envelope.char_start
                or marker.span.char_end
                > envelope.char_end
            ):
                raise ValueError(
                    "Embedded context marker lies "
                    "outside the logical-unit envelope."
                )

        preceding = (
            self.context.preceding_page_marker
        )

        if (
            preceding is not None
            and preceding.span.char_end
            > envelope.char_start
        ):
            raise ValueError(
                "Preceding page marker must end "
                "before the logical unit."
            )

        following = (
            self.context.following_page_marker
        )

        if (
            following is not None
            and following.span.char_start
            < envelope.char_end
        ):
            raise ValueError(
                "Following page marker must start "
                "after the logical unit."
            )


def _validate_identifier(
    value: object,
    field_name: str,
) -> None:
    """Validate a non-empty string."""

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a "
            "non-empty string."
        )


def _validate_span(
    span: object,
) -> None:
    """Validate one source span."""

    if not isinstance(
        span,
        SourceSpan,
    ):
        raise TypeError(
            "Context marker span must be "
            "a SourceSpan."
        )

    if span.line_start < 1:
        raise ValueError(
            "Source line numbers must start "
            "at one."
        )

    if span.line_end < span.line_start:
        raise ValueError(
            "Source line span is invalid."
        )

    if span.char_start < 0:
        raise ValueError(
            "Source character offsets cannot "
            "be negative."
        )

    if span.char_end <= span.char_start:
        raise ValueError(
            "Source character span must have "
            "positive length."
        )


def _validate_attributes(
    attributes: object,
) -> None:
    """Validate immutable marker attributes."""

    if not isinstance(
        attributes,
        tuple,
    ):
        raise TypeError(
            "Context marker attributes must "
            "be a tuple."
        )

    keys: set[str] = set()

    for item in attributes:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
        ):
            raise ValueError(
                "Every context attribute must "
                "be a two-item tuple."
            )

        key, _ = item

        if (
            not isinstance(key, str)
            or not key
        ):
            raise ValueError(
                "Context attribute keys must be "
                "non-empty strings."
            )

        if key in keys:
            raise ValueError(
                f"Duplicate context attribute: "
                f"{key!r}."
            )

        keys.add(key)
