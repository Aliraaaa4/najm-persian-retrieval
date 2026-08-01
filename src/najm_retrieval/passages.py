"""Models for structure-aware retrieval passages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from najm_retrieval.parsing.models import (
    BlockType,
    SourceSpan,
)


PASSAGE_SCHEMA_VERSION = "1.0.0"


class PassageKind(str, Enum):
    """Supported structure-aware passage strategies."""

    DIWAN = "diwan"
    MATHNAWI = "mathnawi"
    MIXED_PROSE = "mixed_prose"


@dataclass(frozen=True)
class PassageConfig:
    """Validated configuration for passage construction."""

    diwan_target_verses: int = 8
    diwan_overlap_verses: int = 1
    diwan_minimum_tail_verses: int = 2

    mathnawi_target_verses: int = 8
    mathnawi_overlap_verses: int = 1
    mathnawi_minimum_tail_verses: int = 6

    prose_target_words: int = 180
    prose_soft_min_words: int = 80
    prose_hard_max_words: int = 300

    schema_version: str = PASSAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate all passage-construction limits."""

        _validate_verse_window(
            label="diwan",
            target=self.diwan_target_verses,
            overlap=self.diwan_overlap_verses,
            minimum_tail=(
                self.diwan_minimum_tail_verses
            ),
        )

        _validate_verse_window(
            label="mathnawi",
            target=self.mathnawi_target_verses,
            overlap=self.mathnawi_overlap_verses,
            minimum_tail=(
                self.mathnawi_minimum_tail_verses
            ),
        )

        if self.prose_soft_min_words <= 0:
            raise ValueError(
                "prose_soft_min_words must be positive."
            )

        if self.prose_target_words <= 0:
            raise ValueError(
                "prose_target_words must be positive."
            )

        if self.prose_hard_max_words <= 0:
            raise ValueError(
                "prose_hard_max_words must be positive."
            )

        if not (
            self.prose_soft_min_words
            <= self.prose_target_words
            <= self.prose_hard_max_words
        ):
            raise ValueError(
                "Prose limits must satisfy "
                "soft_min <= target <= hard_max."
            )

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must not be empty."
            )


@dataclass(frozen=True)
class PassageIssue:
    """A diagnostic produced during passage construction."""

    code: str
    message: str
    source_unit_ids: tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate issue contents."""

        if not self.code.strip():
            raise ValueError(
                "PassageIssue.code must not be empty."
            )

        if not self.message.strip():
            raise ValueError(
                "PassageIssue.message must not be empty."
            )

        if any(
            not unit_id.strip()
            for unit_id in self.source_unit_ids
        ):
            raise ValueError(
                "PassageIssue source unit IDs "
                "must not be empty."
            )


@dataclass(frozen=True)
class PassageBoundary:
    """Heading or section metadata attached to a passage."""

    unit_id: str
    unit_type: BlockType
    display_text: str = ""
    metadata: tuple[
        tuple[str, Any],
        ...,
    ] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate one structural boundary."""

        if not self.unit_id.strip():
            raise ValueError(
                "PassageBoundary.unit_id "
                "must not be empty."
            )

        unit_type_value = _block_type_value(
            self.unit_type
        )

        if unit_type_value not in {
            "heading",
            "section",
        }:
            raise ValueError(
                "PassageBoundary must refer to "
                "a heading or section."
            )

        _validate_metadata(
            self.metadata
        )

    @property
    def metadata_dict(
        self,
    ) -> dict[str, Any]:
        """Return boundary metadata as a dictionary."""

        return dict(
            self.metadata
        )


@dataclass(frozen=True)
class PassageMember:
    """One logical unit or split fragment inside a passage."""

    unit_id: str
    unit_type: BlockType

    display_text: str
    retrieval_text: str
    search_alias_text: str

    segment_index: int = 0
    segment_count: int = 1

    source_spans: tuple[
        SourceSpan,
        ...,
    ] = field(
        default_factory=tuple
    )

    metadata: tuple[
        tuple[str, Any],
        ...,
    ] = field(
        default_factory=tuple
    )

    source_issue_codes: tuple[
        str,
        ...,
    ] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate one passage member."""

        if not self.unit_id.strip():
            raise ValueError(
                "PassageMember.unit_id "
                "must not be empty."
            )

        unit_type_value = _block_type_value(
            self.unit_type
        )

        if unit_type_value not in {
            "verse",
            "paragraph",
            "raw",
        }:
            raise ValueError(
                "PassageMember must refer to "
                "a content unit."
            )

        if not self.display_text.strip():
            raise ValueError(
                "PassageMember.display_text "
                "must not be empty."
            )

        if not self.retrieval_text.strip():
            raise ValueError(
                "PassageMember.retrieval_text "
                "must not be empty."
            )

        if not self.search_alias_text.strip():
            raise ValueError(
                "PassageMember.search_alias_text "
                "must not be empty."
            )

        if self.segment_count <= 0:
            raise ValueError(
                "segment_count must be positive."
            )

        if not (
            0
            <= self.segment_index
            < self.segment_count
        ):
            raise ValueError(
                "segment_index must satisfy "
                "0 <= index < segment_count."
            )

        _validate_metadata(
            self.metadata
        )

        if any(
            not code.strip()
            for code in self.source_issue_codes
        ):
            raise ValueError(
                "Source issue codes must not be empty."
            )

    @property
    def is_fragment(self) -> bool:
        """Return whether this member is a split fragment."""

        return self.segment_count > 1

    @property
    def word_count(self) -> int:
        """Return normalized retrieval word count."""

        return len(
            self.retrieval_text.split()
        )

    @property
    def char_count(self) -> int:
        """Return normalized retrieval character count."""

        return len(
            self.retrieval_text
        )


@dataclass(frozen=True)
class Passage:
    """A structure-aware retrieval passage."""

    passage_id: str
    version_id: str
    profile: str
    kind: PassageKind

    ordinal: int
    include_in_index: bool

    members: tuple[
        PassageMember,
        ...,
    ]

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

    boundaries: tuple[
        PassageBoundary,
        ...,
    ] = field(
        default_factory=tuple
    )

    previous_passage_id: str | None = None
    next_passage_id: str | None = None

    issues: tuple[
        PassageIssue,
        ...,
    ] = field(
        default_factory=tuple
    )

    schema_version: str = PASSAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate one passage and its provenance."""

        if not self.passage_id.strip():
            raise ValueError(
                "passage_id must not be empty."
            )

        if not self.version_id.strip():
            raise ValueError(
                "version_id must not be empty."
            )

        if not self.profile.strip():
            raise ValueError(
                "profile must not be empty."
            )

        if self.ordinal < 0:
            raise ValueError(
                "ordinal must not be negative."
            )

        if not self.members:
            raise ValueError(
                "A passage must contain at least "
                "one member."
            )

        expected_prefix = (
            f"{self.version_id}:"
        )

        for member in self.members:
            if not member.unit_id.startswith(
                expected_prefix
            ):
                raise ValueError(
                    "Every member must belong to "
                    "the passage version."
                )

        for referenced_id in (
            self.heading_path
            + self.section_path
        ):
            if not referenced_id.startswith(
                expected_prefix
            ):
                raise ValueError(
                    "Heading and section references "
                    "must belong to the passage version."
                )

        boundary_ids = {
            boundary.unit_id
            for boundary in self.boundaries
        }

        context_ids = set(
            self.heading_path
            + self.section_path
        )

        if not boundary_ids.issubset(
            context_ids
        ):
            raise ValueError(
                "Passage boundaries must be present "
                "in heading_path or section_path."
            )

        if (
            self.previous_passage_id
            == self.passage_id
        ):
            raise ValueError(
                "A passage cannot be its own "
                "previous passage."
            )

        if (
            self.next_passage_id
            == self.passage_id
        ):
            raise ValueError(
                "A passage cannot be its own "
                "next passage."
            )

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must not be empty."
            )

    @property
    def source_unit_ids(
        self,
    ) -> tuple[str, ...]:
        """Return stable unique source unit IDs in order."""

        return tuple(
            dict.fromkeys(
                member.unit_id
                for member in self.members
            )
        )

    @property
    def source_spans(
        self,
    ) -> tuple[SourceSpan, ...]:
        """Return all member source spans in order."""

        return tuple(
            span
            for member in self.members
            for span in member.source_spans
        )

    @property
    def unit_count(self) -> int:
        """Return unique logical-unit count."""

        return len(
            self.source_unit_ids
        )

    @property
    def member_count(self) -> int:
        """Return member or fragment count."""

        return len(
            self.members
        )

    @property
    def has_split_member(self) -> bool:
        """Return whether any source unit was split."""

        return any(
            member.is_fragment
            for member in self.members
        )

    @property
    def display_text(self) -> str:
        """Return the joined user-facing passage text."""

        return _join_member_text(
            self.members,
            attribute="display_text",
        )

    @property
    def retrieval_text(self) -> str:
        """Return the joined embedding/search text."""

        return _join_member_text(
            self.members,
            attribute="retrieval_text",
        )

    @property
    def search_alias_text(self) -> str:
        """Return the joined loose lexical-search text."""

        return _join_member_text(
            self.members,
            attribute="search_alias_text",
        )

    @property
    def word_count(self) -> int:
        """Return retrieval word count."""

        return len(
            self.retrieval_text.split()
        )

    @property
    def parent_context_id(
        self,
    ) -> str | None:
        """Return the nearest structural parent ID."""

        if self.heading_path:
            return self.heading_path[-1]

        if self.section_path:
            return self.section_path[-1]

        return None


@dataclass(frozen=True)
class PassageBuildResult:
    """Passages plus corpus-level build diagnostics."""

    config: PassageConfig
    passages: tuple[
        Passage,
        ...,
    ]

    skipped_unit_ids: tuple[
        str,
        ...,
    ] = field(
        default_factory=tuple
    )

    issues: tuple[
        PassageIssue,
        ...,
    ] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate passage IDs and skipped-unit coverage."""

        passage_ids = [
            passage.passage_id
            for passage in self.passages
        ]

        if len(passage_ids) != len(
            set(passage_ids)
        ):
            raise ValueError(
                "Passage IDs must be unique."
            )

        if any(
            not unit_id.strip()
            for unit_id in self.skipped_unit_ids
        ):
            raise ValueError(
                "Skipped unit IDs must not be empty."
            )

        covered_ids = set(
            self.covered_unit_ids
        )

        overlapping_skipped_ids = (
            covered_ids
            & set(self.skipped_unit_ids)
        )

        if overlapping_skipped_ids:
            raise ValueError(
                "Skipped units must not also be "
                "covered by passages."
            )

    @property
    def passage_count(self) -> int:
        """Return generated passage count."""

        return len(
            self.passages
        )

    @property
    def indexable_passages(
        self,
    ) -> tuple[Passage, ...]:
        """Return only passages eligible for indexing."""

        return tuple(
            passage
            for passage in self.passages
            if passage.include_in_index
        )

    @property
    def covered_unit_ids(
        self,
    ) -> tuple[str, ...]:
        """Return unique covered unit IDs in order."""

        return tuple(
            dict.fromkeys(
                unit_id
                for passage in self.passages
                for unit_id
                in passage.source_unit_ids
            )
        )


def _validate_verse_window(
    *,
    label: str,
    target: int,
    overlap: int,
    minimum_tail: int,
) -> None:
    """Validate one poetry window configuration."""

    if target <= 0:
        raise ValueError(
            f"{label} target must be positive."
        )

    if not (
        0 <= overlap < target
    ):
        raise ValueError(
            f"{label} overlap must satisfy "
            "0 <= overlap < target."
        )

    if not (
        1 <= minimum_tail <= target
    ):
        raise ValueError(
            f"{label} minimum tail must satisfy "
            "1 <= minimum_tail <= target."
        )


def _validate_metadata(
    metadata: tuple[
        tuple[str, Any],
        ...,
    ],
) -> None:
    """Validate ordered metadata pairs."""

    for key, _ in metadata:
        if not isinstance(
            key,
            str,
        ):
            raise TypeError(
                "Metadata keys must be strings."
            )

        if not key.strip():
            raise ValueError(
                "Metadata keys must not be empty."
            )


def _block_type_value(
    unit_type: BlockType,
) -> str:
    """Return a stable serialized block-type value."""

    value = getattr(
        unit_type,
        "value",
        unit_type,
    )

    return str(value)


def _member_separator(
    left: PassageMember,
    right: PassageMember,
) -> str:
    """Choose a structure-preserving text separator."""

    if left.unit_id == right.unit_id:
        return " "

    left_type = _block_type_value(
        left.unit_type
    )

    right_type = _block_type_value(
        right.unit_type
    )

    if (
        left_type == "verse"
        and right_type == "verse"
    ):
        return "\n"

    return "\n\n"


def _join_member_text(
    members: tuple[
        PassageMember,
        ...,
    ],
    *,
    attribute: str,
) -> str:
    """Join member text without losing structural grouping."""

    if not members:
        return ""

    parts = [
        getattr(
            members[0],
            attribute,
        )
    ]

    for left, right in zip(
        members,
        members[1:],
    ):
        parts.append(
            _member_separator(
                left,
                right,
            )
        )

        parts.append(
            getattr(
                right,
                attribute,
            )
        )

    return "".join(
        parts
    )


__all__ = [
    "PASSAGE_SCHEMA_VERSION",
    "Passage",
    "PassageBoundary",
    "PassageBuildResult",
    "PassageConfig",
    "PassageIssue",
    "PassageKind",
    "PassageMember",
]
