"""Models for structural paratext evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


PARATEXT_CATALOG_SCHEMA_VERSION = "1.0.0"


class ContentRole(str, Enum):
    """Structural role assigned to one serialized passage."""

    AUTHORIAL = "authorial"
    PARATEXT = "paratext"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParatextZone:
    """One contiguous structural zone inside a corpus version."""

    version_id: str
    start_ordinal: int
    end_ordinal: int
    role: ContentRole
    reason: str

    def __post_init__(self) -> None:
        if not self.version_id.strip():
            raise ValueError(
                "version_id must not be empty."
            )

        if (
            not isinstance(self.start_ordinal, int)
            or isinstance(self.start_ordinal, bool)
            or self.start_ordinal < 1
        ):
            raise ValueError(
                "start_ordinal must be a positive integer."
            )

        if (
            not isinstance(self.end_ordinal, int)
            or isinstance(self.end_ordinal, bool)
            or self.end_ordinal < self.start_ordinal
        ):
            raise ValueError(
                "end_ordinal must be at least start_ordinal."
            )

        if not self.reason.strip():
            raise ValueError(
                "reason must not be empty."
            )

    def contains(
        self,
        ordinal: int,
    ) -> bool:
        """Return whether an ordinal belongs to this zone."""

        return (
            self.start_ordinal
            <= ordinal
            <= self.end_ordinal
        )


@dataclass(frozen=True)
class PassageRoleEvidence:
    """Structural role evidence for one retrieved passage."""

    passage_id: str
    version_id: str
    ordinal: int
    role: ContentRole
    configured: bool
    reason: str | None

    def __post_init__(self) -> None:
        for label, value in (
            ("passage_id", self.passage_id),
            ("version_id", self.version_id),
        ):
            if not value.strip():
                raise ValueError(
                    f"{label} must not be empty."
                )

        if (
            not isinstance(self.ordinal, int)
            or isinstance(self.ordinal, bool)
            or self.ordinal < 1
        ):
            raise ValueError(
                "ordinal must be a positive integer."
            )

        if self.configured:
            if (
                self.role
                is ContentRole.UNKNOWN
            ):
                raise ValueError(
                    "Configured evidence cannot have an unknown role."
                )

            if (
                self.reason is None
                or not self.reason.strip()
            ):
                raise ValueError(
                    "Configured evidence requires a reason."
                )
        else:
            if (
                self.role
                is not ContentRole.UNKNOWN
            ):
                raise ValueError(
                    "Unconfigured evidence must have an unknown role."
                )

            if self.reason is not None:
                raise ValueError(
                    "Unconfigured evidence cannot contain a reason."
                )


@dataclass(frozen=True)
class ParatextEvidence:
    """Decision-free structural evidence across retrieved passages."""

    query_text: str
    hits: tuple[
        PassageRoleEvidence,
        ...,
    ]

    evaluated_hit_count: int

    authorial_hit_count: int
    paratext_hit_count: int
    mixed_hit_count: int
    unknown_hit_count: int

    paratext_or_mixed_hit_count: int
    paratext_or_mixed_rate: float

    top_hit_role: ContentRole | None
    top_hit_is_paratext: bool
    top_hit_is_mixed: bool
    top_hit_is_structurally_non_authorial: bool

    schema_version: str = (
        PARATEXT_CATALOG_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        if (
            self.schema_version
            != PARATEXT_CATALOG_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported paratext evidence schema version: "
                f"{self.schema_version}"
            )

        if (
            not isinstance(self.evaluated_hit_count, int)
            or isinstance(self.evaluated_hit_count, bool)
            or self.evaluated_hit_count < 0
            or self.evaluated_hit_count > 10
        ):
            raise ValueError(
                "evaluated_hit_count must be an integer from 0 to 10."
            )

        if len(self.hits) != self.evaluated_hit_count:
            raise ValueError(
                "hits must match evaluated_hit_count."
            )

        expected_counts = {
            ContentRole.AUTHORIAL: 0,
            ContentRole.PARATEXT: 0,
            ContentRole.MIXED: 0,
            ContentRole.UNKNOWN: 0,
        }

        for hit in self.hits:
            expected_counts[
                hit.role
            ] += 1

        if (
            self.authorial_hit_count
            != expected_counts[
                ContentRole.AUTHORIAL
            ]
        ):
            raise ValueError(
                "authorial_hit_count is inconsistent."
            )

        if (
            self.paratext_hit_count
            != expected_counts[
                ContentRole.PARATEXT
            ]
        ):
            raise ValueError(
                "paratext_hit_count is inconsistent."
            )

        if (
            self.mixed_hit_count
            != expected_counts[
                ContentRole.MIXED
            ]
        ):
            raise ValueError(
                "mixed_hit_count is inconsistent."
            )

        if (
            self.unknown_hit_count
            != expected_counts[
                ContentRole.UNKNOWN
            ]
        ):
            raise ValueError(
                "unknown_hit_count is inconsistent."
            )

        expected_non_authorial = (
            self.paratext_hit_count
            + self.mixed_hit_count
        )

        if (
            self.paratext_or_mixed_hit_count
            != expected_non_authorial
        ):
            raise ValueError(
                "paratext_or_mixed_hit_count is inconsistent."
            )

        expected_rate = (
            expected_non_authorial
            / self.evaluated_hit_count
            if self.evaluated_hit_count
            else 0.0
        )

        if (
            abs(
                self.paratext_or_mixed_rate
                - expected_rate
            )
            > 1e-12
        ):
            raise ValueError(
                "paratext_or_mixed_rate is inconsistent."
            )

        expected_top_role = (
            self.hits[0].role
            if self.hits
            else None
        )

        if self.top_hit_role != expected_top_role:
            raise ValueError(
                "top_hit_role is inconsistent."
            )

        expected_is_paratext = (
            expected_top_role
            is ContentRole.PARATEXT
        )

        expected_is_mixed = (
            expected_top_role
            is ContentRole.MIXED
        )

        expected_non_authorial_top = (
            expected_is_paratext
            or expected_is_mixed
        )

        if (
            self.top_hit_is_paratext
            != expected_is_paratext
        ):
            raise ValueError(
                "top_hit_is_paratext is inconsistent."
            )

        if (
            self.top_hit_is_mixed
            != expected_is_mixed
        ):
            raise ValueError(
                "top_hit_is_mixed is inconsistent."
            )

        if (
            self.top_hit_is_structurally_non_authorial
            != expected_non_authorial_top
        ):
            raise ValueError(
                "top_hit_is_structurally_non_authorial is inconsistent."
            )


__all__ = [
    "PARATEXT_CATALOG_SCHEMA_VERSION",
    "ContentRole",
    "ParatextEvidence",
    "ParatextZone",
    "PassageRoleEvidence",
]
