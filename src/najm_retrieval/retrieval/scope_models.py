"""Models for corpus-scope and source-attribution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


SCOPE_CATALOG_SCHEMA_VERSION = "1.0.0"


class ScopeEntityKind(str, Enum):
    """Kinds of named entities that constrain corpus scope."""

    AUTHOR = "author"
    WORK = "work"


@dataclass(frozen=True)
class ScopeCatalogEntity:
    """One author or work recognized by the scope catalog."""

    entity_id: str
    kind: ScopeEntityKind
    label_fa: str
    aliases: tuple[str, ...]
    in_corpus: bool
    version_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("entity_id", self.entity_id),
            ("label_fa", self.label_fa),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must not be empty.")

        if not self.aliases:
            raise ValueError("aliases must contain at least one value.")

        if any(
            not isinstance(alias, str) or not alias.strip()
            for alias in self.aliases
        ):
            raise ValueError("Every alias must be a non-empty string.")

        if len(self.aliases) != len(set(self.aliases)):
            raise ValueError("aliases must be unique.")

        if len(self.version_ids) != len(set(self.version_ids)):
            raise ValueError("version_ids must be unique.")

        if self.in_corpus and not self.version_ids:
            raise ValueError(
                "In-corpus entities must resolve to at least one version."
            )

        if not self.in_corpus and self.version_ids:
            raise ValueError(
                "Out-of-corpus entities cannot resolve to corpus versions."
            )


@dataclass(frozen=True)
class ScopeMention:
    """One catalog entity explicitly mentioned in a query."""

    entity_id: str
    kind: ScopeEntityKind
    label_fa: str
    matched_alias: str
    in_corpus: bool
    version_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.matched_alias.strip():
            raise ValueError("matched_alias must not be empty.")


@dataclass(frozen=True)
class ScopeEvidence:
    """Decision-free corpus-scope evidence for one retrieval result."""

    query_text: str
    mentions: tuple[ScopeMention, ...]

    explicit_scope: bool
    in_corpus_scope_mentioned: bool
    known_out_of_corpus_scope_mentioned: bool

    requested_author_ids: tuple[str, ...]
    requested_work_ids: tuple[str, ...]
    known_out_of_corpus_entity_ids: tuple[str, ...]
    requested_version_ids: tuple[str, ...]

    evaluated_hit_count: int
    retrieved_version_ids_at_10: tuple[str, ...]
    matching_hit_count_at_10: int
    matching_hit_rate_at_10: float | None

    top_hit_version_id: str | None
    top_hit_matches_requested_scope: bool | None
    source_attribution_conflict: bool

    schema_version: str = SCOPE_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError("query_text must not be empty.")

        if self.schema_version != SCOPE_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported scope schema version: {self.schema_version}"
            )

        if self.explicit_scope != bool(self.mentions):
            raise ValueError(
                "explicit_scope must match whether mentions are present."
            )

        expected_in_corpus = any(
            mention.in_corpus
            for mention in self.mentions
        )
        expected_out_of_corpus = any(
            not mention.in_corpus
            for mention in self.mentions
        )

        if self.in_corpus_scope_mentioned != expected_in_corpus:
            raise ValueError(
                "in_corpus_scope_mentioned is inconsistent with mentions."
            )

        if (
            self.known_out_of_corpus_scope_mentioned
            != expected_out_of_corpus
        ):
            raise ValueError(
                "known_out_of_corpus_scope_mentioned is inconsistent "
                "with mentions."
            )

        for label, values in (
            ("requested_author_ids", self.requested_author_ids),
            ("requested_work_ids", self.requested_work_ids),
            (
                "known_out_of_corpus_entity_ids",
                self.known_out_of_corpus_entity_ids,
            ),
            ("requested_version_ids", self.requested_version_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique.")

        if (
            not isinstance(self.evaluated_hit_count, int)
            or isinstance(self.evaluated_hit_count, bool)
            or self.evaluated_hit_count < 0
            or self.evaluated_hit_count > 10
        ):
            raise ValueError(
                "evaluated_hit_count must be an integer from 0 to 10."
            )

        if (
            len(self.retrieved_version_ids_at_10)
            != self.evaluated_hit_count
        ):
            raise ValueError(
                "retrieved_version_ids_at_10 must match evaluated depth."
            )

        if (
            self.matching_hit_count_at_10 < 0
            or self.matching_hit_count_at_10 > self.evaluated_hit_count
        ):
            raise ValueError(
                "matching_hit_count_at_10 exceeds evaluated depth."
            )

        if self.requested_version_ids:
            expected_rate = (
                self.matching_hit_count_at_10
                / self.evaluated_hit_count
                if self.evaluated_hit_count
                else 0.0
            )

            if self.matching_hit_rate_at_10 is None or not math.isclose(
                self.matching_hit_rate_at_10,
                expected_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "matching_hit_rate_at_10 is inconsistent."
                )
        elif (
            self.matching_hit_rate_at_10 is not None
            or self.matching_hit_count_at_10 != 0
        ):
            raise ValueError(
                "Matching metrics require requested_version_ids."
            )

        expected_top_hit = (
            self.retrieved_version_ids_at_10[0]
            if self.retrieved_version_ids_at_10
            else None
        )

        if self.top_hit_version_id != expected_top_hit:
            raise ValueError(
                "top_hit_version_id does not match the retrieved ranking."
            )

        if self.requested_version_ids and self.top_hit_version_id is not None:
            expected_match: bool | None = (
                self.top_hit_version_id
                in self.requested_version_ids
            )
        else:
            expected_match = None

        if self.top_hit_matches_requested_scope != expected_match:
            raise ValueError(
                "top_hit_matches_requested_scope is inconsistent."
            )

        expected_conflict = expected_match is False

        if self.source_attribution_conflict != expected_conflict:
            raise ValueError(
                "source_attribution_conflict is inconsistent."
            )


__all__ = [
    "SCOPE_CATALOG_SCHEMA_VERSION",
    "ScopeCatalogEntity",
    "ScopeEntityKind",
    "ScopeEvidence",
    "ScopeMention",
]
