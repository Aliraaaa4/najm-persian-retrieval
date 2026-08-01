"""Models for lexical and semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


LEXICAL_INDEX_SCHEMA_VERSION = "1.0.0"


class LexicalSearchMode(str, Enum):
    """Supported SQLite FTS5 query strategies."""

    AUTO = "auto"
    PHRASE = "phrase"
    ALL_TERMS = "all_terms"
    ANY_TERMS = "any_terms"


@dataclass(frozen=True)
class SearchHit:
    """One ranked retrieval result."""

    passage_id: str
    version_id: str
    kind: str

    rank: int
    bm25_score: float
    snippet: str

    def __post_init__(self) -> None:
        """Validate one search hit."""

        for label, value in (
            ("passage_id", self.passage_id),
            ("version_id", self.version_id),
            ("kind", self.kind),
        ):
            if not value.strip():
                raise ValueError(
                    f"{label} must not be empty."
                )

        if self.rank < 1:
            raise ValueError(
                "rank must be at least 1."
            )


@dataclass(frozen=True)
class LexicalSearchResult:
    """Complete result of one lexical query."""

    query_text: str
    normalized_query: str
    mode_requested: LexicalSearchMode
    mode_used: LexicalSearchMode

    hits: tuple[SearchHit, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        """Validate result metadata."""

        if not self.query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        if not self.normalized_query.strip():
            raise ValueError(
                "normalized_query must not be empty."
            )

        if self.latency_ms < 0:
            raise ValueError(
                "latency_ms must not be negative."
            )

        expected_ranks = tuple(
            range(
                1,
                len(self.hits) + 1,
            )
        )

        actual_ranks = tuple(
            hit.rank
            for hit in self.hits
        )

        if actual_ranks != expected_ranks:
            raise ValueError(
                "Search-hit ranks must be "
                "contiguous and start at 1."
            )


@dataclass(frozen=True)
class LexicalIndexBuildReport:
    """Summary of one deterministic FTS5 index build."""

    database_path: Path
    source_file_count: int
    passage_count: int
    runtime_seconds: float
    database_bytes: int
    schema_version: str = (
        LEXICAL_INDEX_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        """Validate build statistics."""

        if self.source_file_count < 1:
            raise ValueError(
                "source_file_count must be positive."
            )

        if self.passage_count < 1:
            raise ValueError(
                "passage_count must be positive."
            )

        if self.runtime_seconds < 0:
            raise ValueError(
                "runtime_seconds must not be negative."
            )

        if self.database_bytes < 1:
            raise ValueError(
                "database_bytes must be positive."
            )

        if (
            self.schema_version
            != LEXICAL_INDEX_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported lexical-index "
                f"schema version: "
                f"{self.schema_version}"
            )


__all__ = [
    "LEXICAL_INDEX_SCHEMA_VERSION",
    "LexicalIndexBuildReport",
    "LexicalSearchMode",
    "LexicalSearchResult",
    "SearchHit",
]