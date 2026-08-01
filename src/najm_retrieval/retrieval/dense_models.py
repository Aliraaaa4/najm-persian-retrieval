"""Models for dense semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math


DENSE_INDEX_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class DenseSearchHit:
    """One ranked dense-retrieval result."""

    passage_id: str
    version_id: str
    kind: str
    rank: int
    cosine_score: float

    def __post_init__(self) -> None:
        """Validate one dense search hit."""

        for label, value in (
            ("passage_id", self.passage_id),
            ("version_id", self.version_id),
            ("kind", self.kind),
        ):
            if not value.strip():
                raise ValueError(f"{label} must not be empty.")

        if self.rank < 1:
            raise ValueError("rank must be at least 1.")

        if not math.isfinite(self.cosine_score):
            raise ValueError("cosine_score must be finite.")

        if not -1.0001 <= self.cosine_score <= 1.0001:
            raise ValueError("cosine_score must be between -1 and 1.")


@dataclass(frozen=True)
class DenseSearchResult:
    """Complete result of one dense query."""

    query_text: str
    model_name: str
    hits: tuple[DenseSearchHit, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        """Validate dense result metadata."""

        if not self.query_text.strip():
            raise ValueError("query_text must not be empty.")

        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")

        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative.")

        expected_ranks = tuple(range(1, len(self.hits) + 1))
        actual_ranks = tuple(hit.rank for hit in self.hits)

        if actual_ranks != expected_ranks:
            raise ValueError(
                "Dense-search ranks must be contiguous and start at 1."
            )


__all__ = [
    "DENSE_INDEX_SCHEMA_VERSION",
    "DenseSearchHit",
    "DenseSearchResult",
]
