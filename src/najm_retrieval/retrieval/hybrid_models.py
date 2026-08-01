"""Models for hybrid lexical and dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math


HYBRID_RETRIEVAL_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class HybridSearchHit:
    """One result produced by rank fusion."""

    passage_id: str
    version_id: str
    kind: str
    rank: int
    fusion_score: float
    lexical_rank: int | None
    dense_rank: int | None
    lexical_bm25_score: float | None
    dense_cosine_score: float | None

    def __post_init__(self) -> None:
        """Validate one hybrid search hit."""

        for label, value in (
            ("passage_id", self.passage_id),
            ("version_id", self.version_id),
            ("kind", self.kind),
        ):
            if not value.strip():
                raise ValueError(f"{label} must not be empty.")

        if self.rank < 1:
            raise ValueError("rank must be at least 1.")

        if (
            not math.isfinite(self.fusion_score)
            or self.fusion_score < 0
        ):
            raise ValueError(
                "fusion_score must be finite and non-negative."
            )

        if self.lexical_rank is None and self.dense_rank is None:
            raise ValueError(
                "At least one component rank must be present."
            )

        for label, value in (
            ("lexical_rank", self.lexical_rank),
            ("dense_rank", self.dense_rank),
        ):
            if value is not None and value < 1:
                raise ValueError(
                    f"{label} must be at least 1 when present."
                )

        for label, value in (
            (
                "lexical_bm25_score",
                self.lexical_bm25_score,
            ),
            (
                "dense_cosine_score",
                self.dense_cosine_score,
            ),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"{label} must be finite when present."
                )


@dataclass(frozen=True)
class HybridSearchResult:
    """Complete result of one hybrid query."""

    query_text: str
    hits: tuple[HybridSearchHit, ...]
    latency_ms: float
    lexical_latency_ms: float
    dense_latency_ms: float
    lexical_weight: float
    dense_weight: float
    rrf_constant: float
    candidate_limit: int

    def __post_init__(self) -> None:
        """Validate hybrid result metadata."""

        if not self.query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        for label, value in (
            ("latency_ms", self.latency_ms),
            (
                "lexical_latency_ms",
                self.lexical_latency_ms,
            ),
            (
                "dense_latency_ms",
                self.dense_latency_ms,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{label} must be finite and non-negative."
                )

        for label, value in (
            ("lexical_weight", self.lexical_weight),
            ("dense_weight", self.dense_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{label} must be finite and non-negative."
                )

        if (
            self.lexical_weight == 0
            and self.dense_weight == 0
        ):
            raise ValueError(
                "At least one retrieval weight must be positive."
            )

        if (
            not math.isfinite(self.rrf_constant)
            or self.rrf_constant <= 0
        ):
            raise ValueError(
                "rrf_constant must be finite and positive."
            )

        if not 1 <= self.candidate_limit <= 100:
            raise ValueError(
                "candidate_limit must be between 1 and 100."
            )

        expected_ranks = tuple(
            range(1, len(self.hits) + 1)
        )
        actual_ranks = tuple(
            hit.rank
            for hit in self.hits
        )

        if actual_ranks != expected_ranks:
            raise ValueError(
                "Hybrid-search ranks must be contiguous "
                "and start at 1."
            )


__all__ = [
    "HYBRID_RETRIEVAL_SCHEMA_VERSION",
    "HybridSearchHit",
    "HybridSearchResult",
]
