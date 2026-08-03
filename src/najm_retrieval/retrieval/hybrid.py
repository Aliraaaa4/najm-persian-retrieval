"""Weighted reciprocal-rank fusion for lexical and dense search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Protocol

from najm_retrieval.retrieval.dense_models import (
    DenseSearchResult,
)
from najm_retrieval.retrieval.hybrid_models import (
    HybridRetrievalRun,
    HybridSearchHit,
    HybridSearchResult,
)
from najm_retrieval.retrieval.models import (
    LexicalSearchMode,
    LexicalSearchResult,
)


class HybridRetrieverError(RuntimeError):
    """Raised when lexical and dense results cannot be fused."""


class _LexicalSearcher(Protocol):
    def search(
        self,
        query_text: str,
        *,
        limit: int,
        mode: LexicalSearchMode,
    ) -> LexicalSearchResult:
        """Search one lexical index."""


class _DenseSearcher(Protocol):
    def search(
        self,
        query_text: str,
        *,
        limit: int,
    ) -> DenseSearchResult:
        """Search one dense index."""


@dataclass
class _Candidate:
    passage_id: str
    version_id: str
    kind: str
    fusion_score: float = 0.0
    lexical_rank: int | None = None
    dense_rank: int | None = None
    lexical_bm25_score: float | None = None
    dense_cosine_score: float | None = None


class HybridRetriever:
    """Fuse lexical and dense rankings with weighted RRF."""

    def __init__(
        self,
        lexical_index: _LexicalSearcher,
        dense_index: _DenseSearcher,
        *,
        lexical_weight: float = 2.0,
        dense_weight: float = 1.0,
        rrf_constant: float = 60.0,
        candidate_limit: int = 100,
    ) -> None:
        for label, value in (
            ("lexical_weight", lexical_weight),
            ("dense_weight", dense_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{label} must be finite and non-negative."
                )

        if lexical_weight == 0 and dense_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be positive."
            )

        if (
            not math.isfinite(rrf_constant)
            or rrf_constant <= 0
        ):
            raise ValueError(
                "rrf_constant must be finite and positive."
            )

        if not 1 <= candidate_limit <= 100:
            raise ValueError(
                "candidate_limit must be between 1 and 100."
            )

        self.lexical_index = lexical_index
        self.dense_index = dense_index
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)
        self.rrf_constant = float(rrf_constant)
        self.candidate_limit = candidate_limit

    def search(
        self,
        query_text: str,
        *,
        limit: int = 10,
        lexical_mode: LexicalSearchMode = (
            LexicalSearchMode.AUTO
        ),
    ) -> HybridSearchResult:
        """Search both indexes and return a fused ranking."""

        run = self.search_with_components(
            query_text,
            limit=limit,
            lexical_mode=lexical_mode,
        )

        return run.hybrid_result

    def search_with_components(
        self,
        query_text: str,
        *,
        limit: int = 10,
        lexical_mode: LexicalSearchMode = (
            LexicalSearchMode.AUTO
        ),
    ) -> HybridRetrievalRun:
        """Search once and preserve component and fused results."""

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        if not 1 <= limit <= 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        if limit > self.candidate_limit:
            raise ValueError(
                "limit must not exceed candidate_limit."
            )

        if not isinstance(
            lexical_mode,
            LexicalSearchMode,
        ):
            try:
                lexical_mode = LexicalSearchMode(
                    lexical_mode
                )
            except ValueError as error:
                raise ValueError(
                    "Unsupported lexical search mode: "
                    f"{lexical_mode}"
                ) from error

        started_at = perf_counter()

        lexical_result: LexicalSearchResult | None

        try:
            lexical_result = self.lexical_index.search(
                query_text,
                limit=self.candidate_limit,
                mode=lexical_mode,
            )
        except ValueError as error:
            if "no searchable terms" not in str(error):
                raise
            lexical_result = None

        dense_result = self.dense_index.search(
            query_text,
            limit=self.candidate_limit,
        )

        candidates: dict[str, _Candidate] = {}

        if lexical_result is not None:
            self._add_lexical_candidates(
                candidates,
                lexical_result,
            )

        self._add_dense_candidates(
            candidates,
            dense_result,
        )

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.fusion_score,
                (
                    candidate.lexical_rank
                    if candidate.lexical_rank is not None
                    else 10**9
                ),
                (
                    candidate.dense_rank
                    if candidate.dense_rank is not None
                    else 10**9
                ),
                candidate.passage_id,
            ),
        )

        hits = tuple(
            HybridSearchHit(
                passage_id=candidate.passage_id,
                version_id=candidate.version_id,
                kind=candidate.kind,
                rank=rank,
                fusion_score=candidate.fusion_score,
                lexical_rank=candidate.lexical_rank,
                dense_rank=candidate.dense_rank,
                lexical_bm25_score=(
                    candidate.lexical_bm25_score
                ),
                dense_cosine_score=(
                    candidate.dense_cosine_score
                ),
            )
            for rank, candidate in enumerate(
                ordered[:limit],
                start=1,
            )
        )

        latency_ms = (
            perf_counter() - started_at
        ) * 1000.0

        hybrid_result = HybridSearchResult(
            query_text=query_text,
            hits=hits,
            latency_ms=latency_ms,
            lexical_latency_ms=(
                lexical_result.latency_ms
                if lexical_result is not None
                else 0.0
            ),
            dense_latency_ms=dense_result.latency_ms,
            lexical_weight=self.lexical_weight,
            dense_weight=self.dense_weight,
            rrf_constant=self.rrf_constant,
            candidate_limit=self.candidate_limit,
        )

        return HybridRetrievalRun(
            lexical_result=lexical_result,
            dense_result=dense_result,
            hybrid_result=hybrid_result,
        )

    def _add_lexical_candidates(
        self,
        candidates: dict[str, _Candidate],
        result: LexicalSearchResult,
    ) -> None:
        seen: set[str] = set()

        for hit in result.hits:
            if hit.passage_id in seen:
                raise HybridRetrieverError(
                    "Lexical ranking contains duplicate "
                    f"passage ID: {hit.passage_id}"
                )

            seen.add(hit.passage_id)

            candidate = _Candidate(
                passage_id=hit.passage_id,
                version_id=hit.version_id,
                kind=hit.kind,
                lexical_rank=hit.rank,
                lexical_bm25_score=hit.bm25_score,
            )

            candidate.fusion_score = (
                self.lexical_weight
                / (
                    self.rrf_constant
                    + hit.rank
                )
            )

            candidates[
                hit.passage_id
            ] = candidate

    def _add_dense_candidates(
        self,
        candidates: dict[str, _Candidate],
        result: DenseSearchResult,
    ) -> None:
        seen: set[str] = set()

        for hit in result.hits:
            if hit.passage_id in seen:
                raise HybridRetrieverError(
                    "Dense ranking contains duplicate "
                    f"passage ID: {hit.passage_id}"
                )

            seen.add(hit.passage_id)

            candidate = candidates.get(
                hit.passage_id
            )

            if candidate is None:
                candidate = _Candidate(
                    passage_id=hit.passage_id,
                    version_id=hit.version_id,
                    kind=hit.kind,
                )
                candidates[
                    hit.passage_id
                ] = candidate
            elif (
                candidate.version_id != hit.version_id
                or candidate.kind != hit.kind
            ):
                raise HybridRetrieverError(
                    "Lexical and dense metadata disagree "
                    f"for passage ID: {hit.passage_id}"
                )

            candidate.dense_rank = hit.rank
            candidate.dense_cosine_score = (
                hit.cosine_score
            )
            candidate.fusion_score += (
                self.dense_weight
                / (
                    self.rrf_constant
                    + hit.rank
                )
            )


__all__ = [
    "HybridRetriever",
    "HybridRetrieverError",
]
