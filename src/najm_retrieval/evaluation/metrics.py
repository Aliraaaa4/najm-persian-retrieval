"""Metrics for passage-level retrieval evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, log2
from statistics import mean, median
from typing import Any

from najm_retrieval.evaluation.models import (
    EvaluationSet,
    RelevanceJudgment,
)


DEFAULT_CUTOFFS = (
    1,
    5,
    10,
)


@dataclass(frozen=True)
class QueryCutoffMetrics:
    """Metrics for one query at one rank cutoff."""

    k: int

    recall: float | None = None
    reciprocal_rank: float | None = None
    ndcg: float | None = None
    negative_hit: float | None = None

    def __post_init__(self) -> None:
        """Validate cutoff and score ranges."""

        if (
            not isinstance(self.k, int)
            or isinstance(self.k, bool)
            or self.k < 1
        ):
            raise ValueError(
                "k must be a positive integer."
            )

        for field_name, value in (
            ("recall", self.recall),
            (
                "reciprocal_rank",
                self.reciprocal_rank,
            ),
            ("ndcg", self.ndcg),
            (
                "negative_hit",
                self.negative_hit,
            ),
        ):
            if value is None:
                continue

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be "
                    "between 0 and 1."
                )


@dataclass(frozen=True)
class QueryEvaluationResult:
    """Evaluation result for one query."""

    query_id: str
    is_answerable: bool
    retrieved_count: int

    cutoff_metrics: tuple[
        QueryCutoffMetrics,
        ...,
    ]

    latency_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate query-level metrics."""

        if not self.query_id.strip():
            raise ValueError(
                "query_id must not be empty."
            )

        if self.retrieved_count < 0:
            raise ValueError(
                "retrieved_count must not "
                "be negative."
            )

        if (
            self.latency_ms is not None
            and self.latency_ms < 0
        ):
            raise ValueError(
                "latency_ms must not be negative."
            )

        cutoffs = tuple(
            metric.k
            for metric in self.cutoff_metrics
        )

        if cutoffs != tuple(
            sorted(set(cutoffs))
        ):
            raise ValueError(
                "Query metric cutoffs must be "
                "unique and sorted."
            )


@dataclass(frozen=True)
class AggregateCutoffMetrics:
    """Mean metrics across evaluation queries."""

    k: int

    mean_recall: float | None = None
    mrr: float | None = None
    mean_ndcg: float | None = None
    negative_hit_rate: float | None = None

    def __post_init__(self) -> None:
        """Validate aggregate score ranges."""

        if (
            not isinstance(self.k, int)
            or isinstance(self.k, bool)
            or self.k < 1
        ):
            raise ValueError(
                "k must be a positive integer."
            )

        for field_name, value in (
            (
                "mean_recall",
                self.mean_recall,
            ),
            ("mrr", self.mrr),
            (
                "mean_ndcg",
                self.mean_ndcg,
            ),
            (
                "negative_hit_rate",
                self.negative_hit_rate,
            ),
        ):
            if value is None:
                continue

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be "
                    "between 0 and 1."
                )


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate and per-query retrieval metrics."""

    query_count: int
    answerable_query_count: int
    out_of_corpus_query_count: int

    cutoffs: tuple[int, ...]

    aggregate_metrics: tuple[
        AggregateCutoffMetrics,
        ...,
    ]

    query_results: tuple[
        QueryEvaluationResult,
        ...,
    ]

    mean_latency_ms: float | None = None
    median_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    max_latency_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate report-level invariants."""

        if self.query_count != len(
            self.query_results
        ):
            raise ValueError(
                "query_count must match "
                "query_results."
            )

        if (
            self.answerable_query_count
            + self.out_of_corpus_query_count
            != self.query_count
        ):
            raise ValueError(
                "Answerable and out-of-corpus "
                "counts must equal query_count."
            )

        if self.cutoffs != tuple(
            sorted(set(self.cutoffs))
        ):
            raise ValueError(
                "Report cutoffs must be "
                "unique and sorted."
            )

        aggregate_cutoffs = tuple(
            metric.k
            for metric in self.aggregate_metrics
        )

        if aggregate_cutoffs != self.cutoffs:
            raise ValueError(
                "Aggregate metric cutoffs must "
                "match report cutoffs."
            )


def recall_at_k(
    judgments: Sequence[
        RelevanceJudgment
    ],
    ranked_passage_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return the fraction of relevant passages found by rank k."""

    _validate_k(k)

    relevant = {
        judgment.passage_id
        for judgment in judgments
    }

    if not relevant:
        return 0.0

    retrieved = set(
        ranked_passage_ids[:k]
    )

    return len(
        relevant & retrieved
    ) / len(relevant)


def reciprocal_rank_at_k(
    judgments: Sequence[
        RelevanceJudgment
    ],
    ranked_passage_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant passage."""

    _validate_k(k)

    relevant = {
        judgment.passage_id
        for judgment in judgments
    }

    for rank, passage_id in enumerate(
        ranked_passage_ids[:k],
        start=1,
    ):
        if passage_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    judgments: Sequence[
        RelevanceJudgment
    ],
    ranked_passage_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return graded normalized discounted cumulative gain."""

    _validate_k(k)

    grades = {
        judgment.passage_id: (
            judgment.grade
        )
        for judgment in judgments
    }

    if not grades:
        return 0.0

    dcg = 0.0

    for rank, passage_id in enumerate(
        ranked_passage_ids[:k],
        start=1,
    ):
        grade = grades.get(
            passage_id,
            0,
        )

        if grade > 0:
            dcg += _gain(
                grade
            ) / log2(rank + 1)

    ideal_grades = sorted(
        grades.values(),
        reverse=True,
    )[:k]

    ideal_dcg = sum(
        _gain(grade)
        / log2(rank + 1)
        for rank, grade in enumerate(
            ideal_grades,
            start=1,
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def negative_hit_at_k(
    ranked_passage_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return one when an out-of-corpus query receives a result."""

    _validate_k(k)

    return (
        1.0
        if ranked_passage_ids[:k]
        else 0.0
    )


def evaluate_rankings(
    evaluation_set: EvaluationSet,
    *,
    rankings: Mapping[
        str,
        Sequence[str],
    ],
    latencies_ms: Mapping[
        str,
        float,
    ] | None = None,
    cutoffs: Sequence[int] = (
        DEFAULT_CUTOFFS
    ),
) -> RetrievalEvaluationReport:
    """Evaluate ranked passage IDs against an evaluation set."""

    normalized_cutoffs = (
        _normalize_cutoffs(
            cutoffs
        )
    )

    query_results: list[
        QueryEvaluationResult
    ] = []

    answerable_count = 0
    out_of_corpus_count = 0

    for query in (
        evaluation_set.metric_queries
    ):
        if query.query_id not in rankings:
            raise ValueError(
                "Missing ranking for evaluation "
                f"query: {query.query_id}"
            )

        ranked_ids = _validate_ranking(
            rankings[query.query_id],
            query_id=query.query_id,
        )

        latency_ms = None

        if latencies_ms is not None:
            if (
                query.query_id
                not in latencies_ms
            ):
                raise ValueError(
                    "Missing latency for evaluation "
                    f"query: {query.query_id}"
                )

            latency_ms = _validate_latency(
                latencies_ms[
                    query.query_id
                ],
                query_id=query.query_id,
            )

        cutoff_results: list[
            QueryCutoffMetrics
        ] = []

        if query.is_answerable:
            answerable_count += 1

            for k in normalized_cutoffs:
                cutoff_results.append(
                    QueryCutoffMetrics(
                        k=k,
                        recall=recall_at_k(
                            query.judgments,
                            ranked_ids,
                            k=k,
                        ),
                        reciprocal_rank=(
                            reciprocal_rank_at_k(
                                query.judgments,
                                ranked_ids,
                                k=k,
                            )
                        ),
                        ndcg=ndcg_at_k(
                            query.judgments,
                            ranked_ids,
                            k=k,
                        ),
                    )
                )

        else:
            out_of_corpus_count += 1

            for k in normalized_cutoffs:
                cutoff_results.append(
                    QueryCutoffMetrics(
                        k=k,
                        negative_hit=(
                            negative_hit_at_k(
                                ranked_ids,
                                k=k,
                            )
                        ),
                    )
                )

        query_results.append(
            QueryEvaluationResult(
                query_id=query.query_id,
                is_answerable=(
                    query.is_answerable
                ),
                retrieved_count=len(
                    ranked_ids
                ),
                cutoff_metrics=tuple(
                    cutoff_results
                ),
                latency_ms=latency_ms,
            )
        )

    aggregates = tuple(
        _aggregate_at_cutoff(
            query_results,
            k=k,
        )
        for k in normalized_cutoffs
    )

    latency_values = [
        result.latency_ms
        for result in query_results
        if result.latency_ms is not None
    ]

    latency_summary = (
        _latency_summary(
            latency_values
        )
    )

    return RetrievalEvaluationReport(
        query_count=len(
            query_results
        ),
        answerable_query_count=(
            answerable_count
        ),
        out_of_corpus_query_count=(
            out_of_corpus_count
        ),
        cutoffs=normalized_cutoffs,
        aggregate_metrics=aggregates,
        query_results=tuple(
            query_results
        ),
        mean_latency_ms=latency_summary[
            "mean"
        ],
        median_latency_ms=(
            latency_summary["median"]
        ),
        p95_latency_ms=latency_summary[
            "p95"
        ],
        max_latency_ms=latency_summary[
            "max"
        ],
    )


def _aggregate_at_cutoff(
    query_results: Sequence[
        QueryEvaluationResult
    ],
    *,
    k: int,
) -> AggregateCutoffMetrics:
    """Aggregate all query metrics at one cutoff."""

    answerable_metrics = [
        _metric_for_k(
            result,
            k=k,
        )
        for result in query_results
        if result.is_answerable
    ]

    negative_metrics = [
        _metric_for_k(
            result,
            k=k,
        )
        for result in query_results
        if not result.is_answerable
    ]

    recalls = [
        metric.recall
        for metric in answerable_metrics
        if metric.recall is not None
    ]

    reciprocal_ranks = [
        metric.reciprocal_rank
        for metric in answerable_metrics
        if (
            metric.reciprocal_rank
            is not None
        )
    ]

    ndcgs = [
        metric.ndcg
        for metric in answerable_metrics
        if metric.ndcg is not None
    ]

    negative_hits = [
        metric.negative_hit
        for metric in negative_metrics
        if (
            metric.negative_hit
            is not None
        )
    ]

    return AggregateCutoffMetrics(
        k=k,
        mean_recall=(
            mean(recalls)
            if recalls
            else None
        ),
        mrr=(
            mean(reciprocal_ranks)
            if reciprocal_ranks
            else None
        ),
        mean_ndcg=(
            mean(ndcgs)
            if ndcgs
            else None
        ),
        negative_hit_rate=(
            mean(negative_hits)
            if negative_hits
            else None
        ),
    )


def _metric_for_k(
    result: QueryEvaluationResult,
    *,
    k: int,
) -> QueryCutoffMetrics:
    """Return one query's metrics at a cutoff."""

    for metric in result.cutoff_metrics:
        if metric.k == k:
            return metric

    raise ValueError(
        f"Query {result.query_id} has no "
        f"metrics for cutoff {k}."
    )


def _normalize_cutoffs(
    cutoffs: Sequence[int],
) -> tuple[int, ...]:
    """Validate, deduplicate, and sort rank cutoffs."""

    normalized: list[int] = []

    for k in cutoffs:
        _validate_k(k)
        normalized.append(k)

    if not normalized:
        raise ValueError(
            "At least one cutoff is required."
        )

    return tuple(
        sorted(set(normalized))
    )


def _validate_k(
    k: Any,
) -> None:
    """Validate one rank cutoff."""

    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or k < 1
    ):
        raise ValueError(
            "k must be a positive integer."
        )


def _validate_ranking(
    ranking: Sequence[str],
    *,
    query_id: str,
) -> tuple[str, ...]:
    """Validate one ranked passage-ID sequence."""

    if isinstance(
        ranking,
        (str, bytes),
    ):
        raise ValueError(
            "Ranking must be a sequence of "
            f"passage IDs for {query_id}."
        )

    values: list[str] = []

    for passage_id in ranking:
        if (
            not isinstance(passage_id, str)
            or not passage_id.strip()
        ):
            raise ValueError(
                "Rankings must contain only "
                f"nonempty strings for {query_id}."
            )

        values.append(passage_id)

    if len(values) != len(
        set(values)
    ):
        raise ValueError(
            "Ranking contains duplicate passage "
            f"IDs for {query_id}."
        )

    return tuple(values)


def _validate_latency(
    value: Any,
    *,
    query_id: str,
) -> float:
    """Validate one query latency."""

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
        or value < 0
    ):
        raise ValueError(
            "Latency must be a nonnegative "
            f"number for {query_id}."
        )

    return float(value)


def _latency_summary(
    values: Sequence[float],
) -> dict[str, float | None]:
    """Return mean, median, nearest-rank p95, and maximum."""

    if not values:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    ordered = sorted(values)

    p95_index = max(
        0,
        ceil(
            0.95 * len(ordered)
        ) - 1,
    )

    return {
        "mean": mean(ordered),
        "median": median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _gain(
    grade: int,
) -> float:
    """Convert a relevance grade to exponential gain."""

    return float(
        (2 ** grade) - 1
    )


__all__ = [
    "DEFAULT_CUTOFFS",
    "AggregateCutoffMetrics",
    "QueryCutoffMetrics",
    "QueryEvaluationResult",
    "RetrievalEvaluationReport",
    "evaluate_rankings",
    "ndcg_at_k",
    "negative_hit_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]