"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import pytest

from najm_retrieval.evaluation import (
    EvaluationSet,
    QueryType,
    RelevanceJudgment,
    RetrievalQuery,
    evaluate_rankings,
    ndcg_at_k,
    negative_hit_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def _judgments() -> tuple[
    RelevanceJudgment,
    ...,
]:
    return (
        RelevanceJudgment(
            passage_id="p1",
            grade=3,
        ),
        RelevanceJudgment(
            passage_id="p2",
            grade=1,
        ),
    )


def _evaluation_set() -> EvaluationSet:
    return EvaluationSet(
        queries=(
            RetrievalQuery(
                query_id="rq_0001",
                query_text="پرسش اول",
                query_type=(
                    QueryType.SEMANTIC
                ),
                judgments=_judgments(),
            ),
            RetrievalQuery(
                query_id="rq_0002",
                query_text="پرسش دوم",
                query_type=(
                    QueryType.EXACT_QUOTE
                ),
                judgments=(
                    RelevanceJudgment(
                        passage_id="p4",
                        grade=2,
                    ),
                ),
            ),
            RetrievalQuery(
                query_id="rq_0003",
                query_text="پرسش خارج از پیکره",
                query_type=(
                    QueryType.OUT_OF_CORPUS
                ),
            ),
            RetrievalQuery(
                query_id="rq_draft",
                query_text="پرسش برچسب‌نخورده",
                query_type=(
                    QueryType.SEMANTIC
                ),
                include_in_metrics=False,
            ),
        )
    )


def test_recall_at_k_counts_distinct_relevant_passages() -> None:
    """Recall uses all judged relevant passages."""

    ranking = (
        "noise",
        "p2",
        "p1",
    )

    assert recall_at_k(
        _judgments(),
        ranking,
        k=1,
    ) == 0.0

    assert recall_at_k(
        _judgments(),
        ranking,
        k=2,
    ) == 0.5

    assert recall_at_k(
        _judgments(),
        ranking,
        k=3,
    ) == 1.0


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    """MRR contribution is based on first relevant rank."""

    assert reciprocal_rank_at_k(
        _judgments(),
        (
            "noise",
            "p2",
            "p1",
        ),
        k=10,
    ) == 0.5

    assert reciprocal_rank_at_k(
        _judgments(),
        (
            "noise",
            "other",
        ),
        k=10,
    ) == 0.0


def test_ndcg_respects_graded_relevance() -> None:
    """Higher-grade passages receive more ideal gain."""

    ideal = ndcg_at_k(
        _judgments(),
        (
            "p1",
            "p2",
        ),
        k=2,
    )

    reversed_score = ndcg_at_k(
        _judgments(),
        (
            "p2",
            "p1",
        ),
        k=2,
    )

    assert ideal == pytest.approx(
        1.0
    )

    assert 0.0 < reversed_score < 1.0


def test_negative_hit_detects_false_positive_results() -> None:
    """Out-of-corpus retrievals are measured as negative hits."""

    assert negative_hit_at_k(
        (),
        k=10,
    ) == 0.0

    assert negative_hit_at_k(
        (
            "noise",
        ),
        k=10,
    ) == 1.0


def test_evaluate_rankings_aggregates_metrics_and_latency() -> None:
    """Evaluation returns aggregate answerable and negative metrics."""

    report = evaluate_rankings(
        _evaluation_set(),
        rankings={
            "rq_0001": (
                "noise",
                "p2",
                "p1",
            ),
            "rq_0002": (
                "p4",
            ),
            "rq_0003": (),
        },
        latencies_ms={
            "rq_0001": 3.0,
            "rq_0002": 1.0,
            "rq_0003": 2.0,
        },
        cutoffs=(
            1,
            2,
            3,
        ),
    )

    assert report.query_count == 3

    assert (
        report.answerable_query_count
        == 2
    )

    assert (
        report.out_of_corpus_query_count
        == 1
    )

    at_one = report.aggregate_metrics[0]

    assert at_one.k == 1

    assert at_one.mean_recall == (
        pytest.approx(0.5)
    )

    assert at_one.mrr == (
        pytest.approx(0.5)
    )

    assert at_one.mean_ndcg == (
        pytest.approx(0.5)
    )

    assert (
        at_one.negative_hit_rate
        == 0.0
    )

    assert report.mean_latency_ms == (
        pytest.approx(2.0)
    )

    assert report.median_latency_ms == (
        pytest.approx(2.0)
    )

    assert report.p95_latency_ms == (
        pytest.approx(3.0)
    )

    assert report.max_latency_ms == (
        pytest.approx(3.0)
    )


def test_non_metric_query_does_not_require_ranking() -> None:
    """Draft queries are excluded from metric execution."""

    report = evaluate_rankings(
        _evaluation_set(),
        rankings={
            "rq_0001": (
                "p1",
            ),
            "rq_0002": (
                "p4",
            ),
            "rq_0003": (),
        },
    )

    assert tuple(
        result.query_id
        for result in report.query_results
    ) == (
        "rq_0001",
        "rq_0002",
        "rq_0003",
    )


def test_missing_ranking_is_rejected() -> None:
    """Every metric query must have one ranking."""

    with pytest.raises(
        ValueError,
        match="Missing ranking",
    ):
        evaluate_rankings(
            _evaluation_set(),
            rankings={
                "rq_0001": (
                    "p1",
                ),
                "rq_0002": (
                    "p4",
                ),
            },
        )


def test_duplicate_ranked_passage_is_rejected() -> None:
    """One passage cannot occupy multiple ranks."""

    with pytest.raises(
        ValueError,
        match="duplicate passage",
    ):
        evaluate_rankings(
            _evaluation_set(),
            rankings={
                "rq_0001": (
                    "p1",
                    "p1",
                ),
                "rq_0002": (
                    "p4",
                ),
                "rq_0003": (),
            },
        )


def test_invalid_cutoff_is_rejected() -> None:
    """Rank cutoffs must be positive integers."""

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        evaluate_rankings(
            _evaluation_set(),
            rankings={
                "rq_0001": (
                    "p1",
                ),
                "rq_0002": (
                    "p4",
                ),
                "rq_0003": (),
            },
            cutoffs=(
                0,
                10,
            ),
        )