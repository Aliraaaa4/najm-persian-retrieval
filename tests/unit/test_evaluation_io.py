"""Tests for retrieval evaluation models and JSONL I/O."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from najm_retrieval.evaluation import (
    EvaluationDataError,
    EvaluationSet,
    QueryType,
    RelevanceJudgment,
    RetrievalQuery,
    load_evaluation_jsonl,
    write_evaluation_jsonl,
)


def _make_evaluation_set() -> EvaluationSet:
    exact = RetrievalQuery(
        query_id="rq_0001",
        query_text=(
            "بشنو این نی چون شکایت می‌کند"
        ),
        query_type=(
            QueryType.EXACT_QUOTE
        ),
        judgments=(
            RelevanceJudgment(
                passage_id=(
                    "0672JalalDinRumi."
                    "Mathnawi.PDL00048-per1:"
                    "passage_000001"
                ),
                grade=3,
                rationale=(
                    "The quoted verse occurs in "
                    "this passage."
                ),
            ),
        ),
        expected_version_ids=(
            "0672JalalDinRumi."
            "Mathnawi.PDL00048-per1",
        ),
        tags=(
            "poetry",
            "rumi",
        ),
    )

    unanswerable = RetrievalQuery(
        query_id="rq_0002",
        query_text=(
            "رساله‌ای درباره شبکه عصبی مدرن"
        ),
        query_type=(
            QueryType.OUT_OF_CORPUS
        ),
        judgments=(),
        tags=(
            "negative",
        ),
    )

    return EvaluationSet(
        queries=(
            exact,
            unanswerable,
        )
    )


def test_evaluation_jsonl_round_trip_preserves_persian(
    tmp_path: Path,
) -> None:
    """Persian text and graded labels survive JSONL round trip."""

    evaluation_set = (
        _make_evaluation_set()
    )

    path = (
        tmp_path
        / "queries.jsonl"
    )

    write_evaluation_jsonl(
        evaluation_set,
        path=path,
    )

    text = path.read_text(
        encoding="utf-8"
    )

    assert "بشنو این نی" in text
    assert "\\u0628" not in text

    loaded = load_evaluation_jsonl(
        path
    )

    assert loaded == evaluation_set

    assert len(
        loaded.answerable_queries
    ) == 1

    assert len(
        loaded.out_of_corpus_queries
    ) == 1


def test_evaluation_jsonl_write_is_deterministic(
    tmp_path: Path,
) -> None:
    """Repeated writes produce identical bytes."""

    evaluation_set = (
        _make_evaluation_set()
    )

    path = (
        tmp_path
        / "queries.jsonl"
    )

    write_evaluation_jsonl(
        evaluation_set,
        path=path,
    )

    first = path.read_bytes()

    path.write_text(
        "corrupted",
        encoding="utf-8",
    )

    write_evaluation_jsonl(
        evaluation_set,
        path=path,
    )

    assert path.read_bytes() == first


def test_duplicate_query_ids_are_rejected(
    tmp_path: Path,
) -> None:
    """Collection-level duplicate IDs are invalid."""

    record = {
        "schema_version": "1.0.0",
        "query_id": "rq_duplicate",
        "query_text": "متن پرسش",
        "query_type": "semantic",
        "judgments": [
            {
                "passage_id": (
                    "book:passage_000001"
                ),
                "grade": 2,
                "rationale": None,
            }
        ],
        "expected_version_ids": [],
        "tags": [],
        "include_in_metrics": True,
        "notes": None,
    }

    path = (
        tmp_path
        / "queries.jsonl"
    )

    serialized = json.dumps(
        record,
        ensure_ascii=False,
    )

    path.write_text(
        serialized
        + "\n"
        + serialized
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="query IDs must be unique",
    ):
        load_evaluation_jsonl(
            path
        )


def test_answerable_metric_query_requires_judgment() -> None:
    """Unlabeled drafts cannot enter metric aggregation."""

    with pytest.raises(
        ValueError,
        match="at least one relevance judgment",
    ):
        RetrievalQuery(
            query_id="rq_unlabeled",
            query_text="اخلاق چیست",
            query_type=(
                QueryType.SEMANTIC
            ),
            judgments=(),
            include_in_metrics=True,
        )


def test_out_of_corpus_query_rejects_judgment() -> None:
    """Negative queries must not identify a relevant passage."""

    with pytest.raises(
        ValueError,
        match="cannot have relevant",
    ):
        RetrievalQuery(
            query_id="rq_negative",
            query_text=(
                "مباحث کامپیوتر کوانتومی"
            ),
            query_type=(
                QueryType.OUT_OF_CORPUS
            ),
            judgments=(
                RelevanceJudgment(
                    passage_id=(
                        "book:passage_000001"
                    ),
                ),
            ),
        )


def test_invalid_json_reports_line_number(
    tmp_path: Path,
) -> None:
    """Malformed JSONL identifies its failing line."""

    path = (
        tmp_path
        / "queries.jsonl"
    )

    path.write_text(
        '{"query_id":"valid-looking"}\n'
        '{"broken":\n',
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="line 1|line 2",
    ):
        load_evaluation_jsonl(
            path
        )