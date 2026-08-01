"""Tests for weighted hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from najm_retrieval.retrieval import (
    DenseSearchHit,
    DenseSearchResult,
    HybridRetriever,
    HybridRetrieverError,
    LexicalSearchMode,
    LexicalSearchResult,
    SearchHit,
)


@dataclass
class _FakeLexicalIndex:
    result: LexicalSearchResult | None = None
    error: Exception | None = None
    calls: list[tuple[str, int, LexicalSearchMode]] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def search(
        self,
        query_text: str,
        *,
        limit: int,
        mode: LexicalSearchMode,
    ) -> LexicalSearchResult:
        assert self.calls is not None
        self.calls.append(
            (query_text, limit, mode)
        )

        if self.error is not None:
            raise self.error

        assert self.result is not None
        return self.result


@dataclass
class _FakeDenseIndex:
    result: DenseSearchResult
    calls: list[tuple[str, int]] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def search(
        self,
        query_text: str,
        *,
        limit: int,
    ) -> DenseSearchResult:
        assert self.calls is not None
        self.calls.append(
            (query_text, limit)
        )
        return self.result


def _lexical_result(
    hits: tuple[SearchHit, ...],
) -> LexicalSearchResult:
    return LexicalSearchResult(
        query_text="پرسش",
        normalized_query="پرسش",
        mode_requested=LexicalSearchMode.AUTO,
        mode_used=LexicalSearchMode.ANY_TERMS,
        hits=hits,
        latency_ms=2.0,
    )


def _dense_result(
    hits: tuple[DenseSearchHit, ...],
) -> DenseSearchResult:
    return DenseSearchResult(
        query_text="پرسش",
        model_name="example/model",
        hits=hits,
        latency_ms=3.0,
    )


def _lexical_hit(
    passage_id: str,
    rank: int,
    *,
    version_id: str = "version",
    kind: str = "mixed_prose",
) -> SearchHit:
    return SearchHit(
        passage_id=passage_id,
        version_id=version_id,
        kind=kind,
        rank=rank,
        bm25_score=float(-rank),
        snippet="snippet",
    )


def _dense_hit(
    passage_id: str,
    rank: int,
    *,
    version_id: str = "version",
    kind: str = "mixed_prose",
) -> DenseSearchHit:
    return DenseSearchHit(
        passage_id=passage_id,
        version_id=version_id,
        kind=kind,
        rank=rank,
        cosine_score=1.0 - rank / 100.0,
    )


def test_weighted_rrf_rewards_overlap() -> None:
    lexical = _FakeLexicalIndex(
        _lexical_result(
            (
                _lexical_hit("a", 1),
                _lexical_hit("b", 2),
            )
        )
    )
    dense = _FakeDenseIndex(
        _dense_result(
            (
                _dense_hit("b", 1),
                _dense_hit("c", 2),
            )
        )
    )

    result = HybridRetriever(
        lexical,
        dense,
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=2,
    ).search(
        "پرسش",
        limit=2,
    )

    assert [
        hit.passage_id
        for hit in result.hits
    ] == ["b", "a"]

    first = result.hits[0]

    assert first.lexical_rank == 2
    assert first.dense_rank == 1
    assert first.fusion_score == pytest.approx(
        2.0 / 62.0 + 1.0 / 61.0
    )


def test_default_configuration_matches_pilot_choice() -> None:
    retriever = HybridRetriever(
        _FakeLexicalIndex(
            _lexical_result(())
        ),
        _FakeDenseIndex(
            _dense_result(())
        ),
    )

    assert retriever.lexical_weight == 2.0
    assert retriever.dense_weight == 1.0
    assert retriever.rrf_constant == 60.0
    assert retriever.candidate_limit == 100


def test_equal_score_prefers_lexical_candidate() -> None:
    lexical = _FakeLexicalIndex(
        _lexical_result(
            (_lexical_hit("lexical", 1),)
        )
    )
    dense = _FakeDenseIndex(
        _dense_result(
            (_dense_hit("dense", 1),)
        )
    )

    result = HybridRetriever(
        lexical,
        dense,
        lexical_weight=1.0,
        dense_weight=1.0,
        candidate_limit=1,
    ).search(
        "پرسش",
        limit=1,
    )

    assert result.hits[0].passage_id == "lexical"


def test_component_scores_and_latencies_are_preserved() -> None:
    lexical = _FakeLexicalIndex(
        _lexical_result(
            (_lexical_hit("shared", 1),)
        )
    )
    dense = _FakeDenseIndex(
        _dense_result(
            (_dense_hit("shared", 1),)
        )
    )

    result = HybridRetriever(
        lexical,
        dense,
        candidate_limit=1,
    ).search(
        "پرسش",
        limit=1,
    )

    hit = result.hits[0]

    assert hit.lexical_bm25_score == -1.0
    assert hit.dense_cosine_score == pytest.approx(
        0.99
    )
    assert result.lexical_latency_ms == 2.0
    assert result.dense_latency_ms == 3.0


def test_candidate_limit_is_used_for_both_indexes() -> None:
    lexical = _FakeLexicalIndex(
        _lexical_result(())
    )
    dense = _FakeDenseIndex(
        _dense_result(())
    )

    HybridRetriever(
        lexical,
        dense,
        candidate_limit=25,
    ).search(
        "پرسش",
        limit=10,
    )

    assert lexical.calls == [
        (
            "پرسش",
            25,
            LexicalSearchMode.AUTO,
        )
    ]
    assert dense.calls == [
        ("پرسش", 25)
    ]


def test_lexical_token_failure_falls_back_to_dense() -> None:
    lexical = _FakeLexicalIndex(
        error=ValueError(
            "Query contains no searchable terms."
        )
    )
    dense = _FakeDenseIndex(
        _dense_result(
            (_dense_hit("dense-only", 1),)
        )
    )

    result = HybridRetriever(
        lexical,
        dense,
        candidate_limit=1,
    ).search(
        "!!!",
        limit=1,
    )

    assert result.hits[0].passage_id == (
        "dense-only"
    )
    assert result.hits[0].lexical_rank is None
    assert result.lexical_latency_ms == 0.0


def test_metadata_disagreement_is_rejected() -> None:
    lexical = _FakeLexicalIndex(
        _lexical_result(
            (
                _lexical_hit(
                    "shared",
                    1,
                    version_id="lexical-version",
                ),
            )
        )
    )
    dense = _FakeDenseIndex(
        _dense_result(
            (
                _dense_hit(
                    "shared",
                    1,
                    version_id="dense-version",
                ),
            )
        )
    )

    with pytest.raises(
        HybridRetrieverError,
        match="metadata disagree",
    ):
        HybridRetriever(
            lexical,
            dense,
            candidate_limit=1,
        ).search(
            "پرسش",
            limit=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "lexical_weight": -1.0,
            },
            "lexical_weight",
        ),
        (
            {
                "dense_weight": -1.0,
            },
            "dense_weight",
        ),
        (
            {
                "lexical_weight": 0.0,
                "dense_weight": 0.0,
            },
            "At least one",
        ),
        (
            {
                "rrf_constant": 0.0,
            },
            "rrf_constant",
        ),
        (
            {
                "candidate_limit": 0,
            },
            "candidate_limit",
        ),
        (
            {
                "candidate_limit": 101,
            },
            "candidate_limit",
        ),
    ],
)
def test_constructor_validates_configuration(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        HybridRetriever(
            _FakeLexicalIndex(
                _lexical_result(())
            ),
            _FakeDenseIndex(
                _dense_result(())
            ),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("query_text", "limit", "message"),
    [
        ("   ", 10, "must not be empty"),
        ("پرسش", 0, "between 1 and 100"),
        ("پرسش", 101, "between 1 and 100"),
        ("پرسش", 11, "candidate_limit"),
    ],
)
def test_search_validates_input(
    query_text: str,
    limit: int,
    message: str,
) -> None:
    retriever = HybridRetriever(
        _FakeLexicalIndex(
            _lexical_result(())
        ),
        _FakeDenseIndex(
            _dense_result(())
        ),
        candidate_limit=10,
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        retriever.search(
            query_text,
            limit=limit,
        )
