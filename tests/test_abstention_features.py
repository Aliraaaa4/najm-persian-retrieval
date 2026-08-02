"""Tests for abstention feature extraction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from najm_retrieval.retrieval import (
    ABSTENTION_FEATURE_SCHEMA_VERSION,
    AbstentionFeatureExtractor,
    AbstentionFeatures,
    DenseSearchHit,
    DenseSearchResult,
    HybridSearchHit,
    HybridSearchResult,
    LexicalSearchMode,
    LexicalSearchResult,
    SearchHit,
)


QUERY = "نمونه پرسش"


def lexical_hit(
    passage_id: str,
    *,
    rank: int,
    score: float,
) -> SearchHit:
    return SearchHit(
        passage_id=passage_id,
        version_id="version-1",
        kind="mixed_prose",
        rank=rank,
        bm25_score=score,
        snippet="نمونه",
    )


def dense_hit(
    passage_id: str,
    *,
    rank: int,
    score: float,
) -> DenseSearchHit:
    return DenseSearchHit(
        passage_id=passage_id,
        version_id="version-1",
        kind="mixed_prose",
        rank=rank,
        cosine_score=score,
    )


def hybrid_hit(
    passage_id: str,
    *,
    rank: int,
    score: float,
    lexical_rank: int | None,
    dense_rank: int | None,
    lexical_score: float | None,
    dense_score: float | None,
) -> HybridSearchHit:
    return HybridSearchHit(
        passage_id=passage_id,
        version_id="version-1",
        kind="mixed_prose",
        rank=rank,
        fusion_score=score,
        lexical_rank=lexical_rank,
        dense_rank=dense_rank,
        lexical_bm25_score=lexical_score,
        dense_cosine_score=dense_score,
    )


def lexical_result(
    *hits: SearchHit,
    query_text: str = QUERY,
) -> LexicalSearchResult:
    return LexicalSearchResult(
        query_text=query_text,
        normalized_query=query_text,
        mode_requested=LexicalSearchMode.AUTO,
        mode_used=LexicalSearchMode.ALL_TERMS,
        hits=tuple(hits),
        latency_ms=1.0,
    )


def dense_result(
    *hits: DenseSearchHit,
    query_text: str = QUERY,
) -> DenseSearchResult:
    return DenseSearchResult(
        query_text=query_text,
        model_name="test-model",
        hits=tuple(hits),
        latency_ms=2.0,
    )


def hybrid_result(
    *hits: HybridSearchHit,
    query_text: str = QUERY,
) -> HybridSearchResult:
    return HybridSearchResult(
        query_text=query_text,
        hits=tuple(hits),
        latency_ms=3.0,
        lexical_latency_ms=1.0,
        dense_latency_ms=2.0,
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=100,
    )


def complete_results() -> tuple[
    LexicalSearchResult,
    DenseSearchResult,
    HybridSearchResult,
]:
    lexical = lexical_result(
        lexical_hit("p1", rank=1, score=-8.0),
        lexical_hit("p2", rank=2, score=-7.0),
        lexical_hit("p4", rank=3, score=-6.0),
    )

    dense = dense_result(
        dense_hit("p2", rank=1, score=0.90),
        dense_hit("p1", rank=2, score=0.85),
        dense_hit("p3", rank=3, score=0.80),
    )

    hybrid = hybrid_result(
        hybrid_hit(
            "p1",
            rank=1,
            score=0.048,
            lexical_rank=1,
            dense_rank=2,
            lexical_score=-8.0,
            dense_score=0.85,
        ),
        hybrid_hit(
            "p2",
            rank=2,
            score=0.047,
            lexical_rank=2,
            dense_rank=1,
            lexical_score=-7.0,
            dense_score=0.90,
        ),
        hybrid_hit(
            "p3",
            rank=3,
            score=0.015,
            lexical_rank=None,
            dense_rank=3,
            lexical_score=None,
            dense_score=0.80,
        ),
    )

    return lexical, dense, hybrid


def test_extracts_interpretable_features() -> None:
    lexical, dense, hybrid = complete_results()

    features = AbstentionFeatureExtractor().extract(
        lexical_result=lexical,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    assert features.query_text == QUERY
    assert features.lexical_result_available
    assert (
        features.lexical_mode_used
        is LexicalSearchMode.ALL_TERMS
    )
    assert features.lexical_hit_count == 3
    assert features.dense_hit_count == 3
    assert features.hybrid_hit_count == 3
    assert features.lexical_top_1_passage_id == "p1"
    assert features.dense_top_1_passage_id == "p2"
    assert features.hybrid_top_1_passage_id == "p1"
    assert features.lexical_top_1_bm25 == -8.0
    assert features.dense_top_1_score == 0.90
    assert features.dense_top_2_score == 0.85
    assert features.dense_margin_1_2 == pytest.approx(0.05)
    assert features.overlap_at_10 == 2
    assert features.overlap_at_100 == 2
    assert not features.top_1_same_passage
    assert features.hybrid_top_1_score == 0.048
    assert features.hybrid_top_2_score == 0.047
    assert features.hybrid_margin_1_2 == pytest.approx(0.001)
    assert features.hybrid_top_1_lexical_rank == 1
    assert features.hybrid_top_1_dense_rank == 2
    assert features.hybrid_top_1_dual_supported
    assert (
        features.schema_version
        == ABSTENTION_FEATURE_SCHEMA_VERSION
    )


def test_supports_dense_only_retrieval() -> None:
    dense = dense_result(
        dense_hit("p1", rank=1, score=0.80)
    )
    hybrid = hybrid_result(
        hybrid_hit(
            "p1",
            rank=1,
            score=0.016,
            lexical_rank=None,
            dense_rank=1,
            lexical_score=None,
            dense_score=0.80,
        )
    )

    features = AbstentionFeatureExtractor().extract(
        lexical_result=None,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    assert not features.lexical_result_available
    assert features.lexical_mode_used is None
    assert features.lexical_hit_count == 0
    assert features.lexical_top_1_passage_id is None
    assert features.lexical_top_1_bm25 is None
    assert features.overlap_at_10 == 0
    assert features.overlap_at_100 == 0
    assert not features.top_1_same_passage
    assert features.hybrid_top_1_lexical_rank is None
    assert features.hybrid_top_1_dense_rank == 1
    assert not features.hybrid_top_1_dual_supported


def test_supports_empty_component_results() -> None:
    features = AbstentionFeatureExtractor().extract(
        lexical_result=lexical_result(),
        dense_result=dense_result(),
        hybrid_result=hybrid_result(),
    )

    assert features.lexical_hit_count == 0
    assert features.dense_hit_count == 0
    assert features.hybrid_hit_count == 0
    assert features.dense_top_1_score is None
    assert features.dense_margin_1_2 is None
    assert features.hybrid_top_1_score is None
    assert features.hybrid_margin_1_2 is None


def test_detects_same_component_top_hit() -> None:
    lexical = lexical_result(
        lexical_hit("p1", rank=1, score=-4.0)
    )
    dense = dense_result(
        dense_hit("p1", rank=1, score=0.90)
    )
    hybrid = hybrid_result(
        hybrid_hit(
            "p1",
            rank=1,
            score=0.049,
            lexical_rank=1,
            dense_rank=1,
            lexical_score=-4.0,
            dense_score=0.90,
        )
    )

    features = AbstentionFeatureExtractor().extract(
        lexical_result=lexical,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    assert features.top_1_same_passage
    assert features.overlap_at_10 == 1
    assert features.overlap_at_100 == 1


def test_rejects_mismatched_query_texts() -> None:
    lexical, dense, hybrid = complete_results()

    with pytest.raises(ValueError, match="same query_text"):
        AbstentionFeatureExtractor().extract(
            lexical_result=lexical,
            dense_result=replace(
                dense,
                query_text="پرسش دیگر",
            ),
            hybrid_result=hybrid,
        )


def test_rejects_duplicate_component_ids() -> None:
    lexical = lexical_result(
        lexical_hit("p1", rank=1, score=-4.0),
        lexical_hit("p1", rank=2, score=-3.0),
    )

    with pytest.raises(
        ValueError,
        match="duplicate passage IDs",
    ):
        AbstentionFeatureExtractor().extract(
            lexical_result=lexical,
            dense_result=dense_result(),
            hybrid_result=hybrid_result(),
        )


def test_rejects_hybrid_lexical_rank_without_result() -> None:
    dense = dense_result(
        dense_hit("p1", rank=1, score=0.90)
    )
    hybrid = hybrid_result(
        hybrid_hit(
            "p1",
            rank=1,
            score=0.049,
            lexical_rank=1,
            dense_rank=1,
            lexical_score=-4.0,
            dense_score=0.90,
        )
    )

    with pytest.raises(
        ValueError,
        match="no lexical result",
    ):
        AbstentionFeatureExtractor().extract(
            lexical_result=None,
            dense_result=dense,
            hybrid_result=hybrid,
        )


def test_rejects_hybrid_rank_beyond_component_depth() -> None:
    lexical = lexical_result(
        lexical_hit("p1", rank=1, score=-4.0)
    )
    dense = dense_result(
        dense_hit("p1", rank=1, score=0.90)
    )
    hybrid = hybrid_result(
        hybrid_hit(
            "p1",
            rank=1,
            score=0.049,
            lexical_rank=2,
            dense_rank=1,
            lexical_score=-4.0,
            dense_score=0.90,
        )
    )

    with pytest.raises(
        ValueError,
        match="lexical rank exceeds",
    ):
        AbstentionFeatureExtractor().extract(
            lexical_result=lexical,
            dense_result=dense,
            hybrid_result=hybrid,
        )


def test_rejects_hybrid_component_identity_mismatch() -> None:
    lexical = lexical_result(
        lexical_hit("p1", rank=1, score=-4.0)
    )
    dense = dense_result(
        dense_hit("p2", rank=1, score=0.90)
    )
    hybrid = hybrid_result(
        hybrid_hit(
            "p1",
            rank=1,
            score=0.049,
            lexical_rank=1,
            dense_rank=1,
            lexical_score=-4.0,
            dense_score=0.90,
        )
    )

    with pytest.raises(
        ValueError,
        match="dense metadata disagree",
    ):
        AbstentionFeatureExtractor().extract(
            lexical_result=lexical,
            dense_result=dense,
            hybrid_result=hybrid,
        )


def test_rejects_hybrid_component_score_mismatch() -> None:
    lexical, dense, hybrid = complete_results()
    bad_hybrid = replace(
        hybrid,
        hits=(
            replace(
                hybrid.hits[0],
                dense_cosine_score=0.70,
            ),
            *hybrid.hits[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="dense score does not match",
    ):
        AbstentionFeatureExtractor().extract(
            lexical_result=lexical,
            dense_result=dense,
            hybrid_result=bad_hybrid,
        )


def test_model_rejects_inconsistent_same_top_flag() -> None:
    lexical, dense, hybrid = complete_results()
    features = AbstentionFeatureExtractor().extract(
        lexical_result=lexical,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    with pytest.raises(
        ValueError,
        match="top_1_same_passage",
    ):
        replace(
            features,
            top_1_same_passage=True,
        )


def test_model_rejects_inconsistent_dual_support() -> None:
    lexical, dense, hybrid = complete_results()
    features = AbstentionFeatureExtractor().extract(
        lexical_result=lexical,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    with pytest.raises(
        ValueError,
        match="dual_supported",
    ):
        replace(
            features,
            hybrid_top_1_dual_supported=False,
        )


def test_model_rejects_overlap_beyond_depth() -> None:
    lexical, dense, hybrid = complete_results()
    features = AbstentionFeatureExtractor().extract(
        lexical_result=lexical,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    with pytest.raises(
        ValueError,
        match="overlap_at_10 exceeds",
    ):
        replace(
            features,
            overlap_at_10=4,
            overlap_at_100=4,
        )


def test_model_rejects_unavailable_lexical_evidence() -> None:
    dense = dense_result(
        dense_hit("p1", rank=1, score=0.80)
    )
    hybrid = hybrid_result(
        hybrid_hit(
            "p1",
            rank=1,
            score=0.016,
            lexical_rank=None,
            dense_rank=1,
            lexical_score=None,
            dense_score=0.80,
        )
    )
    features = AbstentionFeatureExtractor().extract(
        lexical_result=None,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    with pytest.raises(
        ValueError,
        match="Unavailable lexical",
    ):
        replace(
            features,
            lexical_top_1_passage_id="p1",
            lexical_top_1_bm25=-4.0,
            lexical_hit_count=1,
        )


def test_features_are_frozen() -> None:
    lexical, dense, hybrid = complete_results()
    features = AbstentionFeatureExtractor().extract(
        lexical_result=lexical,
        dense_result=dense,
        hybrid_result=hybrid,
    )

    with pytest.raises(AttributeError):
        features.overlap_at_10 = 99  # type: ignore[misc]


def test_public_model_can_be_imported() -> None:
    assert AbstentionFeatures.__name__ == "AbstentionFeatures"
