"""Integration tests for trusted retrieval with real evidence components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from najm_retrieval.retrieval import (
    AbstentionReason,
    ContentRole,
    CorpusScopeCatalog,
    DecisionAction,
    DenseSearchHit,
    DenseSearchResult,
    HybridRetrievalRun,
    HybridSearchHit,
    HybridSearchResult,
    LexicalSearchMode,
    LexicalSearchResult,
    ParatextCatalog,
    ParatextEvidenceExtractor,
    ScopeEvidenceExtractor,
    SearchHit,
    TrustedRetriever,
    load_abstention_policy_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "config"
    / "abstention_policy.yaml"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "config"
    / "corpus_manifest.yaml"
)

ALIASES_PATH = (
    PROJECT_ROOT
    / "config"
    / "scope_aliases.yaml"
)

PARATEXT_PATH = (
    PROJECT_ROOT
    / "config"
    / "paratext_zones.yaml"
)

MAJALIS_VERSION = (
    "0672JalalDinRumi."
    "MajalisSabica."
    "AOCP202502141236-per1"
)


@dataclass
class _StaticHybridRetriever:
    """Return one predefined retrieval run without using real indexes."""

    run: HybridRetrievalRun
    lexical_weight: float
    dense_weight: float
    rrf_constant: float
    candidate_limit: int
    calls: list[
        tuple[str, int, LexicalSearchMode]
    ] = field(
        default_factory=list
    )

    def search_with_components(
        self,
        query_text: str,
        *,
        limit: int,
        lexical_mode: LexicalSearchMode,
    ) -> HybridRetrievalRun:
        self.calls.append(
            (
                query_text,
                limit,
                lexical_mode,
            )
        )

        if (
            query_text
            != self.run.hybrid_result.query_text
        ):
            raise ValueError(
                "Static retrieval query does not match "
                "the prepared retrieval run."
            )

        return self.run


def _passage_id(
    ordinal: int,
) -> str:
    return (
        f"{MAJALIS_VERSION}:"
        f"passage_{ordinal:06d}"
    )


def _shared_run(
    query_text: str,
    *,
    ordinal: int,
    dense_score: float = 0.90,
) -> HybridRetrievalRun:
    """Create one strongly supported shared lexical/dense result."""

    passage_id = _passage_id(
        ordinal
    )

    lexical_hit = SearchHit(
        passage_id=passage_id,
        version_id=MAJALIS_VERSION,
        kind="mixed_prose",
        rank=1,
        bm25_score=-1.0,
        snippet="نمونه متن بازیابی‌شده",
    )

    lexical_result = LexicalSearchResult(
        query_text=query_text,
        normalized_query=query_text,
        mode_requested=LexicalSearchMode.AUTO,
        mode_used=LexicalSearchMode.ANY_TERMS,
        hits=(lexical_hit,),
        latency_ms=1.0,
    )

    dense_hit = DenseSearchHit(
        passage_id=passage_id,
        version_id=MAJALIS_VERSION,
        kind="mixed_prose",
        rank=1,
        cosine_score=dense_score,
    )

    dense_result = DenseSearchResult(
        query_text=query_text,
        model_name=(
            "intfloat/multilingual-e5-small"
        ),
        hits=(dense_hit,),
        latency_ms=2.0,
    )

    hybrid_hit = HybridSearchHit(
        passage_id=passage_id,
        version_id=MAJALIS_VERSION,
        kind="mixed_prose",
        rank=1,
        fusion_score=3.0 / 61.0,
        lexical_rank=1,
        dense_rank=1,
        lexical_bm25_score=-1.0,
        dense_cosine_score=dense_score,
    )

    hybrid_result = HybridSearchResult(
        query_text=query_text,
        hits=(hybrid_hit,),
        latency_ms=3.0,
        lexical_latency_ms=1.0,
        dense_latency_ms=2.0,
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=100,
    )

    return HybridRetrievalRun(
        lexical_result=lexical_result,
        dense_result=dense_result,
        hybrid_result=hybrid_result,
    )


def _weak_cross_retriever_run(
    query_text: str,
) -> HybridRetrievalRun:
    """Create low dense evidence with no lexical/dense overlap."""

    lexical_passage_id = _passage_id(
        30
    )
    dense_passage_id = _passage_id(
        35
    )

    lexical_result = LexicalSearchResult(
        query_text=query_text,
        normalized_query=query_text,
        mode_requested=LexicalSearchMode.AUTO,
        mode_used=LexicalSearchMode.ANY_TERMS,
        hits=(
            SearchHit(
                passage_id=lexical_passage_id,
                version_id=MAJALIS_VERSION,
                kind="mixed_prose",
                rank=1,
                bm25_score=-1.0,
                snippet="نتیجه واژگانی",
            ),
        ),
        latency_ms=1.0,
    )

    dense_result = DenseSearchResult(
        query_text=query_text,
        model_name=(
            "intfloat/multilingual-e5-small"
        ),
        hits=(
            DenseSearchHit(
                passage_id=dense_passage_id,
                version_id=MAJALIS_VERSION,
                kind="mixed_prose",
                rank=1,
                cosine_score=0.80,
            ),
        ),
        latency_ms=2.0,
    )

    hybrid_result = HybridSearchResult(
        query_text=query_text,
        hits=(
            HybridSearchHit(
                passage_id=lexical_passage_id,
                version_id=MAJALIS_VERSION,
                kind="mixed_prose",
                rank=1,
                fusion_score=2.0 / 61.0,
                lexical_rank=1,
                dense_rank=None,
                lexical_bm25_score=-1.0,
                dense_cosine_score=None,
            ),
            HybridSearchHit(
                passage_id=dense_passage_id,
                version_id=MAJALIS_VERSION,
                kind="mixed_prose",
                rank=2,
                fusion_score=1.0 / 61.0,
                lexical_rank=None,
                dense_rank=1,
                lexical_bm25_score=None,
                dense_cosine_score=0.80,
            ),
        ),
        latency_ms=3.0,
        lexical_latency_ms=1.0,
        dense_latency_ms=2.0,
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=100,
    )

    return HybridRetrievalRun(
        lexical_result=lexical_result,
        dense_result=dense_result,
        hybrid_result=hybrid_result,
    )


def _build_trusted(
    run: HybridRetrievalRun,
) -> tuple[
    TrustedRetriever,
    _StaticHybridRetriever,
]:
    policy_config = (
        load_abstention_policy_config(
            POLICY_PATH
        )
    )

    profile = (
        policy_config.retrieval_profile
    )

    hybrid_retriever = (
        _StaticHybridRetriever(
            run=run,
            lexical_weight=(
                profile.lexical_weight
            ),
            dense_weight=(
                profile.dense_weight
            ),
            rrf_constant=(
                profile.rrf_constant
            ),
            candidate_limit=(
                profile.candidate_limit
            ),
        )
    )

    scope_catalog = (
        CorpusScopeCatalog.from_files(
            manifest_path=MANIFEST_PATH,
            aliases_path=ALIASES_PATH,
        )
    )

    paratext_catalog = (
        ParatextCatalog.from_yaml(
            PARATEXT_PATH
        )
    )

    trusted_retriever = TrustedRetriever(
        hybrid_retriever,
        policy_config=policy_config,
        scope_extractor=(
            ScopeEvidenceExtractor(
                scope_catalog
            )
        ),
        paratext_extractor=(
            ParatextEvidenceExtractor(
                paratext_catalog
            )
        ),
        corpus_artifact_id=(
            profile.corpus_artifact_id
        ),
        dense_model_name=(
            profile.dense_model_name
        ),
    )

    return (
        trusted_retriever,
        hybrid_retriever,
    )


def test_real_components_return_strong_authorial_result() -> None:
    query = (
        "این متن درباره فساد امت چه می‌گوید؟"
    )

    trusted, hybrid = _build_trusted(
        _shared_run(
            query,
            ordinal=30,
        )
    )

    result = trusted.search(
        query
    )

    assert (
        result.decision.action
        is DecisionAction.RETURN_RESULTS
    )
    assert (
        result.decision.reason
        is AbstentionReason.BASELINE_EVIDENCE_PASSED
    )
    assert not result.abstained
    assert len(result.hits) == 1

    assert (
        result.retrieval_features.overlap_at_10
        == 1
    )
    assert (
        result.paratext_evidence.top_hit_role
        is ContentRole.AUTHORIAL
    )

    assert hybrid.calls == [
        (
            query,
            10,
            LexicalSearchMode.AUTO,
        )
    ]


def test_real_scope_evidence_rejects_hafez_query() -> None:
    query = (
        "حافظ درباره رندی چه گفته است؟"
    )

    trusted, _ = _build_trusted(
        _shared_run(
            query,
            ordinal=30,
        )
    )

    result = trusted.search(
        query
    )

    assert result.abstained
    assert result.hits == ()
    assert len(result.diagnostic_hits) == 1

    assert (
        result.decision.reason
        is AbstentionReason.KNOWN_OUT_OF_CORPUS_SCOPE
    )
    assert (
        result.scope_evidence
        .known_out_of_corpus_scope_mentioned
    )


def test_real_paratext_evidence_rejects_front_matter() -> None:
    query = (
        "این بخش آغازین درباره چیست؟"
    )

    trusted, _ = _build_trusted(
        _shared_run(
            query,
            ordinal=13,
        )
    )

    result = trusted.search(
        query
    )

    assert result.abstained
    assert result.hits == ()

    assert (
        result.decision.reason
        is AbstentionReason.TOP_HIT_PARATEXT
    )
    assert (
        result.paratext_evidence.top_hit_role
        is ContentRole.PARATEXT
    )


def test_real_policy_rejects_weak_cross_retriever_evidence() -> None:
    query = (
        "این عبارت درباره چه موضوعی است؟"
    )

    trusted, _ = _build_trusted(
        _weak_cross_retriever_run(
            query
        )
    )

    result = trusted.search(
        query
    )

    assert result.abstained
    assert result.hits == ()

    assert (
        result.decision.reason
        is AbstentionReason.WEAK_CROSS_RETRIEVER_EVIDENCE
    )
    assert (
        result.retrieval_features.overlap_at_10
        == 0
    )
    assert (
        result.retrieval_features.dense_top_1_score
        == pytest.approx(
            0.80
        )
    )
