"""Tests for trusted retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from najm_retrieval.retrieval import (
    AbstentionPolicyConfig,
    AbstentionReason,
    DecisionAction,
    DenseSearchHit,
    DenseSearchResult,
    HybridRetrievalRun,
    HybridSearchHit,
    HybridSearchResult,
    LexicalSearchMode,
    RetrievalDecision,
    RetrievalProfileMismatchError,
    TrustedRetrievalResult,
    TrustedRetriever,
)


@dataclass(frozen=True)
class _Evidence:
    query_text: str


@dataclass(frozen=True)
class _RetrievalEvidence:
    query_text: str
    hybrid_top_1_passage_id: str | None


class _FakeHybridRetriever:
    def __init__(
        self,
        run: HybridRetrievalRun,
        *,
        config: AbstentionPolicyConfig,
        error: Exception | None = None,
    ) -> None:
        profile = config.retrieval_profile

        self.run = run
        self.error = error
        self.lexical_weight = profile.lexical_weight
        self.dense_weight = profile.dense_weight
        self.rrf_constant = profile.rrf_constant
        self.candidate_limit = profile.candidate_limit
        self.calls: list[
            tuple[str, int, LexicalSearchMode]
        ] = []

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

        if self.error is not None:
            raise self.error

        return self.run


class _FakeFeatureExtractor:
    def __init__(self) -> None:
        self.calls: list[
            tuple[object, object, object]
        ] = []

    def extract(
        self,
        *,
        lexical_result: object,
        dense_result: object,
        hybrid_result: HybridSearchResult,
    ) -> _RetrievalEvidence:
        self.calls.append(
            (
                lexical_result,
                dense_result,
                hybrid_result,
            )
        )

        top_passage_id = (
            hybrid_result.hits[0].passage_id
            if hybrid_result.hits
            else None
        )

        return _RetrievalEvidence(
            query_text=hybrid_result.query_text,
            hybrid_top_1_passage_id=top_passage_id,
        )


class _FakeScopeExtractor:
    def __init__(self) -> None:
        self.calls: list[
            tuple[str, HybridSearchResult]
        ] = []

    def extract(
        self,
        *,
        query_text: str,
        hybrid_result: HybridSearchResult,
    ) -> _Evidence:
        self.calls.append(
            (
                query_text,
                hybrid_result,
            )
        )

        return _Evidence(
            query_text=query_text
        )


class _FakeParatextExtractor:
    def __init__(self) -> None:
        self.calls: list[
            HybridSearchResult
        ] = []

    def extract(
        self,
        hybrid_result: HybridSearchResult,
    ) -> _Evidence:
        self.calls.append(
            hybrid_result
        )

        return _Evidence(
            query_text=hybrid_result.query_text
        )


class _FakePolicy:
    def __init__(
        self,
        *,
        return_results: bool,
    ) -> None:
        self.return_results = return_results
        self.calls: list[
            tuple[object, object, object]
        ] = []

    def decide(
        self,
        *,
        retrieval: _RetrievalEvidence,
        scope: _Evidence,
        paratext: _Evidence,
    ) -> RetrievalDecision:
        self.calls.append(
            (
                retrieval,
                scope,
                paratext,
            )
        )

        if self.return_results:
            reason = (
                AbstentionReason.BASELINE_EVIDENCE_PASSED
            )
            action = DecisionAction.RETURN_RESULTS
        else:
            reason = (
                AbstentionReason.WEAK_CROSS_RETRIEVER_EVIDENCE
            )
            action = DecisionAction.ABSTAIN

        return RetrievalDecision(
            query_text=retrieval.query_text,
            action=action,
            reason=reason,
            return_results=self.return_results,
            top_passage_id=(
                retrieval.hybrid_top_1_passage_id
            ),
            triggered_reasons=(reason,),
        )


def _retrieval_run(
    query_text: str = "پرسش",
) -> HybridRetrievalRun:
    dense_hit = DenseSearchHit(
        passage_id="passage-1",
        version_id="version-1",
        kind="mixed_prose",
        rank=1,
        cosine_score=0.9,
    )

    dense_result = DenseSearchResult(
        query_text=query_text,
        model_name="example/model",
        hits=(dense_hit,),
        latency_ms=3.0,
    )

    hybrid_hit = HybridSearchHit(
        passage_id="passage-1",
        version_id="version-1",
        kind="mixed_prose",
        rank=1,
        fusion_score=0.04,
        lexical_rank=None,
        dense_rank=1,
        lexical_bm25_score=None,
        dense_cosine_score=0.9,
    )

    hybrid_result = HybridSearchResult(
        query_text=query_text,
        hits=(hybrid_hit,),
        latency_ms=5.0,
        lexical_latency_ms=0.0,
        dense_latency_ms=3.0,
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=100,
    )

    return HybridRetrievalRun(
        lexical_result=None,
        dense_result=dense_result,
        hybrid_result=hybrid_result,
    )


def _build_trusted(
    *,
    return_results: bool,
    error: Exception | None = None,
) -> tuple[
    TrustedRetriever,
    _FakeHybridRetriever,
    _FakeFeatureExtractor,
    _FakeScopeExtractor,
    _FakeParatextExtractor,
    _FakePolicy,
]:
    config = AbstentionPolicyConfig()
    profile = config.retrieval_profile

    hybrid = _FakeHybridRetriever(
        _retrieval_run(),
        config=config,
        error=error,
    )
    features = _FakeFeatureExtractor()
    scope = _FakeScopeExtractor()
    paratext = _FakeParatextExtractor()
    policy = _FakePolicy(
        return_results=return_results
    )

    trusted = TrustedRetriever(
        hybrid,
        policy_config=config,
        scope_extractor=scope,
        paratext_extractor=paratext,
        corpus_artifact_id=(
            profile.corpus_artifact_id
        ),
        dense_model_name=(
            profile.dense_model_name
        ),
        feature_extractor=features,
        policy=policy,
    )

    return (
        trusted,
        hybrid,
        features,
        scope,
        paratext,
        policy,
    )


def test_search_runs_all_components_once() -> None:
    (
        trusted,
        hybrid,
        features,
        scope,
        paratext,
        policy,
    ) = _build_trusted(
        return_results=True
    )

    result = trusted.search(
        "پرسش"
    )

    assert [
        hit.passage_id
        for hit in result.hits
    ] == ["passage-1"]

    assert hybrid.calls == [
        (
            "پرسش",
            10,
            LexicalSearchMode.AUTO,
        )
    ]

    assert len(features.calls) == 1
    assert len(scope.calls) == 1
    assert len(paratext.calls) == 1
    assert len(policy.calls) == 1


def test_abstention_hides_public_hits() -> None:
    trusted, *_ = _build_trusted(
        return_results=False
    )

    result = trusted.search(
        "پرسش"
    )

    assert result.abstained is True
    assert result.hits == ()
    assert [
        hit.passage_id
        for hit in result.diagnostic_hits
    ] == ["passage-1"]


def test_constructor_rejects_profile_mismatch() -> None:
    config = AbstentionPolicyConfig()
    profile = config.retrieval_profile

    hybrid = _FakeHybridRetriever(
        _retrieval_run(),
        config=config,
    )

    with pytest.raises(
        RetrievalProfileMismatchError,
        match="dense_model_name",
    ):
        TrustedRetriever(
            hybrid,
            policy_config=config,
            scope_extractor=(
                _FakeScopeExtractor()
            ),
            paratext_extractor=(
                _FakeParatextExtractor()
            ),
            corpus_artifact_id=(
                profile.corpus_artifact_id
            ),
            dense_model_name="wrong/model",
        )


def test_retrieval_errors_are_not_abstentions() -> None:
    trusted, _, features, scope, paratext, policy = (
        _build_trusted(
            return_results=False,
            error=RuntimeError(
                "dense index unavailable"
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="dense index unavailable",
    ):
        trusted.search(
            "پرسش"
        )

    assert features.calls == []
    assert scope.calls == []
    assert paratext.calls == []
    assert policy.calls == []


def test_result_rejects_query_misalignment() -> None:
    run = _retrieval_run(
        query_text="پرسش"
    )

    decision = RetrievalDecision(
        query_text="پرسش",
        action=DecisionAction.RETURN_RESULTS,
        reason=(
            AbstentionReason.BASELINE_EVIDENCE_PASSED
        ),
        return_results=True,
        top_passage_id="passage-1",
        triggered_reasons=(
            AbstentionReason.BASELINE_EVIDENCE_PASSED,
        ),
    )

    with pytest.raises(
        ValueError,
        match="same query_text",
    ):
        TrustedRetrievalResult(
            query_text="پرسش متفاوت",
            decision=decision,
            retrieval_run=run,
            retrieval_features=(
                _RetrievalEvidence(
                    query_text="پرسش",
                    hybrid_top_1_passage_id=(
                        "passage-1"
                    ),
                )
            ),
            scope_evidence=(
                _Evidence(
                    query_text="پرسش"
                )
            ),
            paratext_evidence=(
                _Evidence(
                    query_text="پرسش"
                )
            ),
        )
