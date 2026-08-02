"""Tests for the frozen calibrated abstention rule."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from najm_retrieval.retrieval import (
    AbstentionFeatures,
    AbstentionPolicy,
    AbstentionPolicyConfig,
    AbstentionReason,
    ContentRole,
    DecisionAction,
    LexicalSearchMode,
    ParatextEvidence,
    PassageRoleEvidence,
    ScopeEntityKind,
    ScopeEvidence,
    ScopeMention,
    load_abstention_policy_config,
)
from najm_retrieval.retrieval.policy_config import (
    AbstentionPolicyConfigError,
)


QUERY = "پرسش آزمایشی"

TOP_VERSION = (
    "0672JalalDinRumi."
    "MajalisSabica."
    "AOCP202502141236-per1"
)

TOP_PASSAGE_ID = (
    f"{TOP_VERSION}:"
    "passage_000030"
)


def make_retrieval(
    *,
    dense_score: float,
    overlap_at_10: int = 0,
) -> AbstentionFeatures:
    lexical_available = (
        overlap_at_10 > 0
    )

    return AbstentionFeatures(
        query_text=QUERY,
        lexical_result_available=(
            lexical_available
        ),
        lexical_mode_used=(
            LexicalSearchMode.ANY_TERMS
            if lexical_available
            else None
        ),
        lexical_hit_count=(
            1
            if lexical_available
            else 0
        ),
        dense_hit_count=1,
        hybrid_hit_count=1,
        lexical_top_1_passage_id=(
            TOP_PASSAGE_ID
            if lexical_available
            else None
        ),
        dense_top_1_passage_id=(
            TOP_PASSAGE_ID
        ),
        hybrid_top_1_passage_id=(
            TOP_PASSAGE_ID
        ),
        lexical_top_1_bm25=(
            -1.0
            if lexical_available
            else None
        ),
        dense_top_1_score=(
            dense_score
        ),
        dense_top_2_score=None,
        dense_margin_1_2=None,
        overlap_at_10=(
            overlap_at_10
        ),
        overlap_at_100=(
            overlap_at_10
        ),
        top_1_same_passage=(
            lexical_available
        ),
        hybrid_top_1_score=0.03,
        hybrid_top_2_score=None,
        hybrid_margin_1_2=None,
        hybrid_top_1_lexical_rank=(
            1
            if lexical_available
            else None
        ),
        hybrid_top_1_dense_rank=1,
        hybrid_top_1_dual_supported=(
            lexical_available
        ),
    )


def make_scope() -> ScopeEvidence:
    return ScopeEvidence(
        query_text=QUERY,
        mentions=(),
        explicit_scope=False,
        in_corpus_scope_mentioned=False,
        known_out_of_corpus_scope_mentioned=False,
        requested_author_ids=(),
        requested_work_ids=(),
        known_out_of_corpus_entity_ids=(),
        requested_version_ids=(),
        evaluated_hit_count=1,
        retrieved_version_ids_at_10=(
            TOP_VERSION,
        ),
        matching_hit_count_at_10=0,
        matching_hit_rate_at_10=None,
        top_hit_version_id=(
            TOP_VERSION
        ),
        top_hit_matches_requested_scope=None,
        source_attribution_conflict=False,
    )


def make_paratext() -> ParatextEvidence:
    hit = PassageRoleEvidence(
        passage_id=(
            TOP_PASSAGE_ID
        ),
        version_id=TOP_VERSION,
        ordinal=30,
        role=ContentRole.UNKNOWN,
        configured=False,
        reason=None,
    )

    return ParatextEvidence(
        query_text=QUERY,
        hits=(hit,),
        evaluated_hit_count=1,
        authorial_hit_count=0,
        paratext_hit_count=0,
        mixed_hit_count=0,
        unknown_hit_count=1,
        paratext_or_mixed_hit_count=0,
        paratext_or_mixed_rate=0.0,
        top_hit_role=(
            ContentRole.UNKNOWN
        ),
        top_hit_is_paratext=False,
        top_hit_is_mixed=False,
        top_hit_is_structurally_non_authorial=False,
    )


def decide(
    *,
    dense_score: float,
    overlap_at_10: int = 0,
    config: AbstentionPolicyConfig
    | None = None,
    scope: ScopeEvidence
    | None = None,
):
    return AbstentionPolicy(
        config
    ).decide(
        retrieval=make_retrieval(
            dense_score=(
                dense_score
            ),
            overlap_at_10=(
                overlap_at_10
            ),
        ),
        scope=(
            scope
            if scope is not None
            else make_scope()
        ),
        paratext=make_paratext(),
    )


def test_frozen_yaml_loads_exact_threshold() -> None:
    config = (
        load_abstention_policy_config(
            Path(
                "config/abstention_policy.yaml"
            )
        )
    )

    assert config.policy_id == (
        "abstention-calibration-v1"
    )
    assert config.calibration_split_id == (
        "answerability-calibration-validation-v1"
    )
    assert (
        config.weak_evidence_dense_top_1_threshold
        == pytest.approx(
            0.863
        )
    )
    assert (
        config.weak_evidence_max_overlap_at_10
        == 0
    )


def test_low_dense_score_and_zero_overlap_abstains() -> None:
    decision = decide(
        dense_score=0.860086,
        overlap_at_10=0,
    )

    assert (
        decision.action
        is DecisionAction.ABSTAIN
    )
    assert (
        decision.reason
        is AbstentionReason.WEAK_CROSS_RETRIEVER_EVIDENCE
    )


def test_threshold_boundary_is_strict() -> None:
    decision = decide(
        dense_score=0.863,
        overlap_at_10=0,
    )

    assert (
        decision.action
        is DecisionAction.RETURN_RESULTS
    )


def test_positive_overlap_prevents_weak_evidence_abstention() -> None:
    decision = decide(
        dense_score=0.82,
        overlap_at_10=1,
    )

    assert (
        decision.action
        is DecisionAction.RETURN_RESULTS
    )


def test_weak_rule_can_be_disabled() -> None:
    decision = decide(
        dense_score=0.82,
        overlap_at_10=0,
        config=AbstentionPolicyConfig(
            reject_weak_cross_retriever_evidence=False
        ),
    )

    assert (
        decision.action
        is DecisionAction.RETURN_RESULTS
    )


def test_strong_scope_reason_keeps_precedence() -> None:
    out_of_corpus_mention = ScopeMention(
        entity_id="external:Hafez",
        kind=ScopeEntityKind.AUTHOR,
        label_fa="حافظ",
        matched_alias="حافظ",
        in_corpus=False,
        version_ids=(),
    )

    scope = replace(
        make_scope(),
        mentions=(
            out_of_corpus_mention,
        ),
        explicit_scope=True,
        known_out_of_corpus_scope_mentioned=True,
        known_out_of_corpus_entity_ids=(
            "external:Hafez",
        ),
    )

    decision = decide(
        dense_score=0.82,
        overlap_at_10=0,
        scope=scope,
    )

    assert decision.triggered_reasons == (
        AbstentionReason.KNOWN_OUT_OF_CORPUS_SCOPE,
        AbstentionReason.WEAK_CROSS_RETRIEVER_EVIDENCE,
    )

    assert (
        decision.reason
        is AbstentionReason.KNOWN_OUT_OF_CORPUS_SCOPE
    )


@pytest.mark.parametrize(
    "threshold",
    (
        float("nan"),
        float("inf"),
        -1.01,
        1.01,
    ),
)
def test_invalid_threshold_is_rejected(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between -1 and 1",
    ):
        AbstentionPolicyConfig(
            weak_evidence_dense_top_1_threshold=(
                threshold
            )
        )


@pytest.mark.parametrize(
    "overlap_cap",
    (
        -1,
        11,
        True,
    ),
)
def test_invalid_overlap_cap_is_rejected(
    overlap_cap,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 10",
    ):
        AbstentionPolicyConfig(
            weak_evidence_max_overlap_at_10=(
                overlap_cap
            )
        )


def test_loader_rejects_wrong_schema(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "policy.yaml"
    )

    path.write_text(
        """
schema_version: "9.9.9"
policy_id: "test"
calibration_split_id: "split"
rules:
  reject_known_out_of_corpus_scope: true
  reject_source_attribution_conflict: true
  reject_paratext_top_hit: true
  reject_mixed_top_hit: true
  weak_cross_retriever_evidence:
    enabled: true
    dense_top_1_score_less_than: 0.863
    overlap_at_10_less_than_or_equal: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AbstentionPolicyConfigError,
        match="Unsupported",
    ):
        load_abstention_policy_config(
            path
        )
