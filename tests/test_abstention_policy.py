"""Tests for the deterministic baseline abstention policy."""

from __future__ import annotations

from dataclasses import (
    FrozenInstanceError,
    replace,
)

import pytest

from najm_retrieval.retrieval import (
    AbstentionFeatures,
    AbstentionPolicy,
    AbstentionPolicyConfig,
    AbstentionReason,
    ContentRole,
    DecisionAction,
    ParatextEvidence,
    PassageRoleEvidence,
    RetrievalDecision,
    ScopeEntityKind,
    ScopeEvidence,
    ScopeMention,
)


QUERY = "پرسش آزمایشی"

TOP_VERSION = (
    "0672JalalDinRumi."
    "MajalisSabica."
    "AOCP202502141236-per1"
)

OTHER_VERSION = (
    "0672JalalDinRumi."
    "Diwan.PDL00047-per1"
)

TOP_PASSAGE_ID = (
    f"{TOP_VERSION}:"
    "passage_000030"
)


def make_retrieval(
    *,
    query_text: str = QUERY,
    hit_count: int = 1,
    top_passage_id: str | None = (
        TOP_PASSAGE_ID
    ),
) -> AbstentionFeatures:
    if hit_count == 0:
        top_passage_id = None

    return AbstentionFeatures(
        query_text=query_text,
        lexical_result_available=False,
        lexical_mode_used=None,
        lexical_hit_count=0,
        dense_hit_count=hit_count,
        hybrid_hit_count=hit_count,
        lexical_top_1_passage_id=None,
        dense_top_1_passage_id=(
            top_passage_id
        ),
        hybrid_top_1_passage_id=(
            top_passage_id
        ),
        lexical_top_1_bm25=None,
        dense_top_1_score=(
            0.88
            if hit_count
            else None
        ),
        dense_top_2_score=None,
        dense_margin_1_2=None,
        overlap_at_10=0,
        overlap_at_100=0,
        top_1_same_passage=False,
        hybrid_top_1_score=(
            0.03
            if hit_count
            else None
        ),
        hybrid_top_2_score=None,
        hybrid_margin_1_2=None,
        hybrid_top_1_lexical_rank=None,
        hybrid_top_1_dense_rank=(
            1
            if hit_count
            else None
        ),
        hybrid_top_1_dual_supported=False,
    )


def make_scope(
    *,
    query_text: str = QUERY,
    retrieved_version_ids: tuple[
        str,
        ...,
    ] = (TOP_VERSION,),
    known_ooc: bool = False,
    requested_version_id: str
    | None = None,
) -> ScopeEvidence:
    mentions: list[
        ScopeMention
    ] = []

    requested_author_ids: tuple[
        str,
        ...,
    ] = ()

    requested_version_ids: tuple[
        str,
        ...,
    ] = ()

    if requested_version_id is not None:
        mentions.append(
            ScopeMention(
                entity_id=(
                    "0672JalalDinRumi"
                ),
                kind=(
                    ScopeEntityKind.AUTHOR
                ),
                label_fa="مولوی",
                matched_alias="مولوی",
                in_corpus=True,
                version_ids=(
                    requested_version_id,
                ),
            )
        )

        requested_author_ids = (
            "0672JalalDinRumi",
        )

        requested_version_ids = (
            requested_version_id,
        )

    known_ooc_ids: tuple[
        str,
        ...,
    ] = ()

    if known_ooc:
        mentions.append(
            ScopeMention(
                entity_id=(
                    "external:Hafez"
                ),
                kind=(
                    ScopeEntityKind.AUTHOR
                ),
                label_fa="حافظ",
                matched_alias="حافظ",
                in_corpus=False,
                version_ids=(),
            )
        )

        known_ooc_ids = (
            "external:Hafez",
        )

    evaluated_count = len(
        retrieved_version_ids
    )

    if requested_version_ids:
        matching_count = sum(
            version_id
            in requested_version_ids
            for version_id
            in retrieved_version_ids
        )

        matching_rate: float | None = (
            matching_count
            / evaluated_count
            if evaluated_count
            else 0.0
        )
    else:
        matching_count = 0
        matching_rate = None

    top_version_id = (
        retrieved_version_ids[0]
        if retrieved_version_ids
        else None
    )

    if (
        requested_version_ids
        and top_version_id is not None
    ):
        top_matches: bool | None = (
            top_version_id
            in requested_version_ids
        )
    else:
        top_matches = None

    return ScopeEvidence(
        query_text=query_text,
        mentions=tuple(
            mentions
        ),
        explicit_scope=bool(
            mentions
        ),
        in_corpus_scope_mentioned=any(
            mention.in_corpus
            for mention in mentions
        ),
        known_out_of_corpus_scope_mentioned=(
            known_ooc
        ),
        requested_author_ids=(
            requested_author_ids
        ),
        requested_work_ids=(),
        known_out_of_corpus_entity_ids=(
            known_ooc_ids
        ),
        requested_version_ids=(
            requested_version_ids
        ),
        evaluated_hit_count=(
            evaluated_count
        ),
        retrieved_version_ids_at_10=(
            retrieved_version_ids
        ),
        matching_hit_count_at_10=(
            matching_count
        ),
        matching_hit_rate_at_10=(
            matching_rate
        ),
        top_hit_version_id=(
            top_version_id
        ),
        top_hit_matches_requested_scope=(
            top_matches
        ),
        source_attribution_conflict=(
            top_matches is False
        ),
    )


def make_paratext(
    *,
    query_text: str = QUERY,
    role: ContentRole
    | None = ContentRole.AUTHORIAL,
    passage_id: str = TOP_PASSAGE_ID,
    version_id: str = TOP_VERSION,
) -> ParatextEvidence:
    if role is None:
        return ParatextEvidence(
            query_text=query_text,
            hits=(),
            evaluated_hit_count=0,
            authorial_hit_count=0,
            paratext_hit_count=0,
            mixed_hit_count=0,
            unknown_hit_count=0,
            paratext_or_mixed_hit_count=0,
            paratext_or_mixed_rate=0.0,
            top_hit_role=None,
            top_hit_is_paratext=False,
            top_hit_is_mixed=False,
            top_hit_is_structurally_non_authorial=False,
        )

    configured = (
        role
        is not ContentRole.UNKNOWN
    )

    hit = PassageRoleEvidence(
        passage_id=passage_id,
        version_id=version_id,
        ordinal=30,
        role=role,
        configured=configured,
        reason=(
            "main_work"
            if role
            is ContentRole.AUTHORIAL
            else (
                "editorial_front_matter"
                if role
                is ContentRole.PARATEXT
                else (
                    "front_matter_to_authorial_transition"
                    if role
                    is ContentRole.MIXED
                    else None
                )
            )
        ),
    )

    return ParatextEvidence(
        query_text=query_text,
        hits=(hit,),
        evaluated_hit_count=1,
        authorial_hit_count=(
            1
            if role
            is ContentRole.AUTHORIAL
            else 0
        ),
        paratext_hit_count=(
            1
            if role
            is ContentRole.PARATEXT
            else 0
        ),
        mixed_hit_count=(
            1
            if role
            is ContentRole.MIXED
            else 0
        ),
        unknown_hit_count=(
            1
            if role
            is ContentRole.UNKNOWN
            else 0
        ),
        paratext_or_mixed_hit_count=(
            1
            if role
            in {
                ContentRole.PARATEXT,
                ContentRole.MIXED,
            }
            else 0
        ),
        paratext_or_mixed_rate=(
            1.0
            if role
            in {
                ContentRole.PARATEXT,
                ContentRole.MIXED,
            }
            else 0.0
        ),
        top_hit_role=role,
        top_hit_is_paratext=(
            role
            is ContentRole.PARATEXT
        ),
        top_hit_is_mixed=(
            role
            is ContentRole.MIXED
        ),
        top_hit_is_structurally_non_authorial=(
            role
            in {
                ContentRole.PARATEXT,
                ContentRole.MIXED,
            }
        ),
    )


def decide(
    *,
    retrieval: AbstentionFeatures
    | None = None,
    scope: ScopeEvidence
    | None = None,
    paratext: ParatextEvidence
    | None = None,
    config: AbstentionPolicyConfig
    | None = None,
) -> RetrievalDecision:
    return AbstentionPolicy(
        config
    ).decide(
        retrieval=(
            retrieval
            if retrieval is not None
            else make_retrieval()
        ),
        scope=(
            scope
            if scope is not None
            else make_scope()
        ),
        paratext=(
            paratext
            if paratext is not None
            else make_paratext()
        ),
    )


def test_default_config_enables_strong_rules() -> None:
    config = AbstentionPolicyConfig()

    assert (
        config.reject_known_out_of_corpus_scope
    )
    assert (
        config.reject_source_attribution_conflict
    )
    assert config.reject_paratext_top_hit
    assert config.reject_mixed_top_hit


def test_known_out_of_corpus_scope_abstains() -> None:
    decision = decide(
        scope=make_scope(
            known_ooc=True
        )
    )

    assert (
        decision.action
        is DecisionAction.ABSTAIN
    )
    assert (
        decision.reason
        is AbstentionReason.KNOWN_OUT_OF_CORPUS_SCOPE
    )
    assert not decision.return_results


def test_source_attribution_conflict_abstains() -> None:
    decision = decide(
        scope=make_scope(
            requested_version_id=(
                OTHER_VERSION
            ),
        )
    )

    assert (
        decision.reason
        is AbstentionReason.SOURCE_ATTRIBUTION_CONFLICT
    )


def test_paratext_top_hit_abstains() -> None:
    decision = decide(
        paratext=make_paratext(
            role=ContentRole.PARATEXT
        )
    )

    assert (
        decision.reason
        is AbstentionReason.TOP_HIT_PARATEXT
    )


def test_mixed_top_hit_abstains() -> None:
    decision = decide(
        paratext=make_paratext(
            role=ContentRole.MIXED
        )
    )

    assert (
        decision.reason
        is AbstentionReason.TOP_HIT_MIXED
    )


def test_empty_hybrid_result_abstains() -> None:
    decision = decide(
        retrieval=make_retrieval(
            hit_count=0
        ),
        scope=make_scope(
            retrieved_version_ids=()
        ),
        paratext=make_paratext(
            role=None
        ),
    )

    assert (
        decision.reason
        is AbstentionReason.NO_HYBRID_HITS
    )
    assert decision.top_passage_id is None


def test_authorial_top_hit_returns_results() -> None:
    decision = decide()

    assert (
        decision.action
        is DecisionAction.RETURN_RESULTS
    )
    assert (
        decision.reason
        is AbstentionReason.BASELINE_EVIDENCE_PASSED
    )
    assert decision.return_results
    assert (
        decision.top_passage_id
        == TOP_PASSAGE_ID
    )


def test_unknown_paratext_role_does_not_force_abstention() -> None:
    decision = decide(
        paratext=make_paratext(
            role=ContentRole.UNKNOWN
        )
    )

    assert (
        decision.action
        is DecisionAction.RETURN_RESULTS
    )


def test_rule_precedence_is_stable_and_auditable() -> None:
    decision = decide(
        scope=make_scope(
            known_ooc=True,
            requested_version_id=(
                OTHER_VERSION
            ),
        ),
        paratext=make_paratext(
            role=ContentRole.PARATEXT
        ),
    )

    assert decision.triggered_reasons == (
        AbstentionReason.KNOWN_OUT_OF_CORPUS_SCOPE,
        AbstentionReason.SOURCE_ATTRIBUTION_CONFLICT,
        AbstentionReason.TOP_HIT_PARATEXT,
    )

    assert (
        decision.reason
        is AbstentionReason.KNOWN_OUT_OF_CORPUS_SCOPE
    )


def test_query_text_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="same query_text",
    ):
        decide(
            scope=make_scope(
                query_text="پرسش دیگر"
            )
        )


def test_evidence_depth_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Scope evidence depth",
    ):
        decide(
            scope=make_scope(
                retrieved_version_ids=()
            )
        )


def test_top_passage_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="top passage IDs",
    ):
        decide(
            retrieval=make_retrieval(
                top_passage_id=(
                    f"{TOP_VERSION}:"
                    "passage_000031"
                )
            )
        )


def test_version_ranking_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="same hybrid ranking",
    ):
        decide(
            scope=make_scope(
                retrieved_version_ids=(
                    OTHER_VERSION,
                )
            )
        )


def test_individual_rule_can_be_disabled() -> None:
    decision = decide(
        config=AbstentionPolicyConfig(
            reject_paratext_top_hit=False
        ),
        paratext=make_paratext(
            role=ContentRole.PARATEXT
        ),
    )

    assert (
        decision.action
        is DecisionAction.RETURN_RESULTS
    )


def test_decision_is_frozen_and_validated() -> None:
    decision = decide()

    with pytest.raises(
        FrozenInstanceError,
    ):
        decision.return_results = False  # type: ignore[misc]

    with pytest.raises(
        ValueError,
        match="return_results",
    ):
        replace(
            decision,
            return_results=False,
        )


def test_public_policy_symbols_can_be_imported() -> None:
    assert AbstentionPolicy.__name__ == (
        "AbstentionPolicy"
    )
    assert RetrievalDecision.__name__ == (
        "RetrievalDecision"
    )
