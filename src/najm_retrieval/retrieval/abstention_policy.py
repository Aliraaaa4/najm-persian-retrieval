"""Deterministic baseline abstention policy."""

from __future__ import annotations

from najm_retrieval.retrieval.abstention_models import (
    AbstentionFeatures,
)
from najm_retrieval.retrieval.paratext_models import (
    ParatextEvidence,
)
from najm_retrieval.retrieval.policy_models import (
    AbstentionPolicyConfig,
    AbstentionReason,
    DecisionAction,
    RetrievalDecision,
)
from najm_retrieval.retrieval.scope_models import (
    ScopeEvidence,
)


class AbstentionPolicy:
    """Apply strong evidence rules without score thresholds.

    This baseline intentionally avoids treating retrieval scores as
    probabilities. Threshold-based rules are added only after calibration.
    """

    def __init__(
        self,
        config: AbstentionPolicyConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else AbstentionPolicyConfig()
        )

    def decide(
        self,
        *,
        retrieval: AbstentionFeatures,
        scope: ScopeEvidence,
        paratext: ParatextEvidence,
    ) -> RetrievalDecision:
        """Return one deterministic decision for aligned evidence."""

        self._validate_alignment(
            retrieval=retrieval,
            scope=scope,
            paratext=paratext,
        )

        triggered: list[
            AbstentionReason
        ] = []

        if (
            self.config.reject_known_out_of_corpus_scope
            and scope.known_out_of_corpus_scope_mentioned
        ):
            triggered.append(
                AbstentionReason.KNOWN_OUT_OF_CORPUS_SCOPE
            )

        if (
            self.config.reject_source_attribution_conflict
            and scope.source_attribution_conflict
        ):
            triggered.append(
                AbstentionReason.SOURCE_ATTRIBUTION_CONFLICT
            )

        if (
            self.config.reject_paratext_top_hit
            and paratext.top_hit_is_paratext
        ):
            triggered.append(
                AbstentionReason.TOP_HIT_PARATEXT
            )

        if (
            self.config.reject_mixed_top_hit
            and paratext.top_hit_is_mixed
        ):
            triggered.append(
                AbstentionReason.TOP_HIT_MIXED
            )

        if retrieval.hybrid_hit_count == 0:
            triggered.append(
                AbstentionReason.NO_HYBRID_HITS
            )

        if triggered:
            return RetrievalDecision(
                query_text=retrieval.query_text,
                action=DecisionAction.ABSTAIN,
                reason=triggered[0],
                return_results=False,
                top_passage_id=(
                    retrieval.hybrid_top_1_passage_id
                ),
                triggered_reasons=tuple(
                    triggered
                ),
            )

        return RetrievalDecision(
            query_text=retrieval.query_text,
            action=DecisionAction.RETURN_RESULTS,
            reason=(
                AbstentionReason.BASELINE_EVIDENCE_PASSED
            ),
            return_results=True,
            top_passage_id=(
                retrieval.hybrid_top_1_passage_id
            ),
            triggered_reasons=(
                AbstentionReason.BASELINE_EVIDENCE_PASSED,
            ),
        )

    @staticmethod
    def _validate_alignment(
        *,
        retrieval: AbstentionFeatures,
        scope: ScopeEvidence,
        paratext: ParatextEvidence,
    ) -> None:
        query_texts = {
            retrieval.query_text.strip(),
            scope.query_text.strip(),
            paratext.query_text.strip(),
        }

        if len(query_texts) != 1:
            raise ValueError(
                "Retrieval, scope, and paratext evidence "
                "must use the same query_text."
            )

        expected_depth = min(
            retrieval.hybrid_hit_count,
            10,
        )

        if (
            scope.evaluated_hit_count
            != expected_depth
        ):
            raise ValueError(
                "Scope evidence depth does not match hybrid retrieval depth."
            )

        if (
            paratext.evaluated_hit_count
            != expected_depth
        ):
            raise ValueError(
                "Paratext evidence depth does not match hybrid retrieval depth."
            )

        scope_versions = (
            scope.retrieved_version_ids_at_10
        )

        paratext_versions = tuple(
            hit.version_id
            for hit in paratext.hits
        )

        if scope_versions != paratext_versions:
            raise ValueError(
                "Scope and paratext evidence must describe "
                "the same hybrid ranking."
            )

        if retrieval.hybrid_hit_count == 0:
            if (
                retrieval.hybrid_top_1_passage_id
                is not None
            ):
                raise ValueError(
                    "Empty hybrid evidence cannot contain a top passage."
                )
            return

        if not paratext.hits:
            raise ValueError(
                "Non-empty hybrid evidence requires paratext hits."
            )

        if (
            retrieval.hybrid_top_1_passage_id
            != paratext.hits[0].passage_id
        ):
            raise ValueError(
                "Retrieval and paratext top passage IDs do not match."
            )

        if (
            scope.top_hit_version_id
            != paratext.hits[0].version_id
        ):
            raise ValueError(
                "Scope and paratext top version IDs do not match."
            )


__all__ = [
    "AbstentionPolicy",
]
