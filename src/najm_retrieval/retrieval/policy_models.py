"""Models for deterministic abstention-policy decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


ABSTENTION_POLICY_SCHEMA_VERSION = "1.2.0"


class DecisionAction(str, Enum):
    """Runtime action selected by the abstention policy."""

    RETURN_RESULTS = "return_results"
    ABSTAIN = "abstain"


class AbstentionReason(str, Enum):
    """Auditable reasons emitted by the calibrated policy."""

    KNOWN_OUT_OF_CORPUS_SCOPE = (
        "known_out_of_corpus_scope"
    )
    SOURCE_ATTRIBUTION_CONFLICT = (
        "source_attribution_conflict"
    )
    TOP_HIT_PARATEXT = "top_hit_paratext"
    TOP_HIT_MIXED = "top_hit_mixed"
    NO_HYBRID_HITS = "no_hybrid_hits"
    WEAK_CROSS_RETRIEVER_EVIDENCE = (
        "weak_cross_retriever_evidence"
    )
    BASELINE_EVIDENCE_PASSED = (
        "baseline_evidence_passed"
    )


@dataclass(frozen=True)
class RetrievalProfileConfig:
    """Frozen retrieval settings for which the policy was calibrated."""

    corpus_artifact_id: str = (
        "corpus-ad111acd912e"
    )
    dense_model_name: str = (
        "intfloat/multilingual-e5-small"
    )
    lexical_weight: float = 2.0
    dense_weight: float = 1.0
    rrf_constant: float = 60.0
    candidate_limit: int = 100
    return_limit: int = 10

    def __post_init__(self) -> None:
        for label, value in (
            (
                "corpus_artifact_id",
                self.corpus_artifact_id,
            ),
            (
                "dense_model_name",
                self.dense_model_name,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise ValueError(
                    f"{label} must not be empty."
                )

        for label, value in (
            (
                "lexical_weight",
                self.lexical_weight,
            ),
            (
                "dense_weight",
                self.dense_weight,
            ),
        ):
            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
                or not math.isfinite(
                    float(
                        value
                    )
                )
                or float(
                    value
                )
                < 0
            ):
                raise ValueError(
                    f"{label} must be finite and non-negative."
                )

        if (
            float(
                self.lexical_weight
            )
            == 0.0
            and float(
                self.dense_weight
            )
            == 0.0
        ):
            raise ValueError(
                "At least one retrieval weight must be positive."
            )

        if (
            isinstance(
                self.rrf_constant,
                bool,
            )
            or not isinstance(
                self.rrf_constant,
                (
                    int,
                    float,
                ),
            )
            or not math.isfinite(
                float(
                    self.rrf_constant
                )
            )
            or float(
                self.rrf_constant
            )
            <= 0
        ):
            raise ValueError(
                "rrf_constant must be finite and positive."
            )

        for label, value in (
            (
                "candidate_limit",
                self.candidate_limit,
            ),
            (
                "return_limit",
                self.return_limit,
            ),
        ):
            if (
                not isinstance(
                    value,
                    int,
                )
                or isinstance(
                    value,
                    bool,
                )
                or not 1
                <= value
                <= 100
            ):
                raise ValueError(
                    f"{label} must be an integer between 1 and 100."
                )

        if (
            self.return_limit
            > self.candidate_limit
        ):
            raise ValueError(
                "return_limit must not exceed candidate_limit."
            )


@dataclass(frozen=True)
class AbstentionPolicyConfig:
    """Frozen switches, thresholds, and retrieval provenance."""

    policy_id: str = (
        "abstention-calibration-v1"
    )
    calibration_split_id: str = (
        "answerability-calibration-validation-v1"
    )

    retrieval_profile: RetrievalProfileConfig = field(
        default_factory=RetrievalProfileConfig
    )

    reject_known_out_of_corpus_scope: bool = True
    reject_source_attribution_conflict: bool = True
    reject_paratext_top_hit: bool = True
    reject_mixed_top_hit: bool = True
    reject_weak_cross_retriever_evidence: bool = True

    weak_evidence_dense_top_1_threshold: float = (
        0.863
    )
    weak_evidence_max_overlap_at_10: int = 0

    schema_version: str = (
        ABSTENTION_POLICY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != ABSTENTION_POLICY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported abstention-policy schema version: "
                f"{self.schema_version}"
            )

        for label, value in (
            (
                "policy_id",
                self.policy_id,
            ),
            (
                "calibration_split_id",
                self.calibration_split_id,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise ValueError(
                    f"{label} must not be empty."
                )

        if not isinstance(
            self.retrieval_profile,
            RetrievalProfileConfig,
        ):
            raise ValueError(
                "retrieval_profile must be a RetrievalProfileConfig."
            )

        for label, value in (
            (
                "reject_known_out_of_corpus_scope",
                self.reject_known_out_of_corpus_scope,
            ),
            (
                "reject_source_attribution_conflict",
                self.reject_source_attribution_conflict,
            ),
            (
                "reject_paratext_top_hit",
                self.reject_paratext_top_hit,
            ),
            (
                "reject_mixed_top_hit",
                self.reject_mixed_top_hit,
            ),
            (
                "reject_weak_cross_retriever_evidence",
                self.reject_weak_cross_retriever_evidence,
            ),
        ):
            if not isinstance(value, bool):
                raise ValueError(
                    f"{label} must be Boolean."
                )

        threshold = (
            self.weak_evidence_dense_top_1_threshold
        )

        if (
            isinstance(
                threshold,
                bool,
            )
            or not isinstance(
                threshold,
                (
                    int,
                    float,
                ),
            )
            or not math.isfinite(
                float(
                    threshold
                )
            )
            or not -1.0
            <= float(
                threshold
            )
            <= 1.0
        ):
            raise ValueError(
                "weak_evidence_dense_top_1_threshold "
                "must be finite and between -1 and 1."
            )

        overlap_cap = (
            self.weak_evidence_max_overlap_at_10
        )

        if (
            not isinstance(
                overlap_cap,
                int,
            )
            or isinstance(
                overlap_cap,
                bool,
            )
            or not 0
            <= overlap_cap
            <= 10
        ):
            raise ValueError(
                "weak_evidence_max_overlap_at_10 "
                "must be an integer between 0 and 10."
            )


@dataclass(frozen=True)
class RetrievalDecision:
    """One validated and auditable policy decision."""

    query_text: str
    action: DecisionAction
    reason: AbstentionReason
    return_results: bool

    top_passage_id: str | None
    triggered_reasons: tuple[
        AbstentionReason,
        ...,
    ]

    schema_version: str = (
        ABSTENTION_POLICY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.query_text,
            str,
        ) or not self.query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        if (
            self.schema_version
            != ABSTENTION_POLICY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported retrieval-decision schema version: "
                f"{self.schema_version}"
            )

        if (
            self.top_passage_id is not None
            and (
                not isinstance(
                    self.top_passage_id,
                    str,
                )
                or not self.top_passage_id.strip()
            )
        ):
            raise ValueError(
                "top_passage_id must be non-empty when provided."
            )

        if not self.triggered_reasons:
            raise ValueError(
                "triggered_reasons must not be empty."
            )

        if len(
            self.triggered_reasons
        ) != len(
            set(self.triggered_reasons)
        ):
            raise ValueError(
                "triggered_reasons must be unique."
            )

        if self.reason is not self.triggered_reasons[0]:
            raise ValueError(
                "reason must be the first triggered reason."
            )

        expected_return_results = (
            self.action
            is DecisionAction.RETURN_RESULTS
        )

        if (
            self.return_results
            != expected_return_results
        ):
            raise ValueError(
                "return_results must match action."
            )

        if (
            self.action
            is DecisionAction.RETURN_RESULTS
        ):
            if (
                self.reason
                is not AbstentionReason.BASELINE_EVIDENCE_PASSED
            ):
                raise ValueError(
                    "Returning results requires the baseline-pass reason."
                )

            if self.triggered_reasons != (
                AbstentionReason.BASELINE_EVIDENCE_PASSED,
            ):
                raise ValueError(
                    "A return decision cannot contain abstention reasons."
                )

            if self.top_passage_id is None:
                raise ValueError(
                    "Returning results requires a top passage."
                )
        else:
            if (
                self.reason
                is AbstentionReason.BASELINE_EVIDENCE_PASSED
            ):
                raise ValueError(
                    "An abstention decision cannot use the baseline-pass reason."
                )

            if (
                AbstentionReason.BASELINE_EVIDENCE_PASSED
                in self.triggered_reasons
            ):
                raise ValueError(
                    "An abstention decision cannot include the baseline-pass reason."
                )

        if (
            self.reason
            is AbstentionReason.NO_HYBRID_HITS
            and self.top_passage_id is not None
        ):
            raise ValueError(
                "A no-hybrid-hits decision cannot contain a top passage."
            )


__all__ = [
    "ABSTENTION_POLICY_SCHEMA_VERSION",
    "AbstentionPolicyConfig",
    "AbstentionReason",
    "DecisionAction",
    "RetrievalDecision",
    "RetrievalProfileConfig",
]
