"""Models for deterministic abstention-policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ABSTENTION_POLICY_SCHEMA_VERSION = "1.0.0"


class DecisionAction(str, Enum):
    """Runtime action selected by the abstention policy."""

    RETURN_RESULTS = "return_results"
    ABSTAIN = "abstain"


class AbstentionReason(str, Enum):
    """Auditable reasons emitted by the baseline policy."""

    KNOWN_OUT_OF_CORPUS_SCOPE = (
        "known_out_of_corpus_scope"
    )
    SOURCE_ATTRIBUTION_CONFLICT = (
        "source_attribution_conflict"
    )
    TOP_HIT_PARATEXT = "top_hit_paratext"
    TOP_HIT_MIXED = "top_hit_mixed"
    NO_HYBRID_HITS = "no_hybrid_hits"
    BASELINE_EVIDENCE_PASSED = (
        "baseline_evidence_passed"
    )


@dataclass(frozen=True)
class AbstentionPolicyConfig:
    """Switches for strong deterministic abstention rules."""

    reject_known_out_of_corpus_scope: bool = True
    reject_source_attribution_conflict: bool = True
    reject_paratext_top_hit: bool = True
    reject_mixed_top_hit: bool = True

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
        ):
            if not isinstance(value, bool):
                raise ValueError(
                    f"{label} must be Boolean."
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
]
