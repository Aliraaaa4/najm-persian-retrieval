"""Models for trusted retrieval decisions and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from najm_retrieval.retrieval.abstention_models import (
    AbstentionFeatures,
)
from najm_retrieval.retrieval.hybrid_models import (
    HybridRetrievalRun,
    HybridSearchHit,
)
from najm_retrieval.retrieval.paratext_models import (
    ParatextEvidence,
)
from najm_retrieval.retrieval.policy_models import (
    RetrievalDecision,
)
from najm_retrieval.retrieval.scope_models import (
    ScopeEvidence,
)


TRUSTED_RETRIEVAL_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class TrustedRetrievalResult:
    """Decision, public hits, and internal retrieval evidence."""

    query_text: str
    decision: RetrievalDecision
    retrieval_run: HybridRetrievalRun
    retrieval_features: AbstentionFeatures
    scope_evidence: ScopeEvidence
    paratext_evidence: ParatextEvidence
    schema_version: str = TRUSTED_RETRIEVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate query alignment and top-hit consistency."""

        if not self.query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        if (
            self.schema_version
            != TRUSTED_RETRIEVAL_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported trusted retrieval schema version: "
                f"{self.schema_version}"
            )

        query_texts = {
            self.query_text,
            self.decision.query_text,
            self.retrieval_run.dense_result.query_text,
            self.retrieval_run.hybrid_result.query_text,
            self.retrieval_features.query_text,
            self.scope_evidence.query_text,
            self.paratext_evidence.query_text,
        }

        if self.retrieval_run.lexical_result is not None:
            query_texts.add(
                self.retrieval_run.lexical_result.query_text
            )

        if len(query_texts) != 1:
            raise ValueError(
                "Trusted retrieval components must use "
                "the same query_text."
            )

        diagnostic_hits = (
            self.retrieval_run.hybrid_result.hits
        )

        expected_top_passage_id = (
            diagnostic_hits[0].passage_id
            if diagnostic_hits
            else None
        )

        if (
            self.decision.top_passage_id
            != expected_top_passage_id
        ):
            raise ValueError(
                "Decision top_passage_id does not match "
                "the hybrid ranking."
            )

        if (
            self.decision.return_results
            and not diagnostic_hits
        ):
            raise ValueError(
                "A return-results decision requires "
                "at least one hybrid hit."
            )

    @property
    def hits(
        self,
    ) -> tuple[HybridSearchHit, ...]:
        """Return only hits approved for public use."""

        if not self.decision.return_results:
            return ()

        return self.retrieval_run.hybrid_result.hits

    @property
    def diagnostic_hits(
        self,
    ) -> tuple[HybridSearchHit, ...]:
        """Return all internal hybrid hits for diagnostics."""

        return self.retrieval_run.hybrid_result.hits

    @property
    def abstained(self) -> bool:
        """Whether the policy withheld public results."""

        return not self.decision.return_results


__all__ = [
    "TRUSTED_RETRIEVAL_SCHEMA_VERSION",
    "TrustedRetrievalResult",
]
