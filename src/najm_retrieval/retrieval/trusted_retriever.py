"""Runtime orchestration for retrieval and abstention evidence."""

from __future__ import annotations

from typing import Protocol

from najm_retrieval.retrieval.abstention_features import (
    AbstentionFeatureExtractor,
)
from najm_retrieval.retrieval.abstention_models import (
    AbstentionFeatures,
)
from najm_retrieval.retrieval.abstention_policy import (
    AbstentionPolicy,
)
from najm_retrieval.retrieval.hybrid_models import (
    HybridRetrievalRun,
    HybridSearchResult,
)
from najm_retrieval.retrieval.models import (
    LexicalSearchMode,
)
from najm_retrieval.retrieval.paratext_models import (
    ParatextEvidence,
)
from najm_retrieval.retrieval.policy_models import (
    AbstentionPolicyConfig,
    RetrievalDecision,
)
from najm_retrieval.retrieval.retrieval_profile import (
    validate_retrieval_profile,
)
from najm_retrieval.retrieval.scope_models import (
    ScopeEvidence,
)
from najm_retrieval.retrieval.trusted_models import (
    TrustedRetrievalResult,
)


class _HybridRunner(Protocol):
    lexical_weight: float
    dense_weight: float
    rrf_constant: float
    candidate_limit: int

    def search_with_components(
        self,
        query_text: str,
        *,
        limit: int,
        lexical_mode: LexicalSearchMode,
    ) -> HybridRetrievalRun:
        """Run lexical, dense, and fusion once."""


class _FeatureExtractor(Protocol):
    def extract(
        self,
        *,
        lexical_result: object,
        dense_result: object,
        hybrid_result: HybridSearchResult,
    ) -> AbstentionFeatures:
        """Extract retrieval evidence."""


class _ScopeExtractor(Protocol):
    def extract(
        self,
        *,
        query_text: str,
        hybrid_result: HybridSearchResult,
    ) -> ScopeEvidence:
        """Extract corpus-scope evidence."""


class _ParatextExtractor(Protocol):
    def extract(
        self,
        hybrid_result: HybridSearchResult,
    ) -> ParatextEvidence:
        """Extract structural-role evidence."""


class _DecisionPolicy(Protocol):
    def decide(
        self,
        *,
        retrieval: AbstentionFeatures,
        scope: ScopeEvidence,
        paratext: ParatextEvidence,
    ) -> RetrievalDecision:
        """Return one trusted retrieval decision."""


class TrustedRetriever:
    """Run retrieval once and apply the frozen trust policy."""

    def __init__(
        self,
        hybrid_retriever: _HybridRunner,
        *,
        policy_config: AbstentionPolicyConfig,
        scope_extractor: _ScopeExtractor,
        paratext_extractor: _ParatextExtractor,
        corpus_artifact_id: str,
        dense_model_name: str,
        return_limit: int | None = None,
        feature_extractor: _FeatureExtractor | None = None,
        policy: _DecisionPolicy | None = None,
    ) -> None:
        if (
            not isinstance(corpus_artifact_id, str)
            or not corpus_artifact_id.strip()
        ):
            raise ValueError(
                "corpus_artifact_id must not be empty."
            )

        if (
            not isinstance(dense_model_name, str)
            or not dense_model_name.strip()
        ):
            raise ValueError(
                "dense_model_name must not be empty."
            )

        actual_return_limit = (
            policy_config.retrieval_profile.return_limit
            if return_limit is None
            else return_limit
        )

        validate_retrieval_profile(
            policy_config.retrieval_profile,
            corpus_artifact_id=corpus_artifact_id,
            dense_model_name=dense_model_name,
            lexical_weight=(
                hybrid_retriever.lexical_weight
            ),
            dense_weight=(
                hybrid_retriever.dense_weight
            ),
            rrf_constant=(
                hybrid_retriever.rrf_constant
            ),
            candidate_limit=(
                hybrid_retriever.candidate_limit
            ),
            return_limit=actual_return_limit,
        )

        self.hybrid_retriever = hybrid_retriever
        self.policy_config = policy_config
        self.scope_extractor = scope_extractor
        self.paratext_extractor = paratext_extractor
        self.corpus_artifact_id = corpus_artifact_id
        self.dense_model_name = dense_model_name
        self.return_limit = actual_return_limit

        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else AbstentionFeatureExtractor()
        )

        self.policy = (
            policy
            if policy is not None
            else AbstentionPolicy(
                policy_config
            )
        )

    def search(
        self,
        query_text: str,
        *,
        lexical_mode: LexicalSearchMode = (
            LexicalSearchMode.AUTO
        ),
    ) -> TrustedRetrievalResult:
        """Run retrieval once and return a policy-controlled result."""

        retrieval_run = (
            self.hybrid_retriever.search_with_components(
                query_text,
                limit=self.return_limit,
                lexical_mode=lexical_mode,
            )
        )

        retrieval_features = (
            self.feature_extractor.extract(
                lexical_result=(
                    retrieval_run.lexical_result
                ),
                dense_result=(
                    retrieval_run.dense_result
                ),
                hybrid_result=(
                    retrieval_run.hybrid_result
                ),
            )
        )

        scope_evidence = (
            self.scope_extractor.extract(
                query_text=(
                    retrieval_run.hybrid_result.query_text
                ),
                hybrid_result=(
                    retrieval_run.hybrid_result
                ),
            )
        )

        paratext_evidence = (
            self.paratext_extractor.extract(
                retrieval_run.hybrid_result
            )
        )

        decision = self.policy.decide(
            retrieval=retrieval_features,
            scope=scope_evidence,
            paratext=paratext_evidence,
        )

        return TrustedRetrievalResult(
            query_text=(
                retrieval_run.hybrid_result.query_text
            ),
            decision=decision,
            retrieval_run=retrieval_run,
            retrieval_features=retrieval_features,
            scope_evidence=scope_evidence,
            paratext_evidence=paratext_evidence,
        )


__all__ = [
    "TrustedRetriever",
]
