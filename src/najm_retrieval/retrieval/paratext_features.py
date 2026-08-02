"""Extract structural paratext evidence from hybrid rankings."""

from __future__ import annotations

from collections import Counter

from najm_retrieval.retrieval.hybrid_models import (
    HybridSearchResult,
)
from najm_retrieval.retrieval.paratext_catalog import (
    ParatextCatalog,
)
from najm_retrieval.retrieval.paratext_models import (
    ContentRole,
    ParatextEvidence,
)


class ParatextEvidenceExtractor:
    """Create decision-free structural evidence for the top retrieval hits."""

    def __init__(
        self,
        catalog: ParatextCatalog,
    ) -> None:
        self.catalog = catalog

    def extract(
        self,
        hybrid_result: HybridSearchResult,
    ) -> ParatextEvidence:
        """Classify at most the first ten hybrid hits."""

        evaluated_hits = (
            hybrid_result.hits[:10]
        )

        role_evidence = tuple(
            self.catalog.classify(
                passage_id=hit.passage_id,
                version_id=hit.version_id,
            )
            for hit in evaluated_hits
        )

        counts: Counter[
            ContentRole
        ] = Counter(
            evidence.role
            for evidence in role_evidence
        )

        paratext_count = counts[
            ContentRole.PARATEXT
        ]

        mixed_count = counts[
            ContentRole.MIXED
        ]

        non_authorial_count = (
            paratext_count
            + mixed_count
        )

        evaluated_count = len(
            role_evidence
        )

        top_role = (
            role_evidence[0].role
            if role_evidence
            else None
        )

        return ParatextEvidence(
            query_text=(
                hybrid_result.query_text
            ),
            hits=role_evidence,
            evaluated_hit_count=(
                evaluated_count
            ),
            authorial_hit_count=counts[
                ContentRole.AUTHORIAL
            ],
            paratext_hit_count=(
                paratext_count
            ),
            mixed_hit_count=mixed_count,
            unknown_hit_count=counts[
                ContentRole.UNKNOWN
            ],
            paratext_or_mixed_hit_count=(
                non_authorial_count
            ),
            paratext_or_mixed_rate=(
                non_authorial_count
                / evaluated_count
                if evaluated_count
                else 0.0
            ),
            top_hit_role=top_role,
            top_hit_is_paratext=(
                top_role
                is ContentRole.PARATEXT
            ),
            top_hit_is_mixed=(
                top_role
                is ContentRole.MIXED
            ),
            top_hit_is_structurally_non_authorial=(
                top_role
                in {
                    ContentRole.PARATEXT,
                    ContentRole.MIXED,
                }
            ),
        )


__all__ = [
    "ParatextEvidenceExtractor",
]
