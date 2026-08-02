"""Extract corpus-scope and source-attribution evidence."""

from __future__ import annotations

from najm_retrieval.retrieval.hybrid_models import (
    HybridSearchResult,
)
from najm_retrieval.retrieval.scope_catalog import (
    CorpusScopeCatalog,
)
from najm_retrieval.retrieval.scope_models import (
    ScopeEntityKind,
    ScopeEvidence,
)


class ScopeEvidenceExtractor:
    """Compare explicit query scope with retrieved version identifiers."""

    def __init__(
        self,
        catalog: CorpusScopeCatalog,
    ) -> None:
        self.catalog = catalog

    def extract(
        self,
        *,
        query_text: str,
        hybrid_result: HybridSearchResult,
    ) -> ScopeEvidence:
        """Return decision-free scope and source-attribution evidence."""

        cleaned_query = query_text.strip()

        if not cleaned_query:
            raise ValueError(
                "query_text must not be empty."
            )

        if hybrid_result.query_text.strip() != cleaned_query:
            raise ValueError(
                "query_text and hybrid_result.query_text must match."
            )

        mentions = self.catalog.match_query(
            cleaned_query
        )

        in_corpus_author_mentions = tuple(
            mention
            for mention in mentions
            if mention.in_corpus
            and mention.kind is ScopeEntityKind.AUTHOR
        )

        in_corpus_work_mentions = tuple(
            mention
            for mention in mentions
            if mention.in_corpus
            and mention.kind is ScopeEntityKind.WORK
        )

        out_of_corpus_mentions = tuple(
            mention
            for mention in mentions
            if not mention.in_corpus
        )

        requested_author_ids = tuple(
            mention.entity_id
            for mention in in_corpus_author_mentions
        )

        requested_work_ids = tuple(
            mention.entity_id
            for mention in in_corpus_work_mentions
        )

        if in_corpus_work_mentions:
            requested_version_ids = tuple(
                dict.fromkeys(
                    version_id
                    for mention in in_corpus_work_mentions
                    for version_id in mention.version_ids
                )
            )
        else:
            requested_version_ids = tuple(
                dict.fromkeys(
                    version_id
                    for mention in in_corpus_author_mentions
                    for version_id in mention.version_ids
                )
            )

        evaluated_hits = hybrid_result.hits[:10]

        retrieved_version_ids = tuple(
            hit.version_id
            for hit in evaluated_hits
        )

        if requested_version_ids:
            matching_hit_count = sum(
                version_id in requested_version_ids
                for version_id in retrieved_version_ids
            )

            matching_hit_rate: float | None = (
                matching_hit_count
                / len(retrieved_version_ids)
                if retrieved_version_ids
                else 0.0
            )
        else:
            matching_hit_count = 0
            matching_hit_rate = None

        top_hit_version_id = (
            retrieved_version_ids[0]
            if retrieved_version_ids
            else None
        )

        if (
            requested_version_ids
            and top_hit_version_id is not None
        ):
            top_hit_matches: bool | None = (
                top_hit_version_id
                in requested_version_ids
            )
        else:
            top_hit_matches = None

        return ScopeEvidence(
            query_text=cleaned_query,
            mentions=mentions,
            explicit_scope=bool(mentions),
            in_corpus_scope_mentioned=any(
                mention.in_corpus
                for mention in mentions
            ),
            known_out_of_corpus_scope_mentioned=bool(
                out_of_corpus_mentions
            ),
            requested_author_ids=requested_author_ids,
            requested_work_ids=requested_work_ids,
            known_out_of_corpus_entity_ids=tuple(
                mention.entity_id
                for mention in out_of_corpus_mentions
            ),
            requested_version_ids=requested_version_ids,
            evaluated_hit_count=len(
                retrieved_version_ids
            ),
            retrieved_version_ids_at_10=(
                retrieved_version_ids
            ),
            matching_hit_count_at_10=(
                matching_hit_count
            ),
            matching_hit_rate_at_10=(
                matching_hit_rate
            ),
            top_hit_version_id=(
                top_hit_version_id
            ),
            top_hit_matches_requested_scope=(
                top_hit_matches
            ),
            source_attribution_conflict=(
                top_hit_matches is False
            ),
        )


__all__ = [
    "ScopeEvidenceExtractor",
]
