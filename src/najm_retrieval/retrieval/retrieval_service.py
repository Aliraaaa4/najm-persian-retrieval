"""Combine trusted retrieval hits with stored passage content."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from najm_retrieval.retrieval.hybrid_models import (
    HybridSearchHit,
)
from najm_retrieval.retrieval.models import (
    LexicalSearchMode,
)
from najm_retrieval.retrieval.passage_store_models import (
    PassageStoreRecord,
)
from najm_retrieval.retrieval.retrieval_service_models import (
    RetrievedPassage,
    TrustedRetrievalResponse,
)
from najm_retrieval.retrieval.trusted_models import (
    TrustedRetrievalResult,
)


class RetrievalServiceError(
    RuntimeError
):
    """Raised when retrieval hits cannot be safely enriched."""


class _TrustedRetrieverProtocol(
    Protocol
):
    def search(
        self,
        query_text: str,
        *,
        lexical_mode: LexicalSearchMode = (
            LexicalSearchMode.AUTO
        ),
    ) -> TrustedRetrievalResult:
        """Run one trusted retrieval request."""


class _PassageStoreProtocol(
    Protocol
):
    def get_many(
        self,
        passage_ids: Iterable[str],
    ) -> tuple[
        PassageStoreRecord,
        ...,
    ]:
        """Return stored passages for the requested IDs."""


class RetrievalService:
    """Return display-ready trusted retrieval responses."""

    def __init__(
        self,
        trusted_retriever: (
            _TrustedRetrieverProtocol
        ),
        passage_store: (
            _PassageStoreProtocol
        ),
        *,
        snippet_chars: int = 280,
    ) -> None:
        if (
            not isinstance(
                snippet_chars,
                int,
            )
            or isinstance(
                snippet_chars,
                bool,
            )
            or snippet_chars < 1
        ):
            raise ValueError(
                "snippet_chars must be "
                "an integer of at least 1."
            )

        self._trusted_retriever = (
            trusted_retriever
        )

        self._passage_store = (
            passage_store
        )

        self._snippet_chars = (
            snippet_chars
        )

    def search(
        self,
        query_text: str,
        *,
        lexical_mode: LexicalSearchMode = (
            LexicalSearchMode.AUTO
        ),
    ) -> TrustedRetrievalResponse:
        """Retrieve, apply policy, and attach passage content."""

        if (
            not isinstance(
                query_text,
                str,
            )
            or not query_text.strip()
        ):
            raise ValueError(
                "query_text must be a "
                "non-empty string."
            )

        trusted_result = (
            self._trusted_retriever.search(
                query_text,
                lexical_mode=lexical_mode,
            )
        )

        diagnostic_hits = tuple(
            trusted_result.diagnostic_hits
        )

        self._validate_unique_hits(
            diagnostic_hits
        )

        records = (
            self._passage_store.get_many(
                hit.passage_id
                for hit in diagnostic_hits
            )
        )

        records_by_id: dict[
            str,
            PassageStoreRecord,
        ] = {}

        for record in records:
            if (
                record.passage_id
                in records_by_id
            ):
                raise RetrievalServiceError(
                    "Passage store returned "
                    "duplicate record: "
                    f"{record.passage_id}"
                )

            records_by_id[
                record.passage_id
            ] = record

        missing_ids = tuple(
            hit.passage_id
            for hit in diagnostic_hits
            if (
                hit.passage_id
                not in records_by_id
            )
        )

        if missing_ids:
            formatted = ", ".join(
                missing_ids
            )

            raise RetrievalServiceError(
                "Passage store is missing "
                f"retrieval hits: {formatted}"
            )

        diagnostic_passages = tuple(
            self._enrich_hit(
                hit,
                records_by_id[
                    hit.passage_id
                ],
            )
            for hit in diagnostic_hits
        )

        diagnostics_by_id = {
            passage.passage_id: passage
            for passage
            in diagnostic_passages
        }

        public_hits = tuple(
            trusted_result.hits
        )

        self._validate_unique_hits(
            public_hits
        )

        unknown_public_ids = tuple(
            hit.passage_id
            for hit in public_hits
            if (
                hit.passage_id
                not in diagnostics_by_id
            )
        )

        if unknown_public_ids:
            formatted = ", ".join(
                unknown_public_ids
            )

            raise RetrievalServiceError(
                "Public retrieval hits are "
                "missing from diagnostic hits: "
                f"{formatted}"
            )

        public_passages = tuple(
            diagnostics_by_id[
                hit.passage_id
            ]
            for hit in public_hits
        )

        decision = (
            trusted_result.decision
        )

        return TrustedRetrievalResponse(
            query_text=(
                trusted_result.query_text
            ),
            action=decision.action,
            reason=decision.reason,
            return_results=(
                decision.return_results
            ),
            top_passage_id=(
                decision.top_passage_id
            ),
            triggered_reasons=(
                decision.triggered_reasons
            ),
            passages=public_passages,
            diagnostic_passages=(
                diagnostic_passages
            ),
            retrieval_latency_ms=(
                trusted_result
                .retrieval_run
                .hybrid_result
                .latency_ms
            ),
        )

    def _enrich_hit(
        self,
        hit: HybridSearchHit,
        record: PassageStoreRecord,
    ) -> RetrievedPassage:
        if (
            record.version_id
            != hit.version_id
        ):
            raise RetrievalServiceError(
                "Version mismatch for "
                f"{hit.passage_id}: "
                f"hit={hit.version_id!r}, "
                f"store={record.version_id!r}."
            )

        if record.kind != hit.kind:
            raise RetrievalServiceError(
                "Passage kind mismatch for "
                f"{hit.passage_id}: "
                f"hit={hit.kind!r}, "
                f"store={record.kind!r}."
            )

        return RetrievedPassage(
            passage_id=record.passage_id,
            version_id=record.version_id,
            author_id=record.author_id,
            author_name=record.author_name,
            work_id=record.work_id,
            work_title=record.work_title,
            profile=record.profile,
            kind=record.kind,
            ordinal=record.ordinal,
            display_text=(
                record.display_text
            ),
            snippet=record.snippet(
                max_chars=(
                    self._snippet_chars
                )
            ),
            heading_path=(
                record.heading_path
            ),
            section_path=(
                record.section_path
            ),
            previous_passage_id=(
                record.previous_passage_id
            ),
            next_passage_id=(
                record.next_passage_id
            ),
            rank=hit.rank,
            fusion_score=(
                hit.fusion_score
            ),
            lexical_rank=(
                hit.lexical_rank
            ),
            dense_rank=(
                hit.dense_rank
            ),
            lexical_bm25_score=(
                hit.lexical_bm25_score
            ),
            dense_cosine_score=(
                hit.dense_cosine_score
            ),
        )

    @staticmethod
    def _validate_unique_hits(
        hits: tuple[
            HybridSearchHit,
            ...,
        ],
    ) -> None:
        passage_ids = tuple(
            hit.passage_id
            for hit in hits
        )

        if len(
            set(
                passage_ids
            )
        ) != len(
            passage_ids
        ):
            raise RetrievalServiceError(
                "Retrieval hits contain "
                "duplicate passage IDs."
            )


__all__ = [
    "RetrievalService",
    "RetrievalServiceError",
]
