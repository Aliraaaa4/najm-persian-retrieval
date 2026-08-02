"""Extract interpretable abstention evidence from retrieval results."""

from __future__ import annotations

import math

from najm_retrieval.retrieval.abstention_models import (
    AbstentionFeatures,
)
from najm_retrieval.retrieval.dense_models import (
    DenseSearchHit,
    DenseSearchResult,
)
from najm_retrieval.retrieval.hybrid_models import (
    HybridSearchHit,
    HybridSearchResult,
)
from najm_retrieval.retrieval.models import (
    LexicalSearchResult,
    SearchHit,
)


class AbstentionFeatureExtractor:
    """Create decision-free evidence from component rankings."""

    def extract(
        self,
        *,
        lexical_result: LexicalSearchResult | None,
        dense_result: DenseSearchResult,
        hybrid_result: HybridSearchResult,
    ) -> AbstentionFeatures:
        """Return validated features for one aligned query."""

        self._validate_query_alignment(
            lexical_result=lexical_result,
            dense_result=dense_result,
            hybrid_result=hybrid_result,
        )

        lexical_hits = (
            lexical_result.hits
            if lexical_result is not None
            else ()
        )
        dense_hits = dense_result.hits
        hybrid_hits = hybrid_result.hits

        lexical_ids = self._ranked_ids(
            lexical_hits,
            label="lexical",
        )
        dense_ids = self._ranked_ids(
            dense_hits,
            label="dense",
        )
        self._ranked_ids(
            hybrid_hits,
            label="hybrid",
        )

        self._validate_hybrid_components(
            lexical_result=lexical_result,
            dense_result=dense_result,
            hybrid_result=hybrid_result,
        )

        lexical_top_1 = (
            lexical_hits[0]
            if lexical_hits
            else None
        )
        dense_top_1 = (
            dense_hits[0]
            if dense_hits
            else None
        )
        dense_top_2 = (
            dense_hits[1]
            if len(dense_hits) >= 2
            else None
        )
        hybrid_top_1 = (
            hybrid_hits[0]
            if hybrid_hits
            else None
        )
        hybrid_top_2 = (
            hybrid_hits[1]
            if len(hybrid_hits) >= 2
            else None
        )

        dense_margin = (
            dense_top_1.cosine_score
            - dense_top_2.cosine_score
            if dense_top_1 is not None
            and dense_top_2 is not None
            else None
        )

        hybrid_margin = (
            hybrid_top_1.fusion_score
            - hybrid_top_2.fusion_score
            if hybrid_top_1 is not None
            and hybrid_top_2 is not None
            else None
        )

        overlap_at_10 = len(
            set(lexical_ids[:10])
            & set(dense_ids[:10])
        )
        overlap_at_100 = len(
            set(lexical_ids[:100])
            & set(dense_ids[:100])
        )

        return AbstentionFeatures(
            query_text=dense_result.query_text,
            lexical_result_available=(
                lexical_result is not None
            ),
            lexical_mode_used=(
                lexical_result.mode_used
                if lexical_result is not None
                else None
            ),
            lexical_hit_count=len(lexical_hits),
            dense_hit_count=len(dense_hits),
            hybrid_hit_count=len(hybrid_hits),
            lexical_top_1_passage_id=(
                lexical_top_1.passage_id
                if lexical_top_1 is not None
                else None
            ),
            dense_top_1_passage_id=(
                dense_top_1.passage_id
                if dense_top_1 is not None
                else None
            ),
            hybrid_top_1_passage_id=(
                hybrid_top_1.passage_id
                if hybrid_top_1 is not None
                else None
            ),
            lexical_top_1_bm25=(
                lexical_top_1.bm25_score
                if lexical_top_1 is not None
                else None
            ),
            dense_top_1_score=(
                dense_top_1.cosine_score
                if dense_top_1 is not None
                else None
            ),
            dense_top_2_score=(
                dense_top_2.cosine_score
                if dense_top_2 is not None
                else None
            ),
            dense_margin_1_2=dense_margin,
            overlap_at_10=overlap_at_10,
            overlap_at_100=overlap_at_100,
            top_1_same_passage=bool(
                lexical_top_1 is not None
                and dense_top_1 is not None
                and lexical_top_1.passage_id
                == dense_top_1.passage_id
            ),
            hybrid_top_1_score=(
                hybrid_top_1.fusion_score
                if hybrid_top_1 is not None
                else None
            ),
            hybrid_top_2_score=(
                hybrid_top_2.fusion_score
                if hybrid_top_2 is not None
                else None
            ),
            hybrid_margin_1_2=hybrid_margin,
            hybrid_top_1_lexical_rank=(
                hybrid_top_1.lexical_rank
                if hybrid_top_1 is not None
                else None
            ),
            hybrid_top_1_dense_rank=(
                hybrid_top_1.dense_rank
                if hybrid_top_1 is not None
                else None
            ),
            hybrid_top_1_dual_supported=bool(
                hybrid_top_1 is not None
                and hybrid_top_1.lexical_rank is not None
                and hybrid_top_1.dense_rank is not None
            ),
        )

    @staticmethod
    def _validate_query_alignment(
        *,
        lexical_result: LexicalSearchResult | None,
        dense_result: DenseSearchResult,
        hybrid_result: HybridSearchResult,
    ) -> None:
        """Require component results from the same query."""

        query_texts = {
            dense_result.query_text,
            hybrid_result.query_text,
        }

        if lexical_result is not None:
            query_texts.add(
                lexical_result.query_text
            )

        if len(query_texts) != 1:
            raise ValueError(
                "Lexical, dense, and hybrid results "
                "must use the same query_text."
            )

    @staticmethod
    def _ranked_ids(
        hits: tuple[
            SearchHit
            | DenseSearchHit
            | HybridSearchHit,
            ...,
        ],
        *,
        label: str,
    ) -> tuple[str, ...]:
        """Return passage IDs and reject duplicates."""

        passage_ids = tuple(
            hit.passage_id
            for hit in hits
        )

        if len(passage_ids) != len(
            set(passage_ids)
        ):
            raise ValueError(
                f"{label} ranking contains duplicate passage IDs."
            )

        return passage_ids

    def _validate_hybrid_components(
        self,
        *,
        lexical_result: LexicalSearchResult | None,
        dense_result: DenseSearchResult,
        hybrid_result: HybridSearchResult,
    ) -> None:
        """Verify hybrid component ranks against source rankings."""

        lexical_hits = (
            lexical_result.hits
            if lexical_result is not None
            else ()
        )
        dense_hits = dense_result.hits

        for hybrid_hit in hybrid_result.hits:
            if hybrid_hit.lexical_rank is not None:
                if lexical_result is None:
                    raise ValueError(
                        "Hybrid result contains a lexical rank "
                        "but no lexical result was provided."
                    )

                lexical_rank = hybrid_hit.lexical_rank

                if lexical_rank > len(lexical_hits):
                    raise ValueError(
                        "Hybrid lexical rank exceeds "
                        "the provided lexical depth."
                    )

                lexical_hit = lexical_hits[
                    lexical_rank - 1
                ]

                self._validate_component_hit(
                    component_name="lexical",
                    hybrid_hit=hybrid_hit,
                    passage_id=lexical_hit.passage_id,
                    version_id=lexical_hit.version_id,
                    kind=lexical_hit.kind,
                )

                if (
                    hybrid_hit.lexical_bm25_score is None
                    or not math.isclose(
                        hybrid_hit.lexical_bm25_score,
                        lexical_hit.bm25_score,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "Hybrid lexical score does not "
                        "match the source ranking."
                    )
            elif hybrid_hit.lexical_bm25_score is not None:
                raise ValueError(
                    "Hybrid lexical score requires a lexical rank."
                )

            if hybrid_hit.dense_rank is not None:
                dense_rank = hybrid_hit.dense_rank

                if dense_rank > len(dense_hits):
                    raise ValueError(
                        "Hybrid dense rank exceeds "
                        "the provided dense depth."
                    )

                dense_hit = dense_hits[
                    dense_rank - 1
                ]

                self._validate_component_hit(
                    component_name="dense",
                    hybrid_hit=hybrid_hit,
                    passage_id=dense_hit.passage_id,
                    version_id=dense_hit.version_id,
                    kind=dense_hit.kind,
                )

                if (
                    hybrid_hit.dense_cosine_score is None
                    or not math.isclose(
                        hybrid_hit.dense_cosine_score,
                        dense_hit.cosine_score,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "Hybrid dense score does not "
                        "match the source ranking."
                    )
            elif hybrid_hit.dense_cosine_score is not None:
                raise ValueError(
                    "Hybrid dense score requires a dense rank."
                )

    @staticmethod
    def _validate_component_hit(
        *,
        component_name: str,
        hybrid_hit: HybridSearchHit,
        passage_id: str,
        version_id: str,
        kind: str,
    ) -> None:
        """Require identity agreement for one component rank."""

        if (
            hybrid_hit.passage_id != passage_id
            or hybrid_hit.version_id != version_id
            or hybrid_hit.kind != kind
        ):
            raise ValueError(
                "Hybrid and "
                f"{component_name} metadata disagree "
                f"for passage ID {hybrid_hit.passage_id}."
            )


__all__ = [
    "AbstentionFeatureExtractor",
]
