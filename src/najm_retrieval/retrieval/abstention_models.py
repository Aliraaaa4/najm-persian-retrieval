"""Models for retrieval evidence used by abstention policies."""

from __future__ import annotations

from dataclasses import dataclass
import math

from najm_retrieval.retrieval.models import LexicalSearchMode


ABSTENTION_FEATURE_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class AbstentionFeatures:
    """Interpretable retrieval evidence for one query.

    These values are descriptive features, not calibrated probabilities.
    They intentionally contain no answerable/abstain decision.
    """

    query_text: str

    lexical_result_available: bool
    lexical_mode_used: LexicalSearchMode | None

    lexical_hit_count: int
    dense_hit_count: int
    hybrid_hit_count: int

    lexical_top_1_passage_id: str | None
    dense_top_1_passage_id: str | None
    hybrid_top_1_passage_id: str | None

    lexical_top_1_bm25: float | None

    dense_top_1_score: float | None
    dense_top_2_score: float | None
    dense_margin_1_2: float | None

    overlap_at_10: int
    overlap_at_100: int
    top_1_same_passage: bool

    hybrid_top_1_score: float | None
    hybrid_top_2_score: float | None
    hybrid_margin_1_2: float | None

    hybrid_top_1_lexical_rank: int | None
    hybrid_top_1_dense_rank: int | None
    hybrid_top_1_dual_supported: bool

    schema_version: str = ABSTENTION_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate feature-level invariants."""

        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise ValueError("query_text must not be empty.")

        if self.schema_version != ABSTENTION_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported abstention-feature schema version: "
                f"{self.schema_version}"
            )

        for label, value in (
            ("lexical_hit_count", self.lexical_hit_count),
            ("dense_hit_count", self.dense_hit_count),
            ("hybrid_hit_count", self.hybrid_hit_count),
            ("overlap_at_10", self.overlap_at_10),
            ("overlap_at_100", self.overlap_at_100),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{label} must be a non-negative integer."
                )

        for label, value in (
            ("lexical_top_1_passage_id", self.lexical_top_1_passage_id),
            ("dense_top_1_passage_id", self.dense_top_1_passage_id),
            ("hybrid_top_1_passage_id", self.hybrid_top_1_passage_id),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{label} must be non-empty when provided."
                )

        for label, value in (
            ("lexical_top_1_bm25", self.lexical_top_1_bm25),
            ("dense_top_1_score", self.dense_top_1_score),
            ("dense_top_2_score", self.dense_top_2_score),
            ("dense_margin_1_2", self.dense_margin_1_2),
            ("hybrid_top_1_score", self.hybrid_top_1_score),
            ("hybrid_top_2_score", self.hybrid_top_2_score),
            ("hybrid_margin_1_2", self.hybrid_margin_1_2),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"{label} must be finite when provided."
                )

        for label, value in (
            ("dense_top_1_score", self.dense_top_1_score),
            ("dense_top_2_score", self.dense_top_2_score),
        ):
            if value is not None and not -1.0001 <= value <= 1.0001:
                raise ValueError(
                    f"{label} must be between -1 and 1."
                )

        for label, value in (
            ("dense_margin_1_2", self.dense_margin_1_2),
            ("hybrid_top_1_score", self.hybrid_top_1_score),
            ("hybrid_top_2_score", self.hybrid_top_2_score),
            ("hybrid_margin_1_2", self.hybrid_margin_1_2),
        ):
            if value is not None and value < 0:
                raise ValueError(
                    f"{label} must be non-negative when provided."
                )

        for label, value in (
            (
                "hybrid_top_1_lexical_rank",
                self.hybrid_top_1_lexical_rank,
            ),
            (
                "hybrid_top_1_dense_rank",
                self.hybrid_top_1_dense_rank,
            ),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    f"{label} must be a positive integer when provided."
                )

        if not self.lexical_result_available and (
            self.lexical_mode_used is not None
            or self.lexical_hit_count != 0
            or self.lexical_top_1_passage_id is not None
            or self.lexical_top_1_bm25 is not None
        ):
            raise ValueError(
                "Unavailable lexical results cannot contain lexical evidence."
            )

        if (
            self.lexical_result_available
            and self.lexical_mode_used is None
        ):
            raise ValueError(
                "lexical_mode_used is required when "
                "a lexical result is available."
            )

        self._validate_top_hit(
            label="lexical",
            hit_count=self.lexical_hit_count,
            passage_id=self.lexical_top_1_passage_id,
            top_score=self.lexical_top_1_bm25,
        )
        self._validate_top_hit(
            label="dense",
            hit_count=self.dense_hit_count,
            passage_id=self.dense_top_1_passage_id,
            top_score=self.dense_top_1_score,
        )
        self._validate_top_hit(
            label="hybrid",
            hit_count=self.hybrid_hit_count,
            passage_id=self.hybrid_top_1_passage_id,
            top_score=self.hybrid_top_1_score,
        )

        if self.dense_hit_count < 2:
            if (
                self.dense_top_2_score is not None
                or self.dense_margin_1_2 is not None
            ):
                raise ValueError(
                    "Dense top-2 evidence requires at least two dense hits."
                )
        elif (
            self.dense_top_2_score is None
            or self.dense_margin_1_2 is None
        ):
            raise ValueError(
                "Two or more dense hits require dense top-2 evidence."
            )

        if self.hybrid_hit_count < 2:
            if (
                self.hybrid_top_2_score is not None
                or self.hybrid_margin_1_2 is not None
            ):
                raise ValueError(
                    "Hybrid top-2 evidence requires at least two hybrid hits."
                )
        elif (
            self.hybrid_top_2_score is None
            or self.hybrid_margin_1_2 is None
        ):
            raise ValueError(
                "Two or more hybrid hits require hybrid top-2 evidence."
            )

        if self.overlap_at_10 > min(
            self.lexical_hit_count,
            self.dense_hit_count,
            10,
        ):
            raise ValueError(
                "overlap_at_10 exceeds the available ranking depth."
            )

        if self.overlap_at_100 > min(
            self.lexical_hit_count,
            self.dense_hit_count,
            100,
        ):
            raise ValueError(
                "overlap_at_100 exceeds the available ranking depth."
            )

        if self.overlap_at_10 > self.overlap_at_100:
            raise ValueError(
                "overlap_at_10 cannot exceed overlap_at_100."
            )

        expected_same_top_1 = bool(
            self.lexical_top_1_passage_id is not None
            and self.dense_top_1_passage_id is not None
            and self.lexical_top_1_passage_id
            == self.dense_top_1_passage_id
        )

        if self.top_1_same_passage != expected_same_top_1:
            raise ValueError(
                "top_1_same_passage does not match the component top hits."
            )

        expected_dual_support = bool(
            self.hybrid_top_1_lexical_rank is not None
            and self.hybrid_top_1_dense_rank is not None
        )

        if (
            self.hybrid_top_1_dual_supported
            != expected_dual_support
        ):
            raise ValueError(
                "hybrid_top_1_dual_supported does not "
                "match the component ranks."
            )

        if self.hybrid_hit_count == 0 and (
            self.hybrid_top_1_lexical_rank is not None
            or self.hybrid_top_1_dense_rank is not None
        ):
            raise ValueError(
                "An empty hybrid result cannot contain component ranks."
            )

    @staticmethod
    def _validate_top_hit(
        *,
        label: str,
        hit_count: int,
        passage_id: str | None,
        top_score: float | None,
    ) -> None:
        """Validate top-hit presence against a hit count."""

        if hit_count == 0:
            if passage_id is not None or top_score is not None:
                raise ValueError(
                    f"Empty {label} results cannot "
                    "contain top-hit evidence."
                )
            return

        if passage_id is None or top_score is None:
            raise ValueError(
                f"Non-empty {label} results require top-hit evidence."
            )


__all__ = [
    "ABSTENTION_FEATURE_SCHEMA_VERSION",
    "AbstentionFeatures",
]
