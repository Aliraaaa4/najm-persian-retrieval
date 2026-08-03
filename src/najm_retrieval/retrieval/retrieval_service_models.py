"""Display-ready models produced by the retrieval service."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from najm_retrieval.retrieval.policy_models import (
    AbstentionReason,
    DecisionAction,
)


RETRIEVAL_SERVICE_SCHEMA_VERSION = "1.0.0"


def _require_non_empty_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


def _validate_optional_rank(
    value: int | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
        or value < 1
    ):
        raise ValueError(
            f"{field_name} must be null or at least 1."
        )


def _validate_optional_score(
    value: float | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return

    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
        or not isfinite(
            float(value)
        )
    ):
        raise ValueError(
            f"{field_name} must be null or finite."
        )


def _validate_string_tuple(
    value: tuple[str, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(
        value,
        tuple,
    ):
        raise TypeError(
            f"{field_name} must be a tuple."
        )

    for item in value:
        _require_non_empty_text(
            item,
            field_name=field_name,
        )


@dataclass(frozen=True)
class RetrievedPassage:
    """One enriched retrieval hit suitable for display or an API."""

    passage_id: str
    version_id: str

    author_id: str
    author_name: str

    work_id: str
    work_title: str

    profile: str
    kind: str
    ordinal: int

    display_text: str
    snippet: str

    heading_path: tuple[str, ...]
    section_path: tuple[str, ...]

    previous_passage_id: str | None
    next_passage_id: str | None

    rank: int
    fusion_score: float

    lexical_rank: int | None
    dense_rank: int | None

    lexical_bm25_score: float | None
    dense_cosine_score: float | None

    schema_version: str = (
        RETRIEVAL_SERVICE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        """Validate the public retrieval-result contract."""

        for field_name in (
            "passage_id",
            "version_id",
            "author_id",
            "author_name",
            "work_id",
            "work_title",
            "profile",
            "kind",
            "display_text",
            "snippet",
        ):
            _require_non_empty_text(
                getattr(
                    self,
                    field_name,
                ),
                field_name=field_name,
            )

        if (
            self.schema_version
            != RETRIEVAL_SERVICE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported retrieval service "
                f"schema version: {self.schema_version}"
            )

        if self.ordinal < 1:
            raise ValueError(
                "ordinal must be at least 1."
            )

        if self.rank < 1:
            raise ValueError(
                "rank must be at least 1."
            )

        if not isfinite(
            float(
                self.fusion_score
            )
        ):
            raise ValueError(
                "fusion_score must be finite."
            )

        _validate_optional_rank(
            self.lexical_rank,
            field_name="lexical_rank",
        )

        _validate_optional_rank(
            self.dense_rank,
            field_name="dense_rank",
        )

        _validate_optional_score(
            self.lexical_bm25_score,
            field_name="lexical_bm25_score",
        )

        _validate_optional_score(
            self.dense_cosine_score,
            field_name="dense_cosine_score",
        )

        if (
            self.lexical_rank is None
        ) != (
            self.lexical_bm25_score is None
        ):
            raise ValueError(
                "lexical_rank and "
                "lexical_bm25_score must both "
                "be present or both be null."
            )

        if (
            self.dense_rank is None
        ) != (
            self.dense_cosine_score is None
        ):
            raise ValueError(
                "dense_rank and "
                "dense_cosine_score must both "
                "be present or both be null."
            )

        for field_name in (
            "previous_passage_id",
            "next_passage_id",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value is not None:
                _require_non_empty_text(
                    value,
                    field_name=field_name,
                )

        _validate_string_tuple(
            self.heading_path,
            field_name="heading_path",
        )

        _validate_string_tuple(
            self.section_path,
            field_name="section_path",
        )


@dataclass(frozen=True)
class TrustedRetrievalResponse:
    """One policy-aware response with public and diagnostic hits."""

    query_text: str

    action: DecisionAction
    reason: AbstentionReason
    return_results: bool

    top_passage_id: str | None
    triggered_reasons: tuple[
        AbstentionReason,
        ...,
    ]

    passages: tuple[
        RetrievedPassage,
        ...,
    ]

    diagnostic_passages: tuple[
        RetrievedPassage,
        ...,
    ]

    retrieval_latency_ms: float

    schema_version: str = (
        RETRIEVAL_SERVICE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        """Validate public versus diagnostic result visibility."""

        _require_non_empty_text(
            self.query_text,
            field_name="query_text",
        )

        if (
            self.schema_version
            != RETRIEVAL_SERVICE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported retrieval service "
                f"schema version: {self.schema_version}"
            )

        if (
            not isinstance(
                self.return_results,
                bool,
            )
        ):
            raise TypeError(
                "return_results must be boolean."
            )

        if (
            not isinstance(
                self.passages,
                tuple,
            )
            or not isinstance(
                self.diagnostic_passages,
                tuple,
            )
            or not isinstance(
                self.triggered_reasons,
                tuple,
            )
        ):
            raise TypeError(
                "Response collections must be tuples."
            )

        if (
            not isinstance(
                self.retrieval_latency_ms,
                (int, float),
            )
            or isinstance(
                self.retrieval_latency_ms,
                bool,
            )
            or not isfinite(
                float(
                    self.retrieval_latency_ms
                )
            )
            or self.retrieval_latency_ms < 0
        ):
            raise ValueError(
                "retrieval_latency_ms must "
                "be finite and non-negative."
            )

        public_ids = tuple(
            passage.passage_id
            for passage in self.passages
        )

        diagnostic_ids = tuple(
            passage.passage_id
            for passage
            in self.diagnostic_passages
        )

        if len(
            set(
                public_ids
            )
        ) != len(
            public_ids
        ):
            raise ValueError(
                "Public passages contain "
                "duplicate passage IDs."
            )

        if len(
            set(
                diagnostic_ids
            )
        ) != len(
            diagnostic_ids
        ):
            raise ValueError(
                "Diagnostic passages contain "
                "duplicate passage IDs."
            )

        missing_public_ids = (
            set(
                public_ids
            )
            - set(
                diagnostic_ids
            )
        )

        if missing_public_ids:
            raise ValueError(
                "Public passages must be "
                "contained in diagnostic passages."
            )

        if self.return_results:
            if not self.passages:
                raise ValueError(
                    "A return-results response "
                    "must contain public passages."
                )

            if (
                self.top_passage_id
                != self.passages[
                    0
                ].passage_id
            ):
                raise ValueError(
                    "top_passage_id must match "
                    "the first public passage."
                )

        elif self.passages:
            raise ValueError(
                "An abstention response must "
                "not expose public passages."
            )

        if self.top_passage_id is not None:
            _require_non_empty_text(
                self.top_passage_id,
                field_name="top_passage_id",
            )

            if not self.diagnostic_passages:
                raise ValueError(
                    "top_passage_id requires "
                    "a diagnostic passage."
                )

            if (
                self.top_passage_id
                != self.diagnostic_passages[
                    0
                ].passage_id
            ):
                raise ValueError(
                    "top_passage_id must match "
                    "the first diagnostic passage."
                )


__all__ = [
    "RETRIEVAL_SERVICE_SCHEMA_VERSION",
    "RetrievedPassage",
    "TrustedRetrievalResponse",
]
