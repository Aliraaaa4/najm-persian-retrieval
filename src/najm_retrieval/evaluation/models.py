"""Models for passage-level retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


EVALUATION_SCHEMA_VERSION = "1.0.0"

_QUERY_ID_PATTERN = re.compile(
    r"^[a-z][a-z0-9_-]*$"
)


class QueryType(str, Enum):
    """Supported retrieval-evaluation query categories."""

    EXACT_QUOTE = "exact_quote"
    PARTIAL_QUOTE = "partial_quote"
    SEMANTIC = "semantic"
    ENTITY = "entity"
    ORTHOGRAPHIC_VARIANT = (
        "orthographic_variant"
    )
    OUT_OF_CORPUS = "out_of_corpus"


@dataclass(frozen=True)
class RelevanceJudgment:
    """Graded relevance for one passage."""

    passage_id: str
    grade: int = 1
    rationale: str | None = None

    def __post_init__(self) -> None:
        """Validate passage identity and relevance grade."""

        if not self.passage_id.strip():
            raise ValueError(
                "passage_id must not be empty."
            )

        if (
            not isinstance(self.grade, int)
            or isinstance(self.grade, bool)
            or not 1 <= self.grade <= 3
        ):
            raise ValueError(
                "grade must be an integer "
                "between 1 and 3."
            )

        if (
            self.rationale is not None
            and not self.rationale.strip()
        ):
            raise ValueError(
                "rationale must be nonempty "
                "when provided."
            )


@dataclass(frozen=True)
class RetrievalQuery:
    """One human-reviewable retrieval query."""

    query_id: str
    query_text: str
    query_type: QueryType

    judgments: tuple[
        RelevanceJudgment,
        ...,
    ] = field(
        default_factory=tuple
    )

    expected_version_ids: tuple[
        str,
        ...,
    ] = field(
        default_factory=tuple
    )

    tags: tuple[
        str,
        ...,
    ] = field(
        default_factory=tuple
    )

    include_in_metrics: bool = True
    notes: str | None = None

    schema_version: str = (
        EVALUATION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        """Validate the query and its relevance labels."""

        if not _QUERY_ID_PATTERN.fullmatch(
            self.query_id
        ):
            raise ValueError(
                "query_id must start with a "
                "lowercase letter and contain only "
                "lowercase letters, digits, "
                "underscores, or hyphens."
            )

        if not self.query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        if not isinstance(
            self.include_in_metrics,
            bool,
        ):
            raise ValueError(
                "include_in_metrics must be Boolean."
            )

        passage_ids = [
            judgment.passage_id
            for judgment in self.judgments
        ]

        if len(passage_ids) != len(
            set(passage_ids)
        ):
            raise ValueError(
                "A query cannot contain duplicate "
                "passage judgments."
            )

        if len(
            self.expected_version_ids
        ) != len(
            set(self.expected_version_ids)
        ):
            raise ValueError(
                "expected_version_ids must be unique."
            )

        if len(self.tags) != len(
            set(self.tags)
        ):
            raise ValueError(
                "Query tags must be unique."
            )

        if (
            self.query_type
            is QueryType.OUT_OF_CORPUS
            and self.judgments
        ):
            raise ValueError(
                "An out-of-corpus query cannot have "
                "relevant passage judgments."
            )

        if (
            self.query_type
            is not QueryType.OUT_OF_CORPUS
            and self.include_in_metrics
            and not self.judgments
        ):
            raise ValueError(
                "An answerable metric query must "
                "have at least one relevance judgment."
            )

        if (
            self.notes is not None
            and not self.notes.strip()
        ):
            raise ValueError(
                "notes must be nonempty "
                "when provided."
            )

        if (
            self.schema_version
            != EVALUATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported evaluation "
                f"schema version: "
                f"{self.schema_version}"
            )

    @property
    def relevant_passage_ids(
        self,
    ) -> tuple[str, ...]:
        """Return all judged-relevant passage IDs."""

        return tuple(
            judgment.passage_id
            for judgment in self.judgments
        )

    @property
    def is_answerable(self) -> bool:
        """Return whether the corpus should contain an answer."""

        return (
            self.query_type
            is not QueryType.OUT_OF_CORPUS
        )


@dataclass(frozen=True)
class EvaluationSet:
    """Validated collection of retrieval queries."""

    queries: tuple[
        RetrievalQuery,
        ...,
    ]

    schema_version: str = (
        EVALUATION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        """Validate collection-level invariants."""

        if (
            self.schema_version
            != EVALUATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported evaluation-set "
                f"schema version: "
                f"{self.schema_version}"
            )

        query_ids = [
            query.query_id
            for query in self.queries
        ]

        if len(query_ids) != len(
            set(query_ids)
        ):
            raise ValueError(
                "Evaluation query IDs must be unique."
            )

    @property
    def metric_queries(
        self,
    ) -> tuple[RetrievalQuery, ...]:
        """Return queries included in metric aggregation."""

        return tuple(
            query
            for query in self.queries
            if query.include_in_metrics
        )

    @property
    def answerable_queries(
        self,
    ) -> tuple[RetrievalQuery, ...]:
        """Return answerable metric queries."""

        return tuple(
            query
            for query in self.metric_queries
            if query.is_answerable
        )

    @property
    def out_of_corpus_queries(
        self,
    ) -> tuple[RetrievalQuery, ...]:
        """Return unanswerable metric queries."""

        return tuple(
            query
            for query in self.metric_queries
            if not query.is_answerable
        )


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationSet",
    "QueryType",
    "RelevanceJudgment",
    "RetrievalQuery",
]