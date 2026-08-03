"""Pydantic request and response models for the REST API."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


API_SCHEMA_VERSION = "1.1.0"


class RetrieveRequest(BaseModel):
    """One public retrieval request."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=1000,
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=10,
    )


class HealthResponse(BaseModel):
    """Liveness information."""

    schema_version: str = (
        API_SCHEMA_VERSION
    )

    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness information."""

    schema_version: str = (
        API_SCHEMA_VERSION
    )

    status: str
    ready: bool
    detail: str


class RetrievalScoresResponse(BaseModel):
    """Retriever-specific scores and ranks."""

    fusion_score: float

    lexical_rank: int | None
    dense_rank: int | None

    lexical_bm25_score: float | None
    dense_cosine_score: float | None


class PassageReferenceResponse(BaseModel):
    """Stable source reference for one passage."""

    passage_id: str
    version_id: str

    author_id: str
    author_name: str

    work_id: str
    work_title: str

    profile: str
    kind: str
    ordinal: int

    heading_path: list[str]
    section_path: list[str]

    previous_passage_id: str | None
    next_passage_id: str | None


class RetrievalResultResponse(BaseModel):
    """One public display-ready retrieval result."""

    rank: int

    snippet: str
    display_text: str

    reference: PassageReferenceResponse
    scores: RetrievalScoresResponse


class QuerySuggestionResponse(BaseModel):
    """One public clickable query suggestion."""

    query: str
    label: str
    kind: str

    entity_id: str
    entity_kind: str
    version_ids: list[str]


class RetrieveResponse(BaseModel):
    """Public policy-aware response."""

    schema_version: str = (
        API_SCHEMA_VERSION
    )

    query: str

    action: str
    reason: str
    return_results: bool

    message: str
    top_passage_id: str | None

    triggered_reasons: list[str]

    result_count: int
    results: list[
        RetrievalResultResponse
    ]

    suggestions: list[
        QuerySuggestionResponse
    ] = Field(
        default_factory=list
    )

    retrieval_latency_ms: float


class ErrorDetail(BaseModel):
    """Stable API error payload."""

    code: str
    message: str


__all__ = [
    "API_SCHEMA_VERSION",
    "ErrorDetail",
    "HealthResponse",
    "PassageReferenceResponse",
    "QuerySuggestionResponse",
    "ReadinessResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "RetrievalResultResponse",
    "RetrievalScoresResponse",
]
