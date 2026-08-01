"""Retrieval evaluation models and utilities."""

from najm_retrieval.evaluation.io import (
    EvaluationDataError,
    load_evaluation_jsonl,
    query_to_payload,
    write_evaluation_jsonl,
)
from najm_retrieval.evaluation.models import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationSet,
    QueryType,
    RelevanceJudgment,
    RetrievalQuery,
)


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationDataError",
    "EvaluationSet",
    "QueryType",
    "RelevanceJudgment",
    "RetrievalQuery",
    "load_evaluation_jsonl",
    "query_to_payload",
    "write_evaluation_jsonl",
]