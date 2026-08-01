"""Retrieval evaluation models and utilities."""

from najm_retrieval.evaluation.io import (
    EvaluationDataError,
    load_evaluation_jsonl,
    query_to_payload,
    write_evaluation_jsonl,
)
from najm_retrieval.evaluation.metrics import (
    DEFAULT_CUTOFFS,
    AggregateCutoffMetrics,
    QueryCutoffMetrics,
    QueryEvaluationResult,
    RetrievalEvaluationReport,
    evaluate_rankings,
    ndcg_at_k,
    negative_hit_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from najm_retrieval.evaluation.models import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationSet,
    QueryType,
    RelevanceJudgment,
    RetrievalQuery,
)


__all__ = [
    "DEFAULT_CUTOFFS",
    "EVALUATION_SCHEMA_VERSION",
    "AggregateCutoffMetrics",
    "EvaluationDataError",
    "EvaluationSet",
    "QueryCutoffMetrics",
    "QueryEvaluationResult",
    "QueryType",
    "RelevanceJudgment",
    "RetrievalEvaluationReport",
    "RetrievalQuery",
    "evaluate_rankings",
    "load_evaluation_jsonl",
    "ndcg_at_k",
    "negative_hit_at_k",
    "query_to_payload",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "write_evaluation_jsonl",
]