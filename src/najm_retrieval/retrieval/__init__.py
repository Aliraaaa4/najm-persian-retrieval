"""Retrieval indexes, models, and search utilities."""

from najm_retrieval.retrieval.dense_index import (
    DenseIndex,
    DenseIndexError,
)
from najm_retrieval.retrieval.dense_models import (
    DENSE_INDEX_SCHEMA_VERSION,
    DenseSearchHit,
    DenseSearchResult,
)
from najm_retrieval.retrieval.hybrid import (
    HybridRetriever,
    HybridRetrieverError,
)
from najm_retrieval.retrieval.hybrid_models import (
    HYBRID_RETRIEVAL_SCHEMA_VERSION,
    HybridSearchHit,
    HybridSearchResult,
)
from najm_retrieval.retrieval.lexical_index import (
    LexicalIndex,
    LexicalIndexError,
    build_lexical_index,
)
from najm_retrieval.retrieval.models import (
    LEXICAL_INDEX_SCHEMA_VERSION,
    LexicalIndexBuildReport,
    LexicalSearchMode,
    LexicalSearchResult,
    SearchHit,
)


__all__ = [
    "DENSE_INDEX_SCHEMA_VERSION",
    "HYBRID_RETRIEVAL_SCHEMA_VERSION",
    "LEXICAL_INDEX_SCHEMA_VERSION",
    "DenseIndex",
    "DenseIndexError",
    "DenseSearchHit",
    "DenseSearchResult",
    "HybridRetriever",
    "HybridRetrieverError",
    "HybridSearchHit",
    "HybridSearchResult",
    "LexicalIndex",
    "LexicalIndexBuildReport",
    "LexicalIndexError",
    "LexicalSearchMode",
    "LexicalSearchResult",
    "SearchHit",
    "build_lexical_index",
]
