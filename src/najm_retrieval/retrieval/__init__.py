"""Retrieval indexes, models, and search utilities."""

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
    "LEXICAL_INDEX_SCHEMA_VERSION",
    "LexicalIndex",
    "LexicalIndexBuildReport",
    "LexicalIndexError",
    "LexicalSearchMode",
    "LexicalSearchResult",
    "SearchHit",
    "build_lexical_index",
]