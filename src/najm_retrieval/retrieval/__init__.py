"""Retrieval indexes, models, and search utilities."""

from najm_retrieval.retrieval.scope_catalog import (
    CorpusScopeCatalog,
    ScopeCatalogError,
    normalize_scope_text,
)
from najm_retrieval.retrieval.scope_features import (
    ScopeEvidenceExtractor,
)
from najm_retrieval.retrieval.scope_models import (
    SCOPE_CATALOG_SCHEMA_VERSION,
    ScopeCatalogEntity,
    ScopeEntityKind,
    ScopeEvidence,
    ScopeMention,
)

from najm_retrieval.retrieval.abstention_features import (
    AbstentionFeatureExtractor,
)
from najm_retrieval.retrieval.abstention_models import (
    ABSTENTION_FEATURE_SCHEMA_VERSION,
    AbstentionFeatures,
)

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
    "CorpusScopeCatalog",
    "SCOPE_CATALOG_SCHEMA_VERSION",
    "ScopeCatalogEntity",
    "ScopeCatalogError",
    "ScopeEntityKind",
    "ScopeEvidence",
    "ScopeEvidenceExtractor",
    "ScopeMention",
    "normalize_scope_text",
    "ABSTENTION_FEATURE_SCHEMA_VERSION",
    "AbstentionFeatureExtractor",
    "AbstentionFeatures",
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
