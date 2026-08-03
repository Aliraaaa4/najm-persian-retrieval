"""REST API for the NAJM Persian retrieval service."""

from najm_retrieval.api.app import (
    app,
    create_app,
)
from najm_retrieval.api.query_suggestions import (
    QuerySuggestion,
    QuerySuggestionEngine,
)
from najm_retrieval.api.runtime import (
    ApiSettings,
    build_query_suggestion_engine,
    build_retrieval_service,
)


__all__ = [
    "ApiSettings",
    "QuerySuggestion",
    "QuerySuggestionEngine",
    "app",
    "build_query_suggestion_engine",
    "build_retrieval_service",
    "create_app",
]
