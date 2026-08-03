"""REST API for the NAJM Persian retrieval service."""

from najm_retrieval.api.app import (
    app,
    create_app,
)
from najm_retrieval.api.runtime import (
    ApiSettings,
    build_retrieval_service,
)


__all__ = [
    "ApiSettings",
    "app",
    "build_retrieval_service",
    "create_app",
]
