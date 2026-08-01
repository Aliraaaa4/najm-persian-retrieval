"""Public text-normalization API."""

from najm_retrieval.normalization.models import (
    NORMALIZATION_VERSION,
    NormalizationChange,
    NormalizationResult,
    NormalizationView,
    ScriptProfile,
)
from najm_retrieval.normalization.unicode import (
    classify_script_profile,
    normalize_display,
    normalize_retrieval,
    normalize_search_alias,
    normalize_text,
)


__all__ = [
    "NORMALIZATION_VERSION",
    "NormalizationChange",
    "NormalizationResult",
    "NormalizationView",
    "ScriptProfile",
    "classify_script_profile",
    "normalize_display",
    "normalize_retrieval",
    "normalize_search_alias",
    "normalize_text",
]
