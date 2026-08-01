"""Text preparation utilities for parsed OpenITI documents."""

from najm_retrieval.text_preparation.assembly import (
    assemble_logical_units,
)
from najm_retrieval.text_preparation.attributes import (
    attributes_to_dict,
)
from najm_retrieval.text_preparation.context_resolver import (
    resolve_logical_unit_contexts,
)
from najm_retrieval.text_preparation.context_models import (
    ContextIssue,
    ContextMarker,
    ContextualLogicalUnit,
    LogicalUnitContext,
    STRUCTURAL_CONTEXT_TYPES,
)
from najm_retrieval.text_preparation.models import (
    AssemblyIssue,
    CONTENT_BLOCK_TYPES,
    LogicalUnit,
)

__all__ = [
    "resolve_logical_unit_contexts",
    "ContextIssue",
    "ContextMarker",
    "ContextualLogicalUnit",
    "LogicalUnitContext",
    "STRUCTURAL_CONTEXT_TYPES",
    "AssemblyIssue",
    "CONTENT_BLOCK_TYPES",
    "LogicalUnit",
    "assemble_logical_units",
    "attributes_to_dict",
]
