"""Dispatch OpenITI sources to profile-specific parsers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from najm_retrieval.parsing.core import (
    OpenITISource,
)
from najm_retrieval.parsing.handlers.poetry import (
    parse_structured_poetry,
)
from najm_retrieval.parsing.handlers.prose_ocr import (
    parse_mixed_prose,
    parse_raw_ocr_reference,
)
from najm_retrieval.parsing.models import (
    ParseMetrics,
    ParsedDocument,
)
from najm_retrieval.parsing.profiles import (
    MIXED_PROSE_OCR,
    RAW_OCR_REFERENCE,
    STRUCTURED_POETRY,
)


ParserResult = tuple[
    ParsedDocument,
    ParseMetrics,
]

ParserHandler = Callable[
    [OpenITISource],
    ParserResult,
]


PARSER_HANDLERS: Final[
    dict[str, ParserHandler]
] = {
    STRUCTURED_POETRY: (
        parse_structured_poetry
    ),
    MIXED_PROSE_OCR: (
        parse_mixed_prose
    ),
    RAW_OCR_REFERENCE: (
        parse_raw_ocr_reference
    ),
}


def parse_source(
    *,
    source: OpenITISource,
    profile: str,
) -> ParserResult:
    """Parse one source using its configured profile."""

    handler = PARSER_HANDLERS.get(
        profile
    )

    if handler is None:
        supported = ", ".join(
            sorted(PARSER_HANDLERS)
        )

        raise ValueError(
            f"Unsupported parser profile "
            f"{profile!r}. Supported profiles: "
            f"{supported}"
        )

    document, metrics = handler(
        source
    )

    if document.profile != profile:
        raise ValueError(
            "Parser returned a document with "
            f"profile {document.profile!r}, "
            f"but {profile!r} was requested."
        )

    if document.version_id != source.version_id:
        raise ValueError(
            "Parser changed the source "
            "version identifier."
        )

    if (
        document.reconstruct_body()
        != source.body_text
    ):
        raise ValueError(
            "Dispatched parser did not "
            "reconstruct the exact source body."
        )

    if not metrics.passes_lossless_gate:
        raise ValueError(
            "Dispatched parser failed "
            "the strict lossless gate."
        )

    return document, metrics
