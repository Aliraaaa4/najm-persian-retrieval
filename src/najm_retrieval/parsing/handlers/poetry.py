"""Production parser handler for structured OpenITI poetry."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from najm_retrieval.parsing.assembly import (
    build_parsed_document,
    compute_parse_metrics,
)
from najm_retrieval.parsing.core import (
    OpenITISource,
)
from najm_retrieval.parsing.golden_drafting import (
    draft_structured_poetry_blocks,
)
from najm_retrieval.parsing.models import (
    ParseMetrics,
    ParsedDocument,
)
from najm_retrieval.parsing.profiles import (
    STRUCTURED_POETRY,
)


PARSER_NAME = "structured_poetry_state_machine"
PARSER_VERSION = "0.1.0"


def build_structured_poetry_records(
    source: OpenITISource,
) -> list[dict[str, Any]]:
    """Create lossless records for a complete poetry body."""

    sample = {
        "profile": STRUCTURED_POETRY,
        "lines": [
            {
                "line_number": line.line_number,
                "char_start": line.char_start,
                "char_end": line.char_end,
                "text": line.text,
            }
            for line in source.lines
        ],
    }

    records = draft_structured_poetry_blocks(
        sample
    )

    reconstructed = "".join(
        record["raw_text"]
        for record in records
    )

    if reconstructed != source.body_text:
        raise ValueError(
            "Structured-poetry records do not "
            "reconstruct the exact source body."
        )

    return records


def parse_structured_poetry(
    source: OpenITISource,
) -> tuple[
    ParsedDocument,
    ParseMetrics,
]:
    """Parse one complete structured-poetry source."""

    started_at = perf_counter()

    records = build_structured_poetry_records(
        source
    )

    document = build_parsed_document(
        source=source,
        profile=STRUCTURED_POETRY,
        block_records=records,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
    )

    runtime_seconds = (
        perf_counter() - started_at
    )

    metrics = compute_parse_metrics(
        source=source,
        document=document,
        runtime_seconds=runtime_seconds,
        peak_memory_bytes=0,
    )

    if not metrics.passes_lossless_gate:
        raise ValueError(
            "Structured-poetry parser failed "
            "the strict lossless gate."
        )

    return document, metrics
