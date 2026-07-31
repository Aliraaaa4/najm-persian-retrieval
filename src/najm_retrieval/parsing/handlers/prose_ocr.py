"""Production handlers for OpenITI prose and OCR text profiles."""

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
from najm_retrieval.parsing.mixed_prose_drafting import (
    draft_mixed_prose_blocks,
)
from najm_retrieval.parsing.models import (
    ParseMetrics,
    ParsedDocument,
)
from najm_retrieval.parsing.profiles import (
    MIXED_PROSE_OCR,
)


MIXED_PROSE_PARSER_NAME = (
    "mixed_prose_ocr_state_machine"
)

MIXED_PROSE_PARSER_VERSION = "0.1.0"


def build_mixed_prose_records(
    source: OpenITISource,
) -> list[dict[str, Any]]:
    """Create lossless records for a complete mixed-prose body."""

    sample = {
        "profile": MIXED_PROSE_OCR,
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

    records = draft_mixed_prose_blocks(
        sample
    )

    reconstructed = "".join(
        record["raw_text"]
        for record in records
    )

    if reconstructed != source.body_text:
        raise ValueError(
            "Mixed-prose records do not reconstruct "
            "the exact source body."
        )

    return records


def parse_mixed_prose(
    source: OpenITISource,
) -> tuple[
    ParsedDocument,
    ParseMetrics,
]:
    """Parse one complete mixed-prose OCR source."""

    started_at = perf_counter()

    records = build_mixed_prose_records(
        source
    )

    document = build_parsed_document(
        source=source,
        profile=MIXED_PROSE_OCR,
        block_records=records,
        parser_name=(
            MIXED_PROSE_PARSER_NAME
        ),
        parser_version=(
            MIXED_PROSE_PARSER_VERSION
        ),
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
            "Mixed-prose parser failed "
            "the strict lossless gate."
        )

    return document, metrics
