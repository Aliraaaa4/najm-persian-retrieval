"""Production handlers for OpenITI prose and OCR profiles."""

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
from najm_retrieval.parsing.raw_ocr_drafting import (
    draft_raw_ocr_blocks,
)
from najm_retrieval.parsing.models import (
    ParseMetrics,
    ParsedDocument,
)
from najm_retrieval.parsing.profiles import (
    MIXED_PROSE_OCR,
    RAW_OCR_REFERENCE,
)


MIXED_PROSE_PARSER_NAME = (
    "mixed_prose_ocr_state_machine"
)

MIXED_PROSE_PARSER_VERSION = "0.1.0"

RAW_OCR_PARSER_NAME = (
    "conservative_raw_ocr_parser"
)

RAW_OCR_PARSER_VERSION = "0.1.0"


def _build_sample(
    source: OpenITISource,
    *,
    profile: str,
) -> dict[str, Any]:
    """Convert an exact source body to drafting input."""

    return {
        "profile": profile,
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


def _validate_record_reconstruction(
    *,
    source: OpenITISource,
    records: list[dict[str, Any]],
    parser_label: str,
) -> None:
    """Require exact reconstruction before typed assembly."""

    reconstructed = "".join(
        record["raw_text"]
        for record in records
    )

    if reconstructed != source.body_text:
        raise ValueError(
            f"{parser_label} records do not "
            "reconstruct the exact source body."
        )


def build_mixed_prose_records(
    source: OpenITISource,
) -> list[dict[str, Any]]:
    """Create lossless records for a mixed-prose body."""

    sample = _build_sample(
        source,
        profile=MIXED_PROSE_OCR,
    )

    records = draft_mixed_prose_blocks(
        sample
    )

    _validate_record_reconstruction(
        source=source,
        records=records,
        parser_label="Mixed-prose",
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

    metrics = compute_parse_metrics(
        source=source,
        document=document,
        runtime_seconds=(
            perf_counter() - started_at
        ),
        peak_memory_bytes=0,
    )

    if not metrics.passes_lossless_gate:
        raise ValueError(
            "Mixed-prose parser failed "
            "the strict lossless gate."
        )

    return document, metrics


def build_raw_ocr_records(
    source: OpenITISource,
) -> list[dict[str, Any]]:
    """Create conservative lossless records for raw OCR."""

    sample = _build_sample(
        source,
        profile=RAW_OCR_REFERENCE,
    )

    records = draft_raw_ocr_blocks(
        sample
    )

    _validate_record_reconstruction(
        source=source,
        records=records,
        parser_label="Raw OCR",
    )

    return records


def parse_raw_ocr_reference(
    source: OpenITISource,
) -> tuple[
    ParsedDocument,
    ParseMetrics,
]:
    """Parse one complete raw OCR reference source."""

    started_at = perf_counter()

    records = build_raw_ocr_records(
        source
    )

    document = build_parsed_document(
        source=source,
        profile=RAW_OCR_REFERENCE,
        block_records=records,
        parser_name=RAW_OCR_PARSER_NAME,
        parser_version=(
            RAW_OCR_PARSER_VERSION
        ),
    )

    metrics = compute_parse_metrics(
        source=source,
        document=document,
        runtime_seconds=(
            perf_counter() - started_at
        ),
        peak_memory_bytes=0,
    )

    if not metrics.passes_lossless_gate:
        raise ValueError(
            "Raw OCR parser failed "
            "the strict lossless gate."
        )

    return document, metrics
