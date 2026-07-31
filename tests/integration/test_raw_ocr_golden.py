"""Regression tests for raw OCR parsing against Kraken golden."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from najm_retrieval.parsing.core import (
    OpenITISource,
    SourceLineRecord,
)
from najm_retrieval.parsing.handlers.prose_ocr import (
    build_raw_ocr_records,
    parse_raw_ocr_reference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "parser_goldens"
    / "pilot_masalik_kraken_01.json"
)


def load_golden() -> dict[str, Any]:
    """Load the completed Kraken golden."""

    sample = json.loads(
        GOLDEN_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if sample.get("profile") != (
        "raw_ocr_reference"
    ):
        raise ValueError(
            "Kraken golden has an unexpected profile."
        )

    annotations = sample.get(
        "annotations"
    )

    if not isinstance(annotations, dict):
        raise ValueError(
            "Kraken golden annotations are missing."
        )

    if annotations.get("status") != "complete":
        raise ValueError(
            "Kraken golden is not complete."
        )

    return sample


def build_sample_source(
    sample: dict[str, Any],
) -> OpenITISource:
    """Create an exact source object from a golden sample."""

    raw_text = sample.get("raw_text")

    if not isinstance(raw_text, str):
        raw_text = sample.get(
            "source_text"
        )

    if not isinstance(raw_text, str):
        raise ValueError(
            "Kraken golden has no source text."
        )

    char_start = sample.get(
        "char_start"
    )

    line_start = sample.get(
        "line_start"
    )

    version_id = sample.get(
        "version_id"
    )

    if not isinstance(char_start, int):
        raise ValueError(
            "Golden char_start must be an integer."
        )

    if not isinstance(line_start, int):
        raise ValueError(
            "Golden line_start must be an integer."
        )

    if not isinstance(version_id, str):
        raise ValueError(
            "Golden version_id must be a string."
        )

    line_data = sample.get("lines")

    if not isinstance(line_data, list):
        raise ValueError(
            "Golden lines must be an array."
        )

    lines = tuple(
        SourceLineRecord(
            line_number=line["line_number"],
            char_start=line["char_start"],
            char_end=line["char_end"],
            text=line["text"],
        )
        for line in line_data
    )

    reconstructed = "".join(
        line.text
        for line in lines
    )

    if reconstructed != raw_text:
        raise ValueError(
            "Golden lines do not reconstruct "
            "the sample text."
        )

    prefix = " " * char_start

    return OpenITISource(
        source_path=Path(version_id),
        source_text=prefix + raw_text,
        header_text=prefix,
        body_text=raw_text,
        body_char_start=char_start,
        body_line_start=line_start,
        lines=lines,
    )


def record_signature(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Select structural fields for exact comparison."""

    return {
        "block_id": record.get("block_id"),
        "block_type": record.get("block_type"),
        "line_start": record.get("line_start"),
        "line_end": record.get("line_end"),
        "char_start": record.get("char_start"),
        "char_end": record.get("char_end"),
        "raw_text": record.get("raw_text"),
        "group_id": record.get("group_id"),
        "attributes": record.get(
            "attributes",
            {},
        ),
    }


def test_raw_ocr_records_match_kraken_golden() -> None:
    """Production records must equal reviewed annotations."""

    sample = load_golden()
    source = build_sample_source(sample)

    actual = build_raw_ocr_records(
        source
    )

    expected = sample[
        "annotations"
    ]["blocks"]

    assert [
        record_signature(record)
        for record in actual
    ] == [
        record_signature(record)
        for record in expected
    ]


def test_raw_ocr_parser_passes_lossless_gate() -> None:
    """The Kraken sample must pass strict lossless checks."""

    sample = load_golden()
    source = build_sample_source(sample)

    document, metrics = (
        parse_raw_ocr_reference(
            source
        )
    )

    assert (
        document.reconstruct_body()
        == source.body_text
    )

    assert metrics.uncovered_chars == 0
    assert metrics.overlapping_chars == 0

    assert (
        metrics.reconstruction_matches_source
        is True
    )

    assert (
        metrics.passes_lossless_gate
        is True
    )
