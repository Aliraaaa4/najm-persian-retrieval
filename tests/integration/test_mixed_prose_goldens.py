"""Regression tests for mixed-prose parsing against reviewed goldens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from najm_retrieval.parsing.core import (
    OpenITISource,
    SourceLineRecord,
)
from najm_retrieval.parsing.handlers.prose_ocr import (
    build_mixed_prose_records,
    parse_mixed_prose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_ROOT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "parser_goldens"
)

MIXED_PROSE_GOLDENS = (
    "pilot_masalik_aocp_01.json",
    "pilot_majalis_01.json",
    "pilot_akhlaq_01.json",
)


def load_golden(
    filename: str,
) -> dict[str, Any]:
    """Load one completed mixed-prose golden."""

    path = GOLDEN_ROOT / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Golden file not found: {path}"
        )

    sample = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if sample.get("profile") != "mixed_prose_ocr":
        raise ValueError(
            f"Unexpected profile in {path}: "
            f"{sample.get('profile')!r}"
        )

    annotations = sample.get(
        "annotations"
    )

    if not isinstance(annotations, dict):
        raise ValueError(
            f"Golden annotations missing: {path}"
        )

    if annotations.get("status") != "complete":
        raise ValueError(
            f"Golden is not complete: {path}"
        )

    blocks = annotations.get("blocks")

    if not isinstance(blocks, list):
        raise ValueError(
            f"Golden blocks are missing: {path}"
        )

    return sample


def build_sample_source(
    sample: dict[str, Any],
) -> OpenITISource:
    """Create an OpenITISource representing one golden sample."""

    raw_text = sample.get("raw_text")

    if not isinstance(raw_text, str):
        raw_text = sample.get(
            "source_text"
        )

    if not isinstance(raw_text, str):
        raise ValueError(
            "Golden sample has no raw_text "
            "or source_text."
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
            "the exact sample text."
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
    """Select fields defining one structural record."""

    return {
        "block_id": record.get(
            "block_id"
        ),
        "block_type": record.get(
            "block_type"
        ),
        "line_start": record.get(
            "line_start"
        ),
        "line_end": record.get(
            "line_end"
        ),
        "char_start": record.get(
            "char_start"
        ),
        "char_end": record.get(
            "char_end"
        ),
        "raw_text": record.get(
            "raw_text"
        ),
        "group_id": record.get(
            "group_id"
        ),
        "attributes": record.get(
            "attributes",
            {},
        ),
    }


@pytest.mark.parametrize(
    "filename",
    MIXED_PROSE_GOLDENS,
)
def test_production_records_match_complete_golden(
    filename: str,
) -> None:
    """Production records must equal reviewed annotations."""

    sample = load_golden(filename)
    source = build_sample_source(sample)

    actual_records = build_mixed_prose_records(
        source
    )

    expected_records = sample[
        "annotations"
    ]["blocks"]

    actual_signatures = [
        record_signature(record)
        for record in actual_records
    ]

    expected_signatures = [
        record_signature(record)
        for record in expected_records
    ]

    assert actual_signatures == (
        expected_signatures
    )


@pytest.mark.parametrize(
    "filename",
    MIXED_PROSE_GOLDENS,
)
def test_production_parser_passes_lossless_gate(
    filename: str,
) -> None:
    """Every reviewed prose sample must remain lossless."""

    sample = load_golden(filename)
    source = build_sample_source(sample)

    document, metrics = parse_mixed_prose(
        source
    )

    assert document.profile == (
        "mixed_prose_ocr"
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
