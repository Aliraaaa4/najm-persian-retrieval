"""Tests for reproducible parser sample extraction."""

import json
from pathlib import Path

import pytest

from najm_retrieval.parsing.samples import (
    extract_parser_sample,
    write_parser_sample,
)


def _write_source(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        handle.write(text)


def test_extract_sample_preserves_text_and_offsets(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "test-per1"

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Test\r\n"
        "#META#Header#End#\r\n"
        "PageV01P001\r\n"
        "متن اول\r\n"
        "متن دوم"
    )

    _write_source(
        source_path,
        source_text,
    )

    sample = extract_parser_sample(
        sample_id="test_sample",
        split="development",
        version_id="0001Author.Work.Version-per1",
        profile="mixed_prose_ocr",
        source_path=source_path,
        source_label="0001Author/Work/test-per1",
        line_start=4,
        line_end=5,
    )

    expected_prefix = (
        "######OpenITI#\r\n"
        "#META# title: Test\r\n"
        "#META#Header#End#\r\n"
    )

    assert sample.body_line_start == 4
    assert sample.line_start == 4
    assert sample.line_end == 5

    assert sample.char_start == len(expected_prefix)

    assert sample.raw_text == (
        "PageV01P001\r\n"
        "متن اول\r\n"
    )

    assert sample.char_end == (
        sample.char_start
        + len(sample.raw_text)
    )

    assert tuple(
        line.line_number
        for line in sample.lines
    ) == (4, 5)


def test_extract_sample_rejects_header_lines(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "test-per1"

    _write_source(
        source_path,
        (
            "######OpenITI#\n"
            "#META#Header#End#\n"
            "Body\n"
        ),
    )

    with pytest.raises(
        ValueError,
        match="body starts",
    ):
        extract_parser_sample(
            sample_id="invalid",
            split="development",
            version_id="version-per1",
            profile="structured_poetry",
            source_path=source_path,
            source_label="test-per1",
            line_start=1,
            line_end=2,
        )


def test_extract_sample_rejects_unknown_split(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "test-per1"

    _write_source(
        source_path,
        (
            "######OpenITI#\n"
            "#META#Header#End#\n"
            "Body\n"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported split",
    ):
        extract_parser_sample(
            sample_id="invalid",
            split="training",
            version_id="version-per1",
            profile="structured_poetry",
            source_path=source_path,
            source_label="test-per1",
            line_start=3,
            line_end=3,
        )


def test_write_sample_creates_json_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "test-per1"

    _write_source(
        source_path,
        (
            "######OpenITI#\n"
            "#META#Header#End#\n"
            "Body\n"
        ),
    )

    sample = extract_parser_sample(
        sample_id="sample_01",
        split="holdout",
        version_id="version-per1",
        profile="structured_poetry",
        source_path=source_path,
        source_label="test-per1",
        line_start=3,
        line_end=3,
    )

    output_path = tmp_path / "sample.json"

    write_parser_sample(
        sample,
        output_path,
    )

    data = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert data["sample_id"] == "sample_01"
    assert data["split"] == "holdout"
    assert data["raw_text"] == "Body\n"
    assert data["annotations"]["blocks"] == []

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        write_parser_sample(
            sample,
            output_path,
        )