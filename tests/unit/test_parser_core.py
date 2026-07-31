"""Tests for exact OpenITI source loading and body splitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from najm_retrieval.parsing.core import (
    build_source_line_records,
    load_openiti_source,
    split_openiti_source,
)


def test_load_source_preserves_body_and_offsets(
    tmp_path: Path,
) -> None:
    """The loader must preserve CRLF and exact character offsets."""

    source_path = tmp_path / (
        "0001Author.Work.Version-per1"
    )

    header = (
        "######OpenITI#\r\n"
        "#META# title: Test\r\n"
        "#META#Header#End#\r\n"
    )

    body = (
        "PageV01P001\r\n"
        "متن پایانی"
    )

    source_text = header + body

    source_path.write_bytes(
        b"\xef\xbb\xbf"
        + source_text.encode("utf-8")
    )

    loaded = load_openiti_source(
        source_path
    )

    assert loaded.source_path == source_path
    assert loaded.version_id == source_path.name

    assert loaded.source_text == source_text
    assert loaded.header_text == header
    assert loaded.body_text == body

    assert loaded.body_char_start == len(
        header
    )

    assert loaded.body_line_start == 4

    assert tuple(
        line.line_number
        for line in loaded.lines
    ) == (4, 5)

    first, second = loaded.lines

    assert first.text == "PageV01P001\r\n"
    assert first.char_start == len(header)
    assert first.char_end == (
        len(header)
        + len(first.text)
    )

    assert second.text == "متن پایانی"
    assert second.char_start == (
        first.char_end
    )

    assert second.char_end == len(
        source_text
    )


def test_split_rejects_invalid_magic_line() -> None:
    """An OpenITI source must start with its magic line."""

    source_text = (
        "NotOpenITI\r\n"
        "#META#Header#End#\r\n"
        "Body\r\n"
    )

    with pytest.raises(
        ValueError,
        match="magic line",
    ):
        split_openiti_source(
            source_text,
            source_path=Path("invalid-per1"),
        )


def test_split_rejects_missing_header_end() -> None:
    """The OpenITI header terminator is mandatory."""

    source_text = (
        "######OpenITI#\n"
        "#META# title: Test\n"
        "Body without header terminator\n"
    )

    with pytest.raises(
        ValueError,
        match="Header end marker",
    ):
        split_openiti_source(
            source_text,
            source_path=Path("invalid-per1"),
        )


def test_build_line_records_preserves_final_line() -> None:
    """A final line without newline must remain unchanged."""

    body_text = (
        "خط اول\r\n"
        "خط دوم\n"
        "خط سوم"
    )

    records = build_source_line_records(
        body_text,
        body_char_start=100,
        body_line_start=8,
    )

    assert tuple(
        record.line_number
        for record in records
    ) == (8, 9, 10)

    assert tuple(
        record.text
        for record in records
    ) == (
        "خط اول\r\n",
        "خط دوم\n",
        "خط سوم",
    )

    assert records[0].char_start == 100

    assert records[-1].char_end == (
        100 + len(body_text)
    )

    reconstructed = "".join(
        record.text
        for record in records
    )

    assert reconstructed == body_text