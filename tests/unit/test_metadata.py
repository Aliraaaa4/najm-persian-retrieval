"""Tests for OpenITI metadata loading and header extraction."""

from pathlib import Path

import pytest

from najm_retrieval.corpus.metadata import (
    MetadataError,
    MetadataStatus,
    classify_metadata_value,
    extract_text_header,
    load_openiti_yml,
)


def test_classify_metadata_values() -> None:
    assert classify_metadata_value("Persian") == (
        "Persian",
        MetadataStatus.VALID,
    )

    assert classify_metadata_value("   ") == (
        None,
        MetadataStatus.MISSING,
    )

    assert classify_metadata_value("UNKNOWN") == (
        "UNKNOWN",
        MetadataStatus.PLACEHOLDER,
    )

    assert classify_metadata_value(["a", "b"]) == (
        None,
        MetadataStatus.MALFORMED,
    )
    
    assert classify_metadata_value("Fulān") == (
        "Fulān",
        MetadataStatus.PLACEHOLDER,
    )

    assert classify_metadata_value(
        "URIs from Althurayya, comma separated"
    ) == (
        "URIs from Althurayya, comma separated",
        MetadataStatus.PLACEHOLDER,
    )

    assert classify_metadata_value(
        "AUTH_URI from OpenITI, comma separated"
    ) == (
        "AUTH_URI from OpenITI, comma separated",
        MetadataStatus.PLACEHOLDER,
    )

    assert classify_metadata_value(
        "YEAR-MON-DA (X+ for unknown)"
    ) == (
        "YEAR-MON-DA (X+ for unknown)",
        MetadataStatus.PLACEHOLDER,
    )

    assert classify_metadata_value(
        "viaf@id, wikidata@id, src@id"
    ) == (
        "viaf@id, wikidata@id, src@id",
        MetadataStatus.PLACEHOLDER,
        
    )
    assert classify_metadata_value("N/A") == (
        "N/A",
        MetadataStatus.PLACEHOLDER,
    )

    assert classify_metadata_value("NA") == (
        "NA",
        MetadataStatus.PLACEHOLDER,
    )

    assert classify_metadata_value(
        "Abū Fulān, Abū Fulānaŧ"
    ) == (
        "Abū Fulān, Abū Fulānaŧ",
        MetadataStatus.PLACEHOLDER,
    )


def test_load_valid_openiti_yaml(tmp_path: Path) -> None:
    metadata_path = tmp_path / "version.yml"

    metadata_path.write_text(
        """
00#VERS#URI######: 0001Author.Work.Version-per1
90#VERS#LANG#####Y: Persian
90#VERS#COMMENT##O: A valid comment
""".strip()
        + "\n",
        encoding="utf-8",
    )

    record = load_openiti_yml(metadata_path)

    assert len(record.fields) == 3
    assert len(record.valid_fields) == 3
    assert record.issues == ()

    uri_field = record.get_first("00#VERS#URI######")

    assert uri_field is not None
    assert uri_field.value == "0001Author.Work.Version-per1"
    assert uri_field.status is MetadataStatus.VALID


def test_placeholder_and_non_scalar_values_are_reported(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.yml"

    metadata_path.write_text(
        """
title: UNKNOWN
empty_value:
nested_value:
  key: value
""".strip()
        + "\n",
        encoding="utf-8",
    )

    record = load_openiti_yml(metadata_path)

    statuses = {
        field.name: field.status
        for field in record.fields
    }

    assert statuses["title"] is MetadataStatus.PLACEHOLDER
    assert statuses["empty_value"] is MetadataStatus.MISSING
    assert statuses["nested_value"] is MetadataStatus.MALFORMED

    issue_codes = {
        issue.code
        for issue in record.issues
    }

    assert "placeholder_value" in issue_codes
    assert "non_scalar_value" in issue_codes


def test_extract_valid_text_header(tmp_path: Path) -> None:
    text_path = tmp_path / "text-per1"

    text_path.write_text(
        """
######OpenITI#
#META# URI: 0001Author.Work.Version-per1
#META# Language: Persian
#META#Header#End#
متن اصلی کتاب
""".lstrip(),
        encoding="utf-8",
    )

    header = extract_text_header(text_path)

    assert header.magic_line_valid is True
    assert header.header_end_found is True
    assert len(header.fields) == 2
    assert header.issues == ()

    language = header.get_first("Language")

    assert language is not None
    assert language.value == "Persian"
    assert language.source == "text_header"

    assert "متن اصلی کتاب" not in header.raw_lines


def test_invalid_header_is_reported(tmp_path: Path) -> None:
    text_path = tmp_path / "invalid-per1"

    text_path.write_text(
        """
not-openiti
#META# Line without separator
متن
""".lstrip(),
        encoding="utf-8",
    )

    header = extract_text_header(text_path, max_lines=20)

    issue_codes = {
        issue.code
        for issue in header.issues
    }

    assert header.magic_line_valid is False
    assert header.header_end_found is False
    assert "invalid_magic_line" in issue_codes
    assert "unparsed_header_line" in issue_codes
    assert "missing_header_end" in issue_codes


def test_missing_yaml_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yml"

    with pytest.raises(MetadataError, match="not found"):
        load_openiti_yml(missing_path)