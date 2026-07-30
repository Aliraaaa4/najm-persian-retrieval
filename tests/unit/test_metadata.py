"""Tests for OpenITI metadata loading and header extraction."""
from najm_retrieval.corpus.manifest import WorkConfig
from najm_retrieval.corpus.scanner import CorpusVersionFiles
from pathlib import Path

import pytest

from najm_retrieval.corpus.metadata import (
    MetadataError,
    MetadataStatus,
    load_document_metadata,
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
        
        
        
def _write_document_file(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        content,
        encoding="utf-8",
    )


def test_load_document_metadata_combines_sources(
    tmp_path: Path,
) -> None:
    author_id = "0001Author"
    work_id = "0001Author.Work"
    version_id = "0001Author.Work.Version-per1"

    author_dir = tmp_path / author_id
    work_dir = author_dir / work_id

    author_yml = author_dir / f"{author_id}.yml"
    work_yml = work_dir / f"{work_id}.yml"
    version_yml = work_dir / f"{version_id}.yml"
    text_path = work_dir / version_id

    _write_document_file(
        author_yml,
        (
            "00#AUTH#URI######: 0001Author\n"
            "10#AUTH#SHUHRA#AR: Author Arabic\n"
            "10#AUTH#SHUHRA#EN: Author English\n"
        ),
    )

    _write_document_file(
        work_yml,
        (
            "00#BOOK#URI######: 0001Author.Work\n"
            "10#BOOK#TITLEA#AR: Kitab Test\n"
        ),
    )

    _write_document_file(
        version_yml,
        (
            "00#VERS#URI######: "
            "0001Author.Work.Version-per1\n"
            "80#VERS#BASED####: "
            "https://example.com/source\n"
            "90#VERS#ISSUES###: "
            "UNCORRECTED_OCR, PRIMARY_VERSION\n"
        ),
    )

    _write_document_file(
        text_path,
        (
            "######OpenITI#\n"
            "#META# Origin: eScriptorium\n"
            "#META# transcription layer name: kraken:test\n"
            "#META# avg transcription confidence: 0.98\n"
            "#META#Header#End#\n"
            "Test text\n"
        ),
    )

    version = CorpusVersionFiles(
        author_id=author_id,
        work_id=work_id,
        version_id=version_id,
        text_path=text_path,
        author_yml_path=author_yml,
        work_yml_path=work_yml,
        version_yml_path=version_yml,
        profile="mixed_prose_ocr",
        include_in_index=True,
        is_canonical=True,
    )

    work = WorkConfig(
        work_id=work_id,
        title_fa="اثر آزمایشی",
        profile="mixed_prose_ocr",
        canonical_version=version_id,
        include_in_index=True,
    )

    metadata = load_document_metadata(
        version,
        work,
    )

    assert metadata.author_id == author_id
    assert metadata.author_name == "Author Arabic"
    assert metadata.author_name_en == "Author English"

    assert metadata.work_id == work_id
    assert metadata.title_fa == "اثر آزمایشی"
    assert metadata.title_transliterated == "Kitab Test"

    assert metadata.version_id == version_id
    assert metadata.language_code == "per"
    assert metadata.language == "Persian"

    assert metadata.source_url == (
        "https://example.com/source"
    )

    assert metadata.text_quality == (
        "UNCORRECTED_OCR",
        "PRIMARY_VERSION",
    )

    assert metadata.origin == "eScriptorium"
    assert metadata.transcription_layer == "kraken:test"
    assert metadata.ocr_confidence == 0.98

    assert metadata.is_canonical is True
    assert metadata.include_in_index is True