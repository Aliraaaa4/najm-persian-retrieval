"""Tests for parsing every version returned by the corpus scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from najm_retrieval.corpus.scanner import (
    CorpusScanResult,
    CorpusVersionFiles,
    ScanIssue,
)
from najm_retrieval.parsing.corpus_runner import (
    CorpusRunnerError,
    parse_scanned_corpus,
)
from najm_retrieval.parsing.profiles import (
    MIXED_PROSE_OCR,
    RAW_OCR_REFERENCE,
    STRUCTURED_POETRY,
)


def write_version(
    root: Path,
    *,
    version_id: str,
    profile: str,
    body: str,
    include_in_index: bool,
    is_canonical: bool,
) -> CorpusVersionFiles:
    """Create one minimal on-disk OpenITI version."""

    parts = version_id.split(".")

    author_id = parts[0]
    work_id = ".".join(parts[:2])

    author_dir = root / author_id
    work_dir = author_dir / work_id

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    text_path = work_dir / version_id

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Runner Test\r\n"
        "#META#Header#End#\r\n"
        + body
    )

    text_path.write_bytes(
        source_text.encode("utf-8")
    )

    author_yml_path = (
        author_dir / f"{author_id}.yml"
    )

    work_yml_path = (
        work_dir / f"{work_id}.yml"
    )

    version_yml_path = (
        work_dir / f"{version_id}.yml"
    )

    for yaml_path in (
        author_yml_path,
        work_yml_path,
        version_yml_path,
    ):
        yaml_path.write_text(
            "{}\n",
            encoding="utf-8",
        )

    return CorpusVersionFiles(
        author_id=author_id,
        work_id=work_id,
        version_id=version_id,
        text_path=text_path,
        author_yml_path=author_yml_path,
        work_yml_path=work_yml_path,
        version_yml_path=version_yml_path,
        profile=profile,
        include_in_index=include_in_index,
        is_canonical=is_canonical,
    )


def make_valid_scan(
    tmp_path: Path,
) -> CorpusScanResult:
    """Create a scan containing all three parser profiles."""

    poetry = write_version(
        tmp_path,
        version_id=(
            "0001Author.Poetry.Test-per1"
        ),
        profile=STRUCTURED_POETRY,
        body=(
            "~~PageV01P001\r\n"
            "# بیت نخست %~% بیت دوم\r\n"
        ),
        include_in_index=True,
        is_canonical=True,
    )

    prose = write_version(
        tmp_path,
        version_id=(
            "0002Author.Prose.Test-per1"
        ),
        profile=MIXED_PROSE_OCR,
        body=(
            "PageV01P001\r\n"
            "# این یک بند نثر است.\r\n"
        ),
        include_in_index=True,
        is_canonical=True,
    )

    raw_ocr = write_version(
        tmp_path,
        version_id=(
            "0003Author.OCR.Test-per1"
        ),
        profile=RAW_OCR_REFERENCE,
        body=(
            "PageV00P001\r\n"
            "# heading-like OCR stays raw\r\n"
        ),
        include_in_index=False,
        is_canonical=False,
    )

    return CorpusScanResult(
        corpus_root=tmp_path,
        versions=(
            poetry,
            prose,
            raw_ocr,
        ),
        issues=(),
        unexpected_versions=(),
    )


def test_runner_parses_all_profiles(
    tmp_path: Path,
) -> None:
    """Every scanned version must use its configured parser."""

    scan = make_valid_scan(
        tmp_path
    )

    result = parse_scanned_corpus(
        scan
    )

    assert result.version_count == 3

    assert [
        version.profile
        for version in result.versions
    ] == [
        STRUCTURED_POETRY,
        MIXED_PROSE_OCR,
        RAW_OCR_REFERENCE,
    ]

    assert result.all_lossless is True

    assert all(
        version.document.reconstruct_body()
        == version.source.body_text
        for version in result.versions
    )


def test_runner_preserves_index_flags(
    tmp_path: Path,
) -> None:
    """Reference OCR versions must remain outside the main index."""

    result = parse_scanned_corpus(
        make_valid_scan(tmp_path)
    )

    assert [
        version.version_id
        for version in result.indexable_versions
    ] == [
        "0001Author.Poetry.Test-per1",
        "0002Author.Prose.Test-per1",
    ]

    assert [
        version.version_id
        for version in result.reference_versions
    ] == [
        "0003Author.OCR.Test-per1",
    ]


def test_runner_rejects_scan_issues(
    tmp_path: Path,
) -> None:
    """Parsing must not start after a failed corpus scan."""

    scan = CorpusScanResult(
        corpus_root=tmp_path,
        versions=(),
        issues=(
            ScanIssue(
                code="missing_text",
                message="A required text is missing.",
                path=tmp_path / "missing.txt",
            ),
        ),
        unexpected_versions=(),
    )

    with pytest.raises(
        CorpusRunnerError,
        match="missing_text",
    ):
        parse_scanned_corpus(
            scan
        )


def test_runner_rejects_unexpected_versions(
    tmp_path: Path,
) -> None:
    """Unconfigured source versions must stop the run."""

    unexpected = (
        tmp_path / "Unexpected.Work.Text-per1"
    )

    unexpected.write_text(
        "unexpected",
        encoding="utf-8",
    )

    scan = CorpusScanResult(
        corpus_root=tmp_path,
        versions=(),
        issues=(),
        unexpected_versions=(
            unexpected,
        ),
    )

    with pytest.raises(
        CorpusRunnerError,
        match="Unexpected version",
    ):
        parse_scanned_corpus(
            scan
        )


def test_runner_rejects_duplicate_version_ids(
    tmp_path: Path,
) -> None:
    """A version ID may only be parsed once."""

    valid_scan = make_valid_scan(
        tmp_path
    )

    duplicate_scan = CorpusScanResult(
        corpus_root=tmp_path,
        versions=(
            valid_scan.versions[0],
            valid_scan.versions[0],
        ),
        issues=(),
        unexpected_versions=(),
    )

    with pytest.raises(
        CorpusRunnerError,
        match="Duplicate version ID",
    ):
        parse_scanned_corpus(
            duplicate_scan
        )
