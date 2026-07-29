"""Tests for the manifest-driven corpus scanner."""

from pathlib import Path

from najm_retrieval.corpus.manifest import (
    CorpusManifest,
    DatasetConfig,
    ReferenceVersionConfig,
    WorkConfig,
)
from najm_retrieval.corpus.scanner import scan_corpus


AUTHOR_ID = "0001Author"
WORK_ID = "0001Author.Work"
CANONICAL_VERSION = "0001Author.Work.Canonical-per1"
REFERENCE_VERSION = "0001Author.Work.Reference-per1"


def _write_file(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_manifest() -> CorpusManifest:
    return CorpusManifest(
        dataset=DatasetConfig(
            name="test-corpus",
            repository="https://example.com/test-corpus",
            commit="abc123",
        ),
        works={
            WORK_ID: WorkConfig(
                work_id=WORK_ID,
                title_fa="اثر آزمایشی",
                profile="structured_poetry",
                canonical_version=CANONICAL_VERSION,
                include_in_index=True,
                reference_versions=(
                    ReferenceVersionConfig(
                        version_id=REFERENCE_VERSION,
                        profile="raw_ocr_reference",
                        include_in_index=False,
                    ),
                ),
            )
        },
    )


def _create_version(
    corpus_root: Path,
    version_id: str,
    *,
    valid_magic_line: bool = True,
    create_version_yml: bool = True,
) -> None:
    author_dir = corpus_root / AUTHOR_ID
    work_dir = author_dir / WORK_ID

    _write_file(
        author_dir / f"{AUTHOR_ID}.yml",
        "00#AUTH#URI######: 0001Author",
    )
    _write_file(
        work_dir / f"{WORK_ID}.yml",
        "00#BOOK#URI######: 0001Author.Work",
    )

    first_line = (
        "######OpenITI#"
        if valid_magic_line
        else "not-an-openiti-file"
    )

    _write_file(
        work_dir / version_id,
        f"{first_line}\n#META#Header#End#\nمتن آزمایشی",
    )

    if create_version_yml:
        _write_file(
            work_dir / f"{version_id}.yml",
            f"00#VERS#URI######: {version_id}",
        )


def test_scan_valid_manifest_driven_corpus(tmp_path: Path) -> None:
    manifest = _build_manifest()

    _create_version(tmp_path, CANONICAL_VERSION)
    _create_version(tmp_path, REFERENCE_VERSION)

    result = scan_corpus(tmp_path, manifest)

    assert result.ok is True
    assert len(result.versions) == 2
    assert len(result.canonical_versions) == 1
    assert len(result.indexable_versions) == 1
    assert result.issues == ()
    assert result.unexpected_versions == ()

    canonical = result.canonical_versions[0]

    assert canonical.version_id == CANONICAL_VERSION
    assert canonical.is_canonical is True
    assert canonical.include_in_index is True

    reference = next(
        version
        for version in result.versions
        if version.version_id == REFERENCE_VERSION
    )

    assert reference.is_canonical is False
    assert reference.include_in_index is False


def test_missing_version_yaml_is_reported(tmp_path: Path) -> None:
    manifest = _build_manifest()

    _create_version(
        tmp_path,
        CANONICAL_VERSION,
        create_version_yml=False,
    )
    _create_version(tmp_path, REFERENCE_VERSION)

    result = scan_corpus(tmp_path, manifest)

    issue_codes = {issue.code for issue in result.issues}

    assert "missing_version_yml" in issue_codes
    assert result.ok is False


def test_invalid_magic_line_is_reported(tmp_path: Path) -> None:
    manifest = _build_manifest()

    _create_version(
        tmp_path,
        CANONICAL_VERSION,
        valid_magic_line=False,
    )
    _create_version(tmp_path, REFERENCE_VERSION)

    result = scan_corpus(tmp_path, manifest)

    issue_codes = {issue.code for issue in result.issues}

    assert "invalid_magic_line" in issue_codes
    assert result.ok is False


def test_unexpected_version_is_reported(tmp_path: Path) -> None:
    manifest = _build_manifest()

    _create_version(tmp_path, CANONICAL_VERSION)
    _create_version(tmp_path, REFERENCE_VERSION)

    unexpected_version = "0001Author.Work.Unexpected-per1"
    _create_version(tmp_path, unexpected_version)

    result = scan_corpus(tmp_path, manifest)

    unexpected_names = {
        path.name
        for path in result.unexpected_versions
    }

    assert unexpected_version in unexpected_names
    assert result.ok is False