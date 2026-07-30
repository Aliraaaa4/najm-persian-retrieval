"""Tests for corpus manifest loading and validation."""

from pathlib import Path

import pytest

from najm_retrieval.corpus.manifest import ManifestError, load_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "corpus_manifest.yaml"


def test_load_project_manifest() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    assert manifest.dataset.name == "PER0675AH"
    assert len(manifest.works) == 6
    assert len(manifest.canonical_versions) == 6
    assert len(manifest.reference_versions) == 1
    assert len(manifest.all_versions) == 7
    assert len(manifest.indexable_versions) == 6


def test_masalik_configuration() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    masalik = manifest.get_work("0670IbnAscadHanati.Masalik")

    assert masalik.profile == "mixed_prose_ocr"
    assert (
        masalik.canonical_version
        == "0670IbnAscadHanati.Masalik.AOCP202605141788-per1"
    )

    assert len(masalik.reference_versions) == 1

    kraken = masalik.reference_versions[0]

    assert (
        kraken.version_id
        == "0670IbnAscadHanati.Masalik.Kraken220107010708-per1"
    )
    assert kraken.profile == "raw_ocr_reference"
    assert kraken.include_in_index is False


def test_unknown_work_raises_clear_error() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    with pytest.raises(ManifestError, match="Unknown work ID"):
        manifest.get_work("missing.work")


def test_invalid_profile_is_rejected(tmp_path: Path) -> None:
    invalid_manifest = tmp_path / "invalid_manifest.yaml"
    invalid_manifest.write_text(
        """
dataset:
  name: "test"
  repository: "https://example.com/test"
  commit: "abc123"

works:
  "0001Author.Work":
    title_fa: "اثر آزمایشی"
    profile: "invalid_profile"
    canonical_version: "0001Author.Work.Version-per1"
    include_in_index: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="Unsupported profile"):
        load_manifest(invalid_manifest)


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ManifestError, match="Manifest file not found"):
        load_manifest(missing_path)