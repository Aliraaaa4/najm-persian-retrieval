"""Tests for building a validated SQLite passage store."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from najm_retrieval.retrieval import (
    PASSAGE_STORE_SCHEMA_VERSION,
    PassageStore,
    PassageStoreBuildError,
    PassageStoreBuildReport,
    build_passage_store,
)


AUTHOR_ID = "0001Author"
WORK_ID = "0001Author.Work"
VERSION_ID = "0001Author.Work.TEST-per1"


def _passage(
    ordinal: int,
    *,
    passage_id: str | None = None,
    version_id: str = VERSION_ID,
) -> dict[str, object]:
    current_id = (
        passage_id
        if passage_id is not None
        else (
            f"{version_id}:"
            f"passage_{ordinal:06d}"
        )
    )

    return {
        "schema_version": "1.0.0",
        "passage_id": current_id,
        "version_id": version_id,
        "profile": "structured_poetry",
        "kind": "diwan",
        "ordinal": ordinal,
        "include_in_index": True,
        "source": {
            "unit_ids": [
                f"{version_id}:verse_{ordinal:04d}",
            ],
            "spans": [],
        },
        "context": {
            "heading_path": [
                "غزل‌ها",
            ],
            "section_path": [
                "غزل‌ها",
                f"غزل {ordinal}",
            ],
            "parent_context_id": None,
            "boundaries": [],
        },
        "neighbors": {
            "previous_passage_id": (
                None
                if ordinal == 1
                else (
                    f"{version_id}:"
                    f"passage_{ordinal - 1:06d}"
                )
            ),
            "next_passage_id": (
                None
                if ordinal == 2
                else (
                    f"{version_id}:"
                    f"passage_{ordinal + 1:06d}"
                )
            ),
        },
        "text": {
            "display": (
                f"  متن نمایشی Passage {ordinal}\n"
                "سطر دوم  "
            ),
            "retrieval": (
                f"متن بازیابی Passage {ordinal}"
            ),
            "search_alias": (
                f"متن بازیابی Passage {ordinal}"
            ),
            "word_count": 5,
            "unit_count": 1,
            "member_count": 1,
        },
        "members": [],
        "issues": [],
    }


def _write_source(
    tmp_path: Path,
    *,
    records: list[
        dict[str, object]
    ] | None = None,
    manifest_line_count: int | None = None,
    manifest_sha256: str | None = None,
) -> tuple[
    Path,
    Path,
    Path,
]:
    root = (
        tmp_path
        / "passage_corpus"
    )

    versions = (
        root
        / "versions"
    )

    versions.mkdir(
        parents=True
    )

    source_path = (
        versions
        / f"{VERSION_ID}.jsonl"
    )

    source_records = (
        records
        if records is not None
        else [
            _passage(1),
            _passage(2),
        ]
    )

    source_bytes = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in source_records
    )

    source_path.write_bytes(
        source_bytes
    )

    source_hash = sha256(
        source_bytes
    ).hexdigest()

    manifest = {
        "schema_version": 1,
        "passage_schema_version": "1.0.0",
        "summary": {
            "version_count": 1,
            "indexable_version_count": 1,
            "reference_version_count": 0,
            "jsonl_file_count": 1,
            "passage_count": len(
                source_records
            ),
            "skipped_unit_count": 0,
        },
        "versions": [
            {
                "author_id": AUTHOR_ID,
                "work_id": WORK_ID,
                "version_id": VERSION_ID,
                "profile": "structured_poetry",
                "include_in_index": True,
                "is_canonical": True,
                "passage_count": len(
                    source_records
                ),
                "output": {
                    "path": (
                        f"versions/{VERSION_ID}.jsonl"
                    ),
                    "line_count": (
                        len(source_records)
                        if manifest_line_count
                        is None
                        else manifest_line_count
                    ),
                    "byte_count": len(
                        source_bytes
                    ),
                    "sha256": (
                        source_hash
                        if manifest_sha256
                        is None
                        else manifest_sha256
                    ),
                },
                "document_metadata": None,
            },
        ],
    }

    (
        root
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    corpus_manifest_path = (
        tmp_path
        / "corpus_manifest.yaml"
    )

    corpus_manifest_path.write_text(
        f"""
dataset:
  name: "TEST"
  repository: "https://example.invalid/test"
  commit: "abc123"

works:
  "{WORK_ID}":
    title_fa: "اثر آزمایشی"
    profile: "structured_poetry"
    canonical_version: "{VERSION_ID}"
    include_in_index: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    aliases_path = (
        tmp_path
        / "scope_aliases.yaml"
    )

    aliases_path.write_text(
        f"""
schema_version: "1.0.0"

in_corpus:
  authors:
    "{AUTHOR_ID}":
      label_fa: "نویسنده آزمایشی"
      aliases:
        - "نویسنده آزمایشی"

  works:
    "{WORK_ID}":
      label_fa: "اثر آزمایشی"
      aliases:
        - "اثر آزمایشی"

known_out_of_corpus:
  authors: {{}}
  works: {{}}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return (
        root,
        corpus_manifest_path,
        aliases_path,
    )


def _build(
    tmp_path: Path,
    *,
    records: list[
        dict[str, object]
    ] | None = None,
    manifest_line_count: int | None = None,
    manifest_sha256: str | None = None,
    verify_hashes: bool = True,
) -> tuple[
    Path,
    PassageStoreBuildReport,
]:
    (
        root,
        corpus_manifest_path,
        aliases_path,
    ) = _write_source(
        tmp_path,
        records=records,
        manifest_line_count=(
            manifest_line_count
        ),
        manifest_sha256=(
            manifest_sha256
        ),
    )

    output_path = (
        tmp_path
        / "passage_store.sqlite3"
    )

    report = build_passage_store(
        root,
        corpus_manifest_path=(
            corpus_manifest_path
        ),
        scope_aliases_path=(
            aliases_path
        ),
        output_path=output_path,
        verify_hashes=verify_hashes,
    )

    return (
        output_path,
        report,
    )


def test_builder_creates_readable_store(
    tmp_path: Path,
) -> None:
    output_path, report = _build(
        tmp_path
    )

    assert isinstance(
        report,
        PassageStoreBuildReport,
    )
    assert report.schema_version == (
        PASSAGE_STORE_SCHEMA_VERSION
    )
    assert report.output_path == (
        output_path.resolve()
    )
    assert report.passage_count == 2
    assert report.version_count == 1
    assert report.source_file_count == 1
    assert report.database_byte_count > 0

    store = PassageStore(
        output_path
    )

    assert store.passage_count == 2

    record = store.require(
        f"{VERSION_ID}:passage_000001"
    )

    assert record.author_id == (
        AUTHOR_ID
    )
    assert record.author_name == (
        "نویسنده آزمایشی"
    )
    assert record.work_id == WORK_ID
    assert record.work_title == (
        "اثر آزمایشی"
    )
    assert record.heading_path == (
        "غزل‌ها",
    )
    assert record.section_path == (
        "غزل‌ها",
        "غزل 1",
    )
    assert "\n" in record.display_text

    assert record.display_text == (
        "  متن نمایشی Passage 1\n"
        "سطر دوم  "
    )


def test_hash_mismatch_preserves_existing_output(
    tmp_path: Path,
) -> None:
    (
        root,
        corpus_manifest_path,
        aliases_path,
    ) = _write_source(
        tmp_path,
        manifest_sha256=(
            "0" * 64
        ),
    )

    output_path = (
        tmp_path
        / "passage_store.sqlite3"
    )

    output_path.write_bytes(
        b"existing-database"
    )

    with pytest.raises(
        PassageStoreBuildError,
        match="SHA-256 mismatch",
    ):
        build_passage_store(
            root,
            corpus_manifest_path=(
                corpus_manifest_path
            ),
            scope_aliases_path=(
                aliases_path
            ),
            output_path=output_path,
        )

    assert output_path.read_bytes() == (
        b"existing-database"
    )

    assert not output_path.with_name(
        output_path.name + ".tmp"
    ).exists()


def test_line_count_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    (
        root,
        corpus_manifest_path,
        aliases_path,
    ) = _write_source(
        tmp_path,
        manifest_line_count=99,
    )

    manifest_path = (
        root
        / "manifest.json"
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    payload["versions"][0][
        "passage_count"
    ] = 99

    manifest_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        PassageStoreBuildError,
        match="Line count mismatch",
    ):
        build_passage_store(
            root,
            corpus_manifest_path=(
                corpus_manifest_path
            ),
            scope_aliases_path=(
                aliases_path
            ),
            output_path=(
                tmp_path
                / "store.sqlite3"
            ),
        )


def test_wrong_record_version_is_rejected(
    tmp_path: Path,
) -> None:
    wrong_version = (
        "0001Author.Other.TEST-per1"
    )

    records = [
        _passage(
            1,
            version_id=wrong_version,
        ),
    ]

    (
        root,
        corpus_manifest_path,
        aliases_path,
    ) = _write_source(
        tmp_path,
        records=records,
    )

    with pytest.raises(
        PassageStoreBuildError,
        match="Version mismatch",
    ):
        build_passage_store(
            root,
            corpus_manifest_path=(
                corpus_manifest_path
            ),
            scope_aliases_path=(
                aliases_path
            ),
            output_path=(
                tmp_path
                / "store.sqlite3"
            ),
        )


def test_duplicate_passage_id_is_rejected(
    tmp_path: Path,
) -> None:
    duplicate_id = (
        f"{VERSION_ID}:"
        "passage_000001"
    )

    records = [
        _passage(
            1,
            passage_id=duplicate_id,
        ),
        _passage(
            2,
            passage_id=duplicate_id,
        ),
    ]

    (
        root,
        corpus_manifest_path,
        aliases_path,
    ) = _write_source(
        tmp_path,
        records=records,
    )

    with pytest.raises(
        PassageStoreBuildError,
        match="Duplicate passage ID",
    ):
        build_passage_store(
            root,
            corpus_manifest_path=(
                corpus_manifest_path
            ),
            scope_aliases_path=(
                aliases_path
            ),
            output_path=(
                tmp_path
                / "store.sqlite3"
            ),
        )


def test_hash_verification_can_be_disabled(
    tmp_path: Path,
) -> None:
    output_path, report = _build(
        tmp_path,
        manifest_sha256=(
            "0" * 64
        ),
        verify_hashes=False,
    )

    assert output_path.exists()
    assert report.passage_count == 2
    assert PassageStore(
        output_path
    ).passage_count == 2


def test_version_passage_count_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    (
        root,
        corpus_manifest_path,
        aliases_path,
    ) = _write_source(
        tmp_path
    )

    manifest_path = (
        root
        / "manifest.json"
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    payload["versions"][0][
        "passage_count"
    ] = 99

    manifest_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        PassageStoreBuildError,
        match=(
            "Version passage count mismatch"
        ),
    ):
        build_passage_store(
            root,
            corpus_manifest_path=(
                corpus_manifest_path
            ),
            scope_aliases_path=(
                aliases_path
            ),
            output_path=(
                tmp_path
                / "store.sqlite3"
            ),
        )


def test_missing_indexable_version_is_rejected(
    tmp_path: Path,
) -> None:
    (
        root,
        corpus_manifest_path,
        aliases_path,
    ) = _write_source(
        tmp_path
    )

    manifest_path = (
        root
        / "manifest.json"
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    payload["versions"] = []

    payload["summary"][
        "jsonl_file_count"
    ] = 0

    payload["summary"][
        "passage_count"
    ] = 0

    manifest_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        PassageStoreBuildError,
        match=(
            "Indexable version set mismatch"
        ),
    ):
        build_passage_store(
            root,
            corpus_manifest_path=(
                corpus_manifest_path
            ),
            scope_aliases_path=(
                aliases_path
            ),
            output_path=(
                tmp_path
                / "store.sqlite3"
            ),
        )
