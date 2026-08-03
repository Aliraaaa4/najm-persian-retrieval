"""Tests for the read-only SQLite passage store."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from najm_retrieval.retrieval import (
    PASSAGE_STORE_SCHEMA_VERSION,
    PassageStore,
    PassageStoreError,
    PassageStoreLookupError,
    PassageStoreRecord,
)


PASSAGE_ONE = (
    "0672JalalDinRumi."
    "Mathnawi.PDL00048-per1:"
    "passage_000001"
)

PASSAGE_TWO = (
    "0672JalalDinRumi."
    "Mathnawi.PDL00048-per1:"
    "passage_000002"
)

VERSION_ID = (
    "0672JalalDinRumi."
    "Mathnawi.PDL00048-per1"
)


def _create_database(
    tmp_path: Path,
    *,
    schema_version: str = (
        PASSAGE_STORE_SCHEMA_VERSION
    ),
    declared_count: int | None = None,
    malformed_heading: bool = False,
) -> Path:
    path = (
        tmp_path
        / "passage_store.sqlite3"
    )

    connection = sqlite3.connect(
        path
    )

    try:
        connection.executescript(
            """
            CREATE TABLE metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE passages(
                passage_id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                work_id TEXT NOT NULL,
                work_title TEXT NOT NULL,
                profile TEXT NOT NULL,
                kind TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                display_text TEXT NOT NULL,
                retrieval_text TEXT NOT NULL,
                search_alias_text TEXT NOT NULL,
                previous_passage_id TEXT,
                next_passage_id TEXT,
                heading_path_json TEXT NOT NULL,
                section_path_json TEXT NOT NULL,
                source_unit_ids_json TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                unit_count INTEGER NOT NULL,
                member_count INTEGER NOT NULL
            );
            """
        )

        rows = (
            (
                PASSAGE_ONE,
                1,
                "  متن   نمایشی\nبلند  ",
                None,
                PASSAGE_TWO,
            ),
            (
                PASSAGE_TWO,
                2,
                "متن Passage دوم",
                PASSAGE_ONE,
                None,
            ),
        )

        for (
            passage_id,
            ordinal,
            display_text,
            previous_passage_id,
            next_passage_id,
        ) in rows:
            heading_path_json = json.dumps(
                [
                    "دفتر اول",
                ],
                ensure_ascii=False,
            )

            if (
                malformed_heading
                and passage_id
                == PASSAGE_ONE
            ):
                heading_path_json = (
                    "not-json"
                )

            connection.execute(
                """
                INSERT INTO passages(
                    passage_id,
                    version_id,
                    author_id,
                    author_name,
                    work_id,
                    work_title,
                    profile,
                    kind,
                    ordinal,
                    display_text,
                    retrieval_text,
                    search_alias_text,
                    previous_passage_id,
                    next_passage_id,
                    heading_path_json,
                    section_path_json,
                    source_unit_ids_json,
                    word_count,
                    unit_count,
                    member_count
                )
                VALUES(
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    passage_id,
                    VERSION_ID,
                    "0672JalalDinRumi",
                    "مولانا جلال‌الدین رومی",
                    (
                        "0672JalalDinRumi."
                        "Mathnawi"
                    ),
                    "مثنوی معنوی",
                    "structured_poetry",
                    "mathnawi",
                    ordinal,
                    display_text,
                    "متن بازیابی",
                    "متن بازیابی",
                    previous_passage_id,
                    next_passage_id,
                    heading_path_json,
                    json.dumps(
                        [
                            "دفتر اول",
                            "بخش آغازین",
                        ],
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        [
                            f"{VERSION_ID}:"
                            f"verse_{ordinal:04d}"
                        ],
                        ensure_ascii=False,
                    ),
                    12,
                    1,
                    1,
                ),
            )

        actual_count = len(
            rows
        )

        connection.executemany(
            """
            INSERT INTO metadata(
                key,
                value
            )
            VALUES(
                ?,
                ?
            )
            """,
            (
                (
                    "passage_store_schema_version",
                    schema_version,
                ),
                (
                    "passage_count",
                    str(
                        actual_count
                        if declared_count
                        is None
                        else declared_count
                    ),
                ),
            ),
        )

        connection.commit()

    finally:
        connection.close()

    return path


def test_store_loads_record_and_metadata(
    tmp_path: Path,
) -> None:
    store = PassageStore(
        _create_database(
            tmp_path
        )
    )

    assert store.schema_version == (
        PASSAGE_STORE_SCHEMA_VERSION
    )
    assert store.passage_count == 2
    assert store.contains(
        PASSAGE_ONE
    )

    record = store.require(
        PASSAGE_ONE
    )

    assert isinstance(
        record,
        PassageStoreRecord,
    )
    assert record.passage_id == (
        PASSAGE_ONE
    )
    assert record.author_name == (
        "مولانا جلال‌الدین رومی"
    )
    assert record.work_title == (
        "مثنوی معنوی"
    )
    assert record.heading_path == (
        "دفتر اول",
    )
    assert record.section_path == (
        "دفتر اول",
        "بخش آغازین",
    )
    assert (
        record.next_passage_id
        == PASSAGE_TWO
    )


def test_get_many_preserves_requested_order(
    tmp_path: Path,
) -> None:
    store = PassageStore(
        _create_database(
            tmp_path
        )
    )

    records = store.get_many(
        (
            PASSAGE_TWO,
            "missing:passage_000001",
            PASSAGE_ONE,
            PASSAGE_TWO,
        )
    )

    assert tuple(
        record.passage_id
        for record in records
    ) == (
        PASSAGE_TWO,
        PASSAGE_ONE,
        PASSAGE_TWO,
    )


def test_require_raises_for_missing_passage(
    tmp_path: Path,
) -> None:
    store = PassageStore(
        _create_database(
            tmp_path
        )
    )

    assert (
        store.get(
            "missing:passage_000001"
        )
        is None
    )

    with pytest.raises(
        PassageStoreLookupError,
        match="Unknown passage ID",
    ):
        store.require(
            "missing:passage_000001"
        )


def test_missing_database_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        PassageStoreError,
        match="not found",
    ):
        PassageStore(
            tmp_path
            / "missing.sqlite3"
        )


def test_schema_version_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    path = _create_database(
        tmp_path,
        schema_version="999.0.0",
    )

    with pytest.raises(
        PassageStoreError,
        match="schema version",
    ):
        PassageStore(
            path
        )


def test_declared_count_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    path = _create_database(
        tmp_path,
        declared_count=99,
    )

    with pytest.raises(
        PassageStoreError,
        match="count mismatch",
    ):
        PassageStore(
            path
        )


def test_invalid_json_array_is_rejected_on_lookup(
    tmp_path: Path,
) -> None:
    store = PassageStore(
        _create_database(
            tmp_path,
            malformed_heading=True,
        )
    )

    with pytest.raises(
        PassageStoreError,
        match="heading_path",
    ):
        store.require(
            PASSAGE_ONE
        )


def test_record_snippet_compacts_and_truncates(
    tmp_path: Path,
) -> None:
    store = PassageStore(
        _create_database(
            tmp_path
        )
    )

    snippet = store.require(
        PASSAGE_ONE
    ).snippet(
        max_chars=10
    )

    assert "\n" not in snippet
    assert "  " not in snippet
    assert len(snippet) <= 10
    assert snippet.endswith("…")
