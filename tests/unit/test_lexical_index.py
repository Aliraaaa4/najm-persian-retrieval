"""Tests for the SQLite FTS5 lexical baseline."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from najm_retrieval.retrieval import (
    LexicalIndex,
    LexicalIndexError,
    LexicalSearchMode,
    build_lexical_index,
)


def _write_passages(
    root: Path,
) -> None:
    versions = root / "versions"
    versions.mkdir(
        parents=True
    )

    records = [
        {
            "passage_id": (
                "mathnawi:passage_000001"
            ),
            "version_id": "mathnawi",
            "kind": "mathnawi",
            "text": {
                "search_alias": (
                    "بشنو این نی چون شکایت "
                    "می کند از جداییها "
                    "حکایت می کند"
                ),
            },
        },
        {
            "passage_id": (
                "diwan:passage_000001"
            ),
            "version_id": "diwan",
            "kind": "diwan",
            "text": {
                "search_alias": (
                    "ای رستخیز ناگهان وی "
                    "رحمت بی منتها"
                ),
            },
        },
        {
            "passage_id": (
                "diwan:passage_000002"
            ),
            "version_id": "diwan",
            "kind": "diwan",
            "text": {
                "search_alias": (
                    "بخشش و فضل خدا بر "
                    "مستمندان آمد"
                ),
            },
        },
    ]

    path = (
        versions / "sample.jsonl"
    )

    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_build_and_phrase_search(
    tmp_path: Path,
) -> None:
    """Exact Persian phrase is found at rank one."""

    passage_root = (
        tmp_path / "passages"
    )

    _write_passages(
        passage_root
    )

    database_path = (
        tmp_path / "lexical.sqlite3"
    )

    report = build_lexical_index(
        passage_root=passage_root,
        database_path=database_path,
    )

    assert report.passage_count == 3
    assert report.source_file_count == 1

    index = LexicalIndex(
        database_path
    )

    result = index.search(
        "بشنو این نی چون شکایت می‌کند",
        mode=LexicalSearchMode.PHRASE,
    )

    assert result.mode_used is (
        LexicalSearchMode.PHRASE
    )

    assert result.hits[0].passage_id == (
        "mathnawi:passage_000001"
    )


def test_auto_falls_back_from_phrase_to_all_terms(
    tmp_path: Path,
) -> None:
    """AUTO uses AND when query order differs from text."""

    passage_root = (
        tmp_path / "passages"
    )

    _write_passages(
        passage_root
    )

    database_path = (
        tmp_path / "lexical.sqlite3"
    )

    build_lexical_index(
        passage_root=passage_root,
        database_path=database_path,
    )

    result = LexicalIndex(
        database_path
    ).search(
        "شکایت نی بشنو",
    )

    assert result.mode_used is (
        LexicalSearchMode.ALL_TERMS
    )

    assert result.hits[0].passage_id == (
        "mathnawi:passage_000001"
    )


def test_any_terms_filters_common_stopwords(
    tmp_path: Path,
) -> None:
    """Relaxed search emphasizes informative Persian terms."""

    passage_root = (
        tmp_path / "passages"
    )

    _write_passages(
        passage_root
    )

    database_path = (
        tmp_path / "lexical.sqlite3"
    )

    build_lexical_index(
        passage_root=passage_root,
        database_path=database_path,
    )

    result = LexicalIndex(
        database_path
    ).search(
        "و بخشش فضل خدا",
        mode=LexicalSearchMode.ANY_TERMS,
    )

    assert result.hits[0].passage_id == (
        "diwan:passage_000002"
    )


def test_build_rejects_existing_index_without_overwrite(
    tmp_path: Path,
) -> None:
    """Existing databases require explicit replacement."""

    passage_root = (
        tmp_path / "passages"
    )

    _write_passages(
        passage_root
    )

    database_path = (
        tmp_path / "lexical.sqlite3"
    )

    build_lexical_index(
        passage_root=passage_root,
        database_path=database_path,
    )

    with pytest.raises(
        LexicalIndexError,
        match="already exists",
    ):
        build_lexical_index(
            passage_root=passage_root,
            database_path=database_path,
        )

    second = build_lexical_index(
        passage_root=passage_root,
        database_path=database_path,
        overwrite=True,
    )

    assert second.passage_count == 3


def test_empty_query_is_rejected(
    tmp_path: Path,
) -> None:
    """Queries without searchable tokens are invalid."""

    passage_root = (
        tmp_path / "passages"
    )

    _write_passages(
        passage_root
    )

    database_path = (
        tmp_path / "lexical.sqlite3"
    )

    build_lexical_index(
        passage_root=passage_root,
        database_path=database_path,
    )

    index = LexicalIndex(
        database_path
    )

    with pytest.raises(
        ValueError,
        match="no searchable terms",
    ):
        index.search(
            "   !!!   "
        )


def test_search_limit_is_validated(
    tmp_path: Path,
) -> None:
    """Search result limits stay within API-safe bounds."""

    passage_root = (
        tmp_path / "passages"
    )

    _write_passages(
        passage_root
    )

    database_path = (
        tmp_path / "lexical.sqlite3"
    )

    build_lexical_index(
        passage_root=passage_root,
        database_path=database_path,
    )

    index = LexicalIndex(
        database_path
    )

    with pytest.raises(
        ValueError,
        match="between 1 and 100",
    ):
        index.search(
            "بشنو",
            limit=0,
        )