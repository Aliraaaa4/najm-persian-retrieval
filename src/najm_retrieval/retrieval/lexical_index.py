"""SQLite FTS5 lexical baseline for passage retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import sqlite3
import time

from najm_retrieval.normalization import (
    normalize_search_alias,
)
from najm_retrieval.retrieval.models import (
    LEXICAL_INDEX_SCHEMA_VERSION,
    LexicalIndexBuildReport,
    LexicalSearchMode,
    LexicalSearchResult,
    SearchHit,
)


_TOKEN_PATTERN = re.compile(
    r"[^\W_]+",
    flags=re.UNICODE,
)

_STOPWORDS = frozenset(
    {
        "از",
        "است",
        "این",
        "آن",
        "او",
        "با",
        "بر",
        "برای",
        "به",
        "بود",
        "تا",
        "تو",
        "خود",
        "در",
        "را",
        "شد",
        "شود",
        "که",
        "ما",
        "من",
        "می",
        "و",
        "یا",
        "یک",
    }
)


class LexicalIndexError(RuntimeError):
    """Raised when an FTS5 index cannot be built or searched."""


class LexicalIndex:
    """Read and search one generated SQLite FTS5 index."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = Path(
            database_path
        )

        if not self.database_path.is_file():
            raise LexicalIndexError(
                "Lexical index does not exist: "
                f"{self.database_path}"
            )

        self._validate_schema()

    def search(
        self,
        query_text: str,
        *,
        limit: int = 10,
        mode: LexicalSearchMode = (
            LexicalSearchMode.AUTO
        ),
    ) -> LexicalSearchResult:
        """Search passages with deterministic FTS5 ranking."""

        if not isinstance(
            mode,
            LexicalSearchMode,
        ):
            try:
                mode = LexicalSearchMode(mode)
            except ValueError as error:
                raise ValueError(
                    f"Unsupported search mode: {mode}"
                ) from error

        if not 1 <= limit <= 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        normalized = normalize_search_alias(
            query_text
        )

        terms = _query_terms(
            normalized
        )

        if not terms:
            raise ValueError(
                "Query contains no searchable terms."
            )

        candidate_modes = _candidate_modes(
            requested=mode,
            term_count=len(terms),
        )

        total_started_at = (
            time.perf_counter()
        )

        selected_mode = candidate_modes[-1]
        selected_rows: list[
            sqlite3.Row
        ] = []

        with self._connect() as connection:
            for candidate_mode in candidate_modes:
                expression = _match_expression(
                    normalized,
                    terms=terms,
                    mode=candidate_mode,
                )

                rows = connection.execute(
                    """
                    SELECT
                        passage_id,
                        version_id,
                        kind,
                        bm25(passages_fts)
                            AS bm25_score,
                        snippet(
                            passages_fts,
                            3,
                            '[',
                            ']',
                            ' … ',
                            24
                        ) AS snippet
                    FROM passages_fts
                    WHERE passages_fts MATCH ?
                    ORDER BY
                        bm25_score ASC,
                        passage_id ASC
                    LIMIT ?
                    """,
                    (
                        expression,
                        limit,
                    ),
                ).fetchall()

                selected_mode = candidate_mode
                selected_rows = rows

                if rows:
                    break

        latency_ms = (
            time.perf_counter()
            - total_started_at
        ) * 1000

        hits = tuple(
            SearchHit(
                passage_id=row[
                    "passage_id"
                ],
                version_id=row[
                    "version_id"
                ],
                kind=row["kind"],
                rank=rank,
                bm25_score=float(
                    row["bm25_score"]
                ),
                snippet=row["snippet"] or "",
            )
            for rank, row in enumerate(
                selected_rows,
                start=1,
            )
        )

        return LexicalSearchResult(
            query_text=query_text,
            normalized_query=normalized,
            mode_requested=mode,
            mode_used=selected_mode,
            hits=hits,
            latency_ms=latency_ms,
        )

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """Open one configured SQLite connection."""

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _validate_schema(self) -> None:
        """Validate schema version and required tables."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT value
                    FROM metadata
                    WHERE key = 'schema_version'
                    """
                ).fetchone()

                if row is None:
                    raise LexicalIndexError(
                        "Lexical index metadata is missing."
                    )

                if (
                    row["value"]
                    != LEXICAL_INDEX_SCHEMA_VERSION
                ):
                    raise LexicalIndexError(
                        "Unsupported lexical-index "
                        f"schema version: "
                        f"{row['value']}"
                    )

                table_row = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE
                        type = 'table'
                        AND name = 'passages_fts'
                    """
                ).fetchone()

                if table_row is None:
                    raise LexicalIndexError(
                        "passages_fts table is missing."
                    )

        except sqlite3.Error as error:
            raise LexicalIndexError(
                "Invalid lexical index database: "
                f"{self.database_path}"
            ) from error


def build_lexical_index(
    *,
    passage_root: str | Path,
    database_path: str | Path,
    overwrite: bool = False,
) -> LexicalIndexBuildReport:
    """Build one deterministic FTS5 index from passage JSONL."""

    root = Path(
        passage_root
    )

    output_path = Path(
        database_path
    )

    if not root.is_dir():
        raise LexicalIndexError(
            "Passage root does not exist: "
            f"{root}"
        )

    jsonl_paths = sorted(
        (root / "versions").glob(
            "*.jsonl"
        )
    )

    if not jsonl_paths:
        raise LexicalIndexError(
            "No passage JSONL files were found in: "
            f"{root / 'versions'}"
        )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise LexicalIndexError(
            "Lexical index already exists. "
            "Use overwrite=True to replace it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    started_at = time.perf_counter()

    passage_count = 0
    seen_passage_ids: set[str] = set()

    connection = sqlite3.connect(
        temporary_path
    )

    try:
        connection.execute(
            "PRAGMA journal_mode = OFF"
        )

        connection.execute(
            "PRAGMA synchronous = OFF"
        )

        connection.execute(
            "PRAGMA temp_store = MEMORY"
        )

        connection.execute(
            """
            CREATE TABLE metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE passages_fts
                USING fts5(
                    passage_id UNINDEXED,
                    version_id UNINDEXED,
                    kind UNINDEXED,
                    search_alias,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
        except sqlite3.OperationalError as error:
            raise LexicalIndexError(
                "Could not create SQLite FTS5 index: "
                f"{error}"
            ) from error

        with connection:
            for path in jsonl_paths:
                batch: list[
                    tuple[str, str, str, str]
                ] = []

                with path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    for line_number, line in enumerate(
                        file,
                        start=1,
                    ):
                        if not line.strip():
                            continue

                        record = _load_passage_record(
                            line,
                            path=path,
                            line_number=line_number,
                        )

                        passage_id = record[
                            "passage_id"
                        ]

                        if passage_id in seen_passage_ids:
                            raise LexicalIndexError(
                                "Duplicate passage ID: "
                                f"{passage_id}"
                            )

                        seen_passage_ids.add(
                            passage_id
                        )

                        batch.append(
                            (
                                passage_id,
                                record["version_id"],
                                record["kind"],
                                record[
                                    "search_alias"
                                ],
                            )
                        )

                        if len(batch) >= 500:
                            connection.executemany(
                                """
                                INSERT INTO passages_fts(
                                    passage_id,
                                    version_id,
                                    kind,
                                    search_alias
                                )
                                VALUES (?, ?, ?, ?)
                                """,
                                batch,
                            )

                            passage_count += len(
                                batch
                            )

                            batch.clear()

                if batch:
                    connection.executemany(
                        """
                        INSERT INTO passages_fts(
                            passage_id,
                            version_id,
                            kind,
                            search_alias
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        batch,
                    )

                    passage_count += len(
                        batch
                    )

            connection.executemany(
                """
                INSERT INTO metadata(
                    key,
                    value
                )
                VALUES (?, ?)
                """,
                (
                    (
                        "schema_version",
                        LEXICAL_INDEX_SCHEMA_VERSION,
                    ),
                    (
                        "passage_count",
                        str(passage_count),
                    ),
                    (
                        "source_file_count",
                        str(len(jsonl_paths)),
                    ),
                ),
            )

            connection.execute(
                """
                INSERT INTO passages_fts(
                    passages_fts
                )
                VALUES ('optimize')
                """
            )

        connection.close()

        temporary_path.replace(
            output_path
        )

    except Exception:
        connection.close()

        temporary_path.unlink(
            missing_ok=True
        )

        raise

    runtime_seconds = (
        time.perf_counter()
        - started_at
    )

    return LexicalIndexBuildReport(
        database_path=output_path,
        source_file_count=len(
            jsonl_paths
        ),
        passage_count=passage_count,
        runtime_seconds=runtime_seconds,
        database_bytes=(
            output_path.stat().st_size
        ),
    )


def _load_passage_record(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> dict[str, str]:
    """Load fields required by the lexical index."""

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise LexicalIndexError(
            "Invalid JSON in "
            f"{path} at line {line_number}."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise LexicalIndexError(
            "Passage record must be an object in "
            f"{path} at line {line_number}."
        )

    text = payload.get(
        "text"
    )

    if not isinstance(
        text,
        dict,
    ):
        raise LexicalIndexError(
            "Passage text must be an object in "
            f"{path} at line {line_number}."
        )

    values: dict[str, Any] = {
        "passage_id": payload.get(
            "passage_id"
        ),
        "version_id": payload.get(
            "version_id"
        ),
        "kind": payload.get(
            "kind"
        ),
        "search_alias": text.get(
            "search_alias"
        ),
    }

    for field_name, value in values.items():
        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise LexicalIndexError(
                f"{field_name} must be a "
                "nonempty string in "
                f"{path} at line {line_number}."
            )

    return {
        key: str(value)
        for key, value in values.items()
    }


def _query_terms(
    normalized_query: str,
) -> tuple[str, ...]:
    """Tokenize and deduplicate normalized query terms."""

    terms: list[str] = []
    seen: set[str] = set()

    for term in _TOKEN_PATTERN.findall(
        normalized_query
    ):
        if term in seen:
            continue

        seen.add(term)
        terms.append(term)

    return tuple(terms)


def _candidate_modes(
    *,
    requested: LexicalSearchMode,
    term_count: int,
) -> tuple[LexicalSearchMode, ...]:
    """Return fallback order for one requested mode."""

    if requested is not LexicalSearchMode.AUTO:
        return (
            requested,
        )

    if term_count == 1:
        return (
            LexicalSearchMode.ALL_TERMS,
            LexicalSearchMode.ANY_TERMS,
        )

    return (
        LexicalSearchMode.PHRASE,
        LexicalSearchMode.ALL_TERMS,
        LexicalSearchMode.ANY_TERMS,
    )


def _match_expression(
    normalized_query: str,
    *,
    terms: tuple[str, ...],
    mode: LexicalSearchMode,
) -> str:
    """Create one safely escaped FTS5 MATCH expression."""

    if mode is LexicalSearchMode.PHRASE:
        return _quote_fts(
            normalized_query
        )

    if mode is LexicalSearchMode.ALL_TERMS:
        return " AND ".join(
            _quote_fts(term)
            for term in terms
        )

    if mode is LexicalSearchMode.ANY_TERMS:
        important_terms = tuple(
            term
            for term in terms
            if (
                term not in _STOPWORDS
                and len(term) > 1
            )
        )

        relaxed_terms = (
            important_terms
            if important_terms
            else terms
        )

        return " OR ".join(
            _quote_fts(term)
            for term in relaxed_terms
        )

    raise ValueError(
        "AUTO mode must be resolved before "
        "building a MATCH expression."
    )


def _quote_fts(
    value: str,
) -> str:
    """Escape one FTS5 phrase or token."""

    return (
        '"'
        + value.replace(
            '"',
            '""',
        )
        + '"'
    )


__all__ = [
    "LexicalIndex",
    "LexicalIndexError",
    "build_lexical_index",
]