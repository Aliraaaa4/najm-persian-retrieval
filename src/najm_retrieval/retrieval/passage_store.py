"""Read-only SQLite lookup for serialized passages."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import sqlite3

from najm_retrieval.retrieval.passage_store_models import (
    PASSAGE_STORE_SCHEMA_VERSION,
    PassageStoreRecord,
)


class PassageStoreError(RuntimeError):
    """Raised when a passage store is missing or invalid."""


class PassageStoreLookupError(
    PassageStoreError
):
    """Raised when a required passage does not exist."""


_REQUIRED_PASSAGE_COLUMNS = {
    "passage_id",
    "version_id",
    "author_id",
    "author_name",
    "work_id",
    "work_title",
    "profile",
    "kind",
    "ordinal",
    "display_text",
    "retrieval_text",
    "search_alias_text",
    "previous_passage_id",
    "next_passage_id",
    "heading_path_json",
    "section_path_json",
    "source_unit_ids_json",
    "word_count",
    "unit_count",
    "member_count",
}


_SELECT_COLUMNS = """
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
"""


class PassageStore:
    """Read passage text and metadata by stable passage ID."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = Path(
            database_path
        ).resolve()

        if not self.database_path.is_file():
            raise PassageStoreError(
                "Passage store database not found: "
                f"{self.database_path}"
            )

        try:
            with self._connect() as connection:
                self._validate_database(
                    connection
                )

                metadata_rows = (
                    connection.execute(
                        """
                        SELECT key, value
                        FROM metadata
                        """
                    ).fetchall()
                )

                metadata = {
                    str(row["key"]): str(
                        row["value"]
                    )
                    for row in metadata_rows
                }

                schema_version = metadata.get(
                    "passage_store_schema_version"
                )

                if (
                    schema_version
                    != PASSAGE_STORE_SCHEMA_VERSION
                ):
                    raise PassageStoreError(
                        "Unsupported passage store "
                        "schema version: "
                        f"{schema_version!r}; expected "
                        f"{PASSAGE_STORE_SCHEMA_VERSION!r}."
                    )

                raw_passage_count = metadata.get(
                    "passage_count"
                )

                try:
                    declared_count = int(
                        raw_passage_count
                    )
                except (
                    TypeError,
                    ValueError,
                ) as error:
                    raise PassageStoreError(
                        "Passage store metadata has "
                        "an invalid passage_count."
                    ) from error

                actual_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM passages
                        """
                    ).fetchone()[0]
                )

                if declared_count != actual_count:
                    raise PassageStoreError(
                        "Passage store count mismatch: "
                        f"metadata={declared_count}, "
                        f"database={actual_count}."
                    )

                self._metadata = metadata
                self._passage_count = (
                    actual_count
                )

        except PassageStoreError:
            raise

        except sqlite3.Error as error:
            raise PassageStoreError(
                "Could not open passage store "
                f"database: {self.database_path}"
            ) from error

    @property
    def schema_version(self) -> str:
        """Return the validated database schema version."""

        return self._metadata[
            "passage_store_schema_version"
        ]

    @property
    def passage_count(self) -> int:
        """Return the number of stored passages."""

        return self._passage_count

    def contains(
        self,
        passage_id: str,
    ) -> bool:
        """Return whether one passage ID exists."""

        self._validate_passage_id(
            passage_id
        )

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM passages
                WHERE passage_id = ?
                LIMIT 1
                """,
                (
                    passage_id,
                ),
            ).fetchone()

        return row is not None

    def get(
        self,
        passage_id: str,
    ) -> PassageStoreRecord | None:
        """Return one passage or None when it is absent."""

        self._validate_passage_id(
            passage_id
        )

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM passages
                WHERE passage_id = ?
                """,
                (
                    passage_id,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._record_from_row(
            row
        )

    def require(
        self,
        passage_id: str,
    ) -> PassageStoreRecord:
        """Return one passage or raise a clear lookup error."""

        record = self.get(
            passage_id
        )

        if record is None:
            raise PassageStoreLookupError(
                f"Unknown passage ID: {passage_id}"
            )

        return record

    def get_many(
        self,
        passage_ids: Iterable[str],
    ) -> tuple[
        PassageStoreRecord,
        ...,
    ]:
        """Return found records in requested order."""

        requested_ids = tuple(
            passage_ids
        )

        for passage_id in requested_ids:
            self._validate_passage_id(
                passage_id
            )

        if not requested_ids:
            return ()

        unique_ids = tuple(
            dict.fromkeys(
                requested_ids
            )
        )

        found: dict[
            str,
            PassageStoreRecord,
        ] = {}

        with self._connect() as connection:
            for start in range(
                0,
                len(unique_ids),
                500,
            ):
                chunk = unique_ids[
                    start : start + 500
                ]

                placeholders = ",".join(
                    "?"
                    for _ in chunk
                )

                rows = connection.execute(
                    f"""
                    SELECT {_SELECT_COLUMNS}
                    FROM passages
                    WHERE passage_id IN (
                        {placeholders}
                    )
                    """,
                    chunk,
                ).fetchall()

                for row in rows:
                    record = self._record_from_row(
                        row
                    )

                    found[
                        record.passage_id
                    ] = record

        return tuple(
            found[passage_id]
            for passage_id in requested_ids
            if passage_id in found
        )

    def _connect(
        self,
    ) -> sqlite3.Connection:
        uri = (
            self.database_path.as_uri()
            + "?mode=ro"
        )

        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=5.0,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA query_only = ON"
        )

        return connection

    @staticmethod
    def _validate_passage_id(
        passage_id: str,
    ) -> None:
        if (
            not isinstance(
                passage_id,
                str,
            )
            or not passage_id.strip()
        ):
            raise ValueError(
                "passage_id must be a "
                "non-empty string."
            )

    @staticmethod
    def _validate_database(
        connection: sqlite3.Connection,
    ) -> None:
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

        table_names = {
            str(row["name"])
            for row in table_rows
        }

        for required_table in (
            "metadata",
            "passages",
        ):
            if (
                required_table
                not in table_names
            ):
                raise PassageStoreError(
                    "Passage store is missing "
                    f"table: {required_table}"
                )

        column_rows = connection.execute(
            """
            PRAGMA table_info(passages)
            """
        ).fetchall()

        column_names = {
            str(row["name"])
            for row in column_rows
        }

        missing_columns = (
            _REQUIRED_PASSAGE_COLUMNS
            - column_names
        )

        if missing_columns:
            missing = ", ".join(
                sorted(
                    missing_columns
                )
            )

            raise PassageStoreError(
                "Passage store is missing "
                f"columns: {missing}"
            )

    @staticmethod
    def _decode_string_tuple(
        raw_value: str,
        *,
        field_name: str,
        passage_id: str,
    ) -> tuple[str, ...]:
        try:
            payload = json.loads(
                raw_value
            )
        except (
            TypeError,
            json.JSONDecodeError,
        ) as error:
            raise PassageStoreError(
                f"Invalid {field_name} JSON "
                f"for passage {passage_id}."
            ) from error

        if (
            not isinstance(
                payload,
                list,
            )
            or any(
                not isinstance(
                    item,
                    str,
                )
                or not item.strip()
                for item in payload
            )
        ):
            raise PassageStoreError(
                f"{field_name} must be a "
                f"JSON string array for "
                f"passage {passage_id}."
            )

        return tuple(
            payload
        )

    @classmethod
    def _record_from_row(
        cls,
        row: sqlite3.Row,
    ) -> PassageStoreRecord:
        passage_id = str(
            row["passage_id"]
        )

        return PassageStoreRecord(
            passage_id=passage_id,
            version_id=str(
                row["version_id"]
            ),
            author_id=str(
                row["author_id"]
            ),
            author_name=str(
                row["author_name"]
            ),
            work_id=str(
                row["work_id"]
            ),
            work_title=str(
                row["work_title"]
            ),
            profile=str(
                row["profile"]
            ),
            kind=str(
                row["kind"]
            ),
            ordinal=int(
                row["ordinal"]
            ),
            display_text=str(
                row["display_text"]
            ),
            retrieval_text=str(
                row["retrieval_text"]
            ),
            search_alias_text=str(
                row["search_alias_text"]
            ),
            previous_passage_id=(
                str(
                    row[
                        "previous_passage_id"
                    ]
                )
                if row[
                    "previous_passage_id"
                ]
                is not None
                else None
            ),
            next_passage_id=(
                str(
                    row[
                        "next_passage_id"
                    ]
                )
                if row[
                    "next_passage_id"
                ]
                is not None
                else None
            ),
            heading_path=(
                cls._decode_string_tuple(
                    row[
                        "heading_path_json"
                    ],
                    field_name=(
                        "heading_path"
                    ),
                    passage_id=passage_id,
                )
            ),
            section_path=(
                cls._decode_string_tuple(
                    row[
                        "section_path_json"
                    ],
                    field_name=(
                        "section_path"
                    ),
                    passage_id=passage_id,
                )
            ),
            source_unit_ids=(
                cls._decode_string_tuple(
                    row[
                        "source_unit_ids_json"
                    ],
                    field_name=(
                        "source_unit_ids"
                    ),
                    passage_id=passage_id,
                )
            ),
            word_count=int(
                row["word_count"]
            ),
            unit_count=int(
                row["unit_count"]
            ),
            member_count=int(
                row["member_count"]
            ),
        )


__all__ = [
    "PassageStore",
    "PassageStoreError",
    "PassageStoreLookupError",
]
