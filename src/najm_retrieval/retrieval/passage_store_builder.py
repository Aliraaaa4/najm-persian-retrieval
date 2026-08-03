"""Build a validated SQLite passage store from passage JSONL files."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any

from najm_retrieval.corpus.manifest import (
    CorpusManifest,
    ManifestError,
    load_manifest,
)
from najm_retrieval.retrieval.passage_store import (
    PassageStoreError,
)
from najm_retrieval.retrieval.passage_store_models import (
    PASSAGE_STORE_SCHEMA_VERSION,
    PassageStoreBuildReport,
)
from najm_retrieval.retrieval.scope_catalog import (
    CorpusScopeCatalog,
    ScopeCatalogError,
)


class PassageStoreBuildError(
    PassageStoreError
):
    """Raised when source artifacts cannot produce a valid store."""


@dataclass(frozen=True)
class _VersionSource:
    author_id: str
    author_name: str
    work_id: str
    work_title: str
    version_id: str
    profile: str
    path: Path
    expected_line_count: int
    expected_byte_count: int
    expected_sha256: str


_CREATE_SCHEMA_SQL = """
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
    ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
    display_text TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    search_alias_text TEXT NOT NULL,
    previous_passage_id TEXT,
    next_passage_id TEXT,
    heading_path_json TEXT NOT NULL,
    section_path_json TEXT NOT NULL,
    source_unit_ids_json TEXT NOT NULL,
    word_count INTEGER NOT NULL CHECK(word_count >= 0),
    unit_count INTEGER NOT NULL CHECK(unit_count >= 0),
    member_count INTEGER NOT NULL CHECK(member_count >= 0),
    UNIQUE(version_id, ordinal)
);

CREATE INDEX passages_version_id_index
ON passages(version_id);

CREATE INDEX passages_work_id_index
ON passages(work_id);

CREATE INDEX passages_author_id_index
ON passages(author_id);
"""


_INSERT_PASSAGE_SQL = """
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
"""


def build_passage_store(
    passage_corpus_root: str | Path,
    *,
    corpus_manifest_path: str | Path,
    scope_aliases_path: str | Path,
    output_path: str | Path,
    verify_hashes: bool = True,
) -> PassageStoreBuildReport:
    """Build and atomically publish one validated passage database."""

    root = Path(
        passage_corpus_root
    ).resolve()

    if not root.is_dir():
        raise PassageStoreBuildError(
            "Passage corpus directory not found: "
            f"{root}"
        )

    source_manifest_path = (
        root
        / "manifest.json"
    )

    source_manifest = _load_json_object(
        source_manifest_path,
        description="passage corpus manifest",
    )

    try:
        corpus_manifest = load_manifest(
            corpus_manifest_path
        )

        scope_catalog = (
            CorpusScopeCatalog.from_files(
                manifest_path=(
                    corpus_manifest_path
                ),
                aliases_path=(
                    scope_aliases_path
                ),
            )
        )

    except (
        ManifestError,
        ScopeCatalogError,
    ) as error:
        raise PassageStoreBuildError(
            str(error)
        ) from error

    sources = _build_version_sources(
        root=root,
        source_manifest=source_manifest,
        corpus_manifest=corpus_manifest,
        scope_catalog=scope_catalog,
    )

    output = Path(
        output_path
    ).resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output.with_name(
        output.name + ".tmp"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    source_manifest_sha256 = _sha256_file(
        source_manifest_path
    )

    passage_count = 0

    try:
        connection = sqlite3.connect(
            temporary_path
        )

        try:
            connection.execute(
                "PRAGMA journal_mode = OFF"
            )
            connection.execute(
                "PRAGMA synchronous = FULL"
            )
            connection.execute(
                "PRAGMA temp_store = MEMORY"
            )

            connection.executescript(
                _CREATE_SCHEMA_SQL
            )

            for source in sources:
                passage_count += (
                    _insert_version_passages(
                        connection,
                        source=source,
                        verify_hashes=(
                            verify_hashes
                        ),
                        passage_schema_version=(
                            _require_text(
                                source_manifest,
                                "passage_schema_version",
                                context=(
                                    "passage corpus manifest"
                                ),
                            )
                        ),
                    )
                )

            declared_total = _manifest_summary_int(
                source_manifest,
                "passage_count",
            )

            if passage_count != declared_total:
                raise PassageStoreBuildError(
                    "Passage count mismatch: "
                    f"manifest={declared_total}, "
                    f"inserted={passage_count}."
                )

            metadata = {
                "passage_store_schema_version": (
                    PASSAGE_STORE_SCHEMA_VERSION
                ),
                "passage_count": str(
                    passage_count
                ),
                "version_count": str(
                    len(sources)
                ),
                "source_file_count": str(
                    len(sources)
                ),
                "source_manifest_schema_version": str(
                    source_manifest.get(
                        "schema_version"
                    )
                ),
                "source_passage_schema_version": (
                    _require_text(
                        source_manifest,
                        "passage_schema_version",
                        context=(
                            "passage corpus manifest"
                        ),
                    )
                ),
                "source_manifest_sha256": (
                    source_manifest_sha256
                ),
                "dataset_name": (
                    corpus_manifest.dataset.name
                ),
                "dataset_commit": (
                    corpus_manifest.dataset.commit
                ),
            }

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
                tuple(
                    sorted(
                        metadata.items()
                    )
                ),
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        temporary_path.replace(
            output
        )

    except PassageStoreBuildError:
        temporary_path.unlink(
            missing_ok=True
        )
        raise

    except (
        OSError,
        sqlite3.Error,
    ) as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise PassageStoreBuildError(
            "Could not build passage store: "
            f"{error}"
        ) from error

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise

    return PassageStoreBuildReport(
        output_path=output,
        passage_count=passage_count,
        version_count=len(
            sources
        ),
        source_file_count=len(
            sources
        ),
        database_byte_count=(
            output.stat().st_size
        ),
        source_manifest_sha256=(
            source_manifest_sha256
        ),
    )


def _build_version_sources(
    *,
    root: Path,
    source_manifest: dict[str, Any],
    corpus_manifest: CorpusManifest,
    scope_catalog: CorpusScopeCatalog,
) -> tuple[_VersionSource, ...]:
    schema_version = source_manifest.get(
        "schema_version"
    )

    if schema_version != 1:
        raise PassageStoreBuildError(
            "Unsupported passage corpus "
            f"manifest schema: {schema_version!r}."
        )

    _require_text(
        source_manifest,
        "passage_schema_version",
        context="passage corpus manifest",
    )

    versions = source_manifest.get(
        "versions"
    )

    if not isinstance(
        versions,
        list,
    ):
        raise PassageStoreBuildError(
            "'versions' must be a list in "
            "the passage corpus manifest."
        )

    sources: list[
        _VersionSource
    ] = []

    seen_version_ids: set[str] = set()

    indexable_versions = set(
        corpus_manifest.indexable_versions
    )

    for index, value in enumerate(
        versions,
        start=1,
    ):
        context = (
            "passage corpus manifest "
            f"version {index}"
        )

        version = _require_mapping(
            value,
            context=context,
        )

        include_in_index = version.get(
            "include_in_index"
        )

        if not isinstance(
            include_in_index,
            bool,
        ):
            raise PassageStoreBuildError(
                f"'include_in_index' must be "
                f"boolean in {context}."
            )

        if not include_in_index:
            continue

        author_id = _require_text(
            version,
            "author_id",
            context=context,
        )

        work_id = _require_text(
            version,
            "work_id",
            context=context,
        )

        version_id = _require_text(
            version,
            "version_id",
            context=context,
        )

        profile = _require_text(
            version,
            "profile",
            context=context,
        )

        if version_id in seen_version_ids:
            raise PassageStoreBuildError(
                "Duplicate version ID in "
                f"passage manifest: {version_id}"
            )

        seen_version_ids.add(
            version_id
        )

        if (
            not work_id.startswith(
                author_id + "."
            )
        ):
            raise PassageStoreBuildError(
                f"Work {work_id!r} does not "
                f"belong to author {author_id!r}."
            )

        try:
            work = corpus_manifest.get_work(
                work_id
            )
        except ManifestError as error:
            raise PassageStoreBuildError(
                str(error)
            ) from error

        if (
            version_id
            not in work.version_ids
        ):
            raise PassageStoreBuildError(
                f"Version {version_id!r} does "
                f"not belong to work {work_id!r}."
            )

        if (
            version_id
            not in indexable_versions
        ):
            raise PassageStoreBuildError(
                f"Version {version_id!r} is "
                "not indexable in the project manifest."
            )

        try:
            author_entity = (
                scope_catalog.get_entity(
                    author_id
                )
            )
        except ScopeCatalogError as error:
            raise PassageStoreBuildError(
                str(error)
            ) from error

        output = _require_mapping(
            version.get("output"),
            context=(
                f"output metadata for {version_id}"
            ),
        )

        declared_passage_count = (
            _require_non_negative_int(
                version,
                "passage_count",
                context=context,
            )
        )

        expected_line_count = (
            _require_non_negative_int(
                output,
                "line_count",
                context=(
                    f"output metadata for "
                    f"{version_id}"
                ),
            )
        )

        if (
            declared_passage_count
            != expected_line_count
        ):
            raise PassageStoreBuildError(
                "Version passage count mismatch "
                f"for {version_id}: "
                f"version={declared_passage_count}, "
                f"output={expected_line_count}."
            )

        relative_path_text = (
            _require_text(
                output,
                "path",
                context=(
                    f"output metadata for "
                    f"{version_id}"
                ),
            )
        )

        relative_path = Path(
            relative_path_text
        )

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise PassageStoreBuildError(
                "Passage JSONL path must be "
                f"relative and contained: {relative_path}"
            )

        source_path = (
            root
            / relative_path
        ).resolve()

        try:
            source_path.relative_to(
                root
            )
        except ValueError as error:
            raise PassageStoreBuildError(
                "Passage JSONL path escapes "
                f"the corpus directory: {source_path}"
            ) from error

        sources.append(
            _VersionSource(
                author_id=author_id,
                author_name=(
                    author_entity.label_fa
                ),
                work_id=work_id,
                work_title=work.title_fa,
                version_id=version_id,
                profile=profile,
                path=source_path,
                expected_line_count=(
                    expected_line_count
                ),
                expected_byte_count=(
                    _require_non_negative_int(
                        output,
                        "byte_count",
                        context=(
                            f"output metadata for "
                            f"{version_id}"
                        ),
                    )
                ),
                expected_sha256=(
                    _require_sha256(
                        output,
                        "sha256",
                        context=(
                            f"output metadata for "
                            f"{version_id}"
                        ),
                    )
                ),
            )
        )

    resolved_version_ids = {
        source.version_id
        for source in sources
    }

    if (
        resolved_version_ids
        != indexable_versions
    ):
        missing = sorted(
            indexable_versions
            - resolved_version_ids
        )

        unexpected = sorted(
            resolved_version_ids
            - indexable_versions
        )

        raise PassageStoreBuildError(
            "Indexable version set mismatch: "
            f"missing={missing}, "
            f"unexpected={unexpected}."
        )

    declared_files = _manifest_summary_int(
        source_manifest,
        "jsonl_file_count",
    )

    if declared_files != len(
        sources
    ):
        raise PassageStoreBuildError(
            "JSONL file count mismatch: "
            f"manifest={declared_files}, "
            f"resolved={len(sources)}."
        )

    return tuple(
        sources
    )


def _insert_version_passages(
    connection: sqlite3.Connection,
    *,
    source: _VersionSource,
    verify_hashes: bool,
    passage_schema_version: str,
) -> int:
    if not source.path.is_file():
        raise PassageStoreBuildError(
            "Passage JSONL file not found: "
            f"{source.path}"
        )

    actual_byte_count = (
        source.path.stat().st_size
    )

    if (
        actual_byte_count
        != source.expected_byte_count
    ):
        raise PassageStoreBuildError(
            "Byte count mismatch for "
            f"{source.version_id}: "
            f"manifest={source.expected_byte_count}, "
            f"actual={actual_byte_count}."
        )

    if verify_hashes:
        actual_sha256 = _sha256_file(
            source.path
        )

        if (
            actual_sha256
            != source.expected_sha256
        ):
            raise PassageStoreBuildError(
                "SHA-256 mismatch for "
                f"{source.version_id}: "
                f"manifest={source.expected_sha256}, "
                f"actual={actual_sha256}."
            )

    line_count = 0

    try:
        handle = source.path.open(
            "r",
            encoding="utf-8",
        )
    except (
        OSError,
        UnicodeError,
    ) as error:
        raise PassageStoreBuildError(
            "Could not read passage JSONL: "
            f"{source.path}"
        ) from error

    with handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                raise PassageStoreBuildError(
                    "Blank JSONL line in "
                    f"{source.path}:{line_number}."
                )

            try:
                payload = json.loads(
                    line
                )
            except json.JSONDecodeError as error:
                raise PassageStoreBuildError(
                    "Invalid JSON in "
                    f"{source.path}:{line_number}."
                ) from error

            row = _build_passage_row(
                payload,
                source=source,
                line_number=line_number,
                passage_schema_version=(
                    passage_schema_version
                ),
            )

            try:
                connection.execute(
                    _INSERT_PASSAGE_SQL,
                    row,
                )
            except sqlite3.IntegrityError as error:
                raise PassageStoreBuildError(
                    "Duplicate passage ID or "
                    "ordinal while inserting "
                    f"{source.path}:{line_number}."
                ) from error

            line_count += 1

    if (
        line_count
        != source.expected_line_count
    ):
        raise PassageStoreBuildError(
            "Line count mismatch for "
            f"{source.version_id}: "
            f"manifest={source.expected_line_count}, "
            f"actual={line_count}."
        )

    return line_count


def _build_passage_row(
    raw_value: Any,
    *,
    source: _VersionSource,
    line_number: int,
    passage_schema_version: str,
) -> tuple[Any, ...]:
    context = (
        f"{source.path}:{line_number}"
    )

    passage = _require_mapping(
        raw_value,
        context=context,
    )

    schema_version = _require_text(
        passage,
        "schema_version",
        context=context,
    )

    if (
        schema_version
        != passage_schema_version
    ):
        raise PassageStoreBuildError(
            "Passage schema mismatch in "
            f"{context}: expected "
            f"{passage_schema_version!r}, "
            f"found {schema_version!r}."
        )

    passage_id = _require_text(
        passage,
        "passage_id",
        context=context,
    )

    version_id = _require_text(
        passage,
        "version_id",
        context=context,
    )

    if version_id != source.version_id:
        raise PassageStoreBuildError(
            "Version mismatch in "
            f"{context}: expected "
            f"{source.version_id!r}, "
            f"found {version_id!r}."
        )

    if not passage_id.startswith(
        source.version_id + ":"
    ):
        raise PassageStoreBuildError(
            "Passage ID does not belong "
            f"to version in {context}: "
            f"{passage_id}"
        )

    include_in_index = passage.get(
        "include_in_index"
    )

    if include_in_index is not True:
        raise PassageStoreBuildError(
            "Stored passage must have "
            f"include_in_index=true in {context}."
        )

    profile = _require_text(
        passage,
        "profile",
        context=context,
    )

    if profile != source.profile:
        raise PassageStoreBuildError(
            "Profile mismatch in "
            f"{context}: expected "
            f"{source.profile!r}, "
            f"found {profile!r}."
        )

    kind = _require_text(
        passage,
        "kind",
        context=context,
    )

    ordinal = _require_positive_int(
        passage,
        "ordinal",
        context=context,
    )

    text = _require_mapping(
        passage.get("text"),
        context=f"text in {context}",
    )

    source_payload = _require_mapping(
        passage.get("source"),
        context=f"source in {context}",
    )

    context_payload = _require_mapping(
        passage.get("context"),
        context=f"context in {context}",
    )

    neighbors = _require_mapping(
        passage.get("neighbors"),
        context=f"neighbors in {context}",
    )

    return (
        passage_id,
        version_id,
        source.author_id,
        source.author_name,
        source.work_id,
        source.work_title,
        profile,
        kind,
        ordinal,
        _require_preserved_text(
            text,
            "display",
            context=f"text in {context}",
        ),
        _require_preserved_text(
            text,
            "retrieval",
            context=f"text in {context}",
        ),
        _require_preserved_text(
            text,
            "search_alias",
            context=f"text in {context}",
        ),
        _optional_text(
            neighbors.get(
                "previous_passage_id"
            ),
            field_name=(
                "previous_passage_id"
            ),
            context=context,
        ),
        _optional_text(
            neighbors.get(
                "next_passage_id"
            ),
            field_name=(
                "next_passage_id"
            ),
            context=context,
        ),
        _encode_string_list(
            context_payload.get(
                "heading_path"
            ),
            field_name="heading_path",
            context=context,
        ),
        _encode_string_list(
            context_payload.get(
                "section_path"
            ),
            field_name="section_path",
            context=context,
        ),
        _encode_string_list(
            source_payload.get(
                "unit_ids"
            ),
            field_name="source.unit_ids",
            context=context,
        ),
        _require_non_negative_int(
            text,
            "word_count",
            context=f"text in {context}",
        ),
        _require_non_negative_int(
            text,
            "unit_count",
            context=f"text in {context}",
        ),
        _require_non_negative_int(
            text,
            "member_count",
            context=f"text in {context}",
        ),
    )


def _load_json_object(
    path: Path,
    *,
    description: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise PassageStoreBuildError(
            f"{description.capitalize()} "
            f"not found: {path}"
        )

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise PassageStoreBuildError(
            f"Could not load {description}: {path}"
        ) from error

    return _require_mapping(
        value,
        context=description,
    )


def _require_mapping(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        raise PassageStoreBuildError(
            f"{context} must be a JSON object."
        )

    return value


def _require_text(
    mapping: dict[str, Any],
    field_name: str,
    *,
    context: str,
) -> str:
    value = mapping.get(
        field_name
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise PassageStoreBuildError(
            f"'{field_name}' must be a "
            f"non-empty string in {context}."
        )

    return value.strip()


def _require_non_negative_int(
    mapping: dict[str, Any],
    field_name: str,
    *,
    context: str,
) -> int:
    value = mapping.get(
        field_name
    )

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
        or value < 0
    ):
        raise PassageStoreBuildError(
            f"'{field_name}' must be a "
            f"non-negative integer in {context}."
        )

    return value


def _require_positive_int(
    mapping: dict[str, Any],
    field_name: str,
    *,
    context: str,
) -> int:
    value = _require_non_negative_int(
        mapping,
        field_name,
        context=context,
    )

    if value < 1:
        raise PassageStoreBuildError(
            f"'{field_name}' must be at "
            f"least 1 in {context}."
        )

    return value


def _optional_text(
    value: Any,
    *,
    field_name: str,
    context: str,
) -> str | None:
    if value is None:
        return None

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise PassageStoreBuildError(
            f"'{field_name}' must be null "
            f"or a non-empty string in {context}."
        )

    return value.strip()


def _encode_string_list(
    value: Any,
    *,
    field_name: str,
    context: str,
) -> str:
    if (
        not isinstance(
            value,
            list,
        )
        or any(
            not isinstance(
                item,
                str,
            )
            or not item.strip()
            for item in value
        )
    ):
        raise PassageStoreBuildError(
            f"'{field_name}' must be a "
            f"string array in {context}."
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _require_preserved_text(
    mapping: dict[str, Any],
    field_name: str,
    *,
    context: str,
) -> str:
    """Validate non-empty text while preserving exact content."""

    value = mapping.get(
        field_name
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise PassageStoreBuildError(
            f"'{field_name}' must be a "
            f"non-empty string in {context}."
        )

    return value


def _require_sha256(
    mapping: dict[str, Any],
    field_name: str,
    *,
    context: str,
) -> str:
    value = _require_text(
        mapping,
        field_name,
        context=context,
    ).lower()

    if (
        len(value) != 64
        or any(
            character
            not in "0123456789abcdef"
            for character in value
        )
    ):
        raise PassageStoreBuildError(
            f"'{field_name}' must be a "
            f"SHA-256 hexadecimal value in {context}."
        )

    return value


def _manifest_summary_int(
    source_manifest: dict[str, Any],
    field_name: str,
) -> int:
    summary = _require_mapping(
        source_manifest.get(
            "summary"
        ),
        context=(
            "passage corpus manifest summary"
        ),
    )

    return _require_non_negative_int(
        summary,
        field_name,
        context=(
            "passage corpus manifest summary"
        ),
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


__all__ = [
    "PassageStoreBuildError",
    "build_passage_store",
]
