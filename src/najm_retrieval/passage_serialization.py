"""Stable JSONL serialization for retrieval passages."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from najm_retrieval.parsing.serialization import (
    to_json_compatible,
)
from najm_retrieval.passage_corpus_models import (
    BuiltPassageVersion,
    PassageCorpusBuildResult,
)
from najm_retrieval.passages import (
    PASSAGE_SCHEMA_VERSION,
    Passage,
)


PASSAGE_CORPUS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PassageOutputPaths:
    """Paths produced by passage corpus serialization."""

    manifest_path: Path
    report_path: Path

    version_paths: tuple[
        Path,
        ...,
    ]


@dataclass(frozen=True)
class _WrittenArtifact:
    """Metadata describing one written JSONL file."""

    relative_path: str
    line_count: int
    byte_count: int
    sha256: str


def build_passage_payload(
    passage: Passage,
) -> dict[str, Any]:
    """Build one complete JSONL passage record."""

    return {
        "schema_version": (
            passage.schema_version
        ),
        "passage_id": passage.passage_id,
        "version_id": passage.version_id,
        "profile": passage.profile,
        "kind": passage.kind.value,
        "ordinal": passage.ordinal,
        "include_in_index": (
            passage.include_in_index
        ),
        "source": {
            "unit_ids": list(
                passage.source_unit_ids
            ),
            "spans": to_json_compatible(
                passage.source_spans
            ),
        },
        "context": {
            "heading_path": list(
                passage.heading_path
            ),
            "section_path": list(
                passage.section_path
            ),
            "parent_context_id": (
                passage.parent_context_id
            ),
            "boundaries": [
                {
                    "unit_id": boundary.unit_id,
                    "unit_type": _type_value(
                        boundary.unit_type
                    ),
                    "display_text": (
                        boundary.display_text
                    ),
                    "metadata": (
                        to_json_compatible(
                            boundary.metadata
                        )
                    ),
                }
                for boundary
                in passage.boundaries
            ],
        },
        "neighbors": {
            "previous_passage_id": (
                passage.previous_passage_id
            ),
            "next_passage_id": (
                passage.next_passage_id
            ),
        },
        "text": {
            "display": passage.display_text,
            "retrieval": (
                passage.retrieval_text
            ),
            "search_alias": (
                passage.search_alias_text
            ),
            "word_count": (
                passage.word_count
            ),
            "unit_count": (
                passage.unit_count
            ),
            "member_count": (
                passage.member_count
            ),
        },
        "members": [
            {
                "unit_id": member.unit_id,
                "unit_type": _type_value(
                    member.unit_type
                ),
                "segment_index": (
                    member.segment_index
                ),
                "segment_count": (
                    member.segment_count
                ),
                "display_text": (
                    member.display_text
                ),
                "retrieval_text": (
                    member.retrieval_text
                ),
                "search_alias_text": (
                    member.search_alias_text
                ),
                "source_spans": (
                    to_json_compatible(
                        member.source_spans
                    )
                ),
                "metadata": (
                    to_json_compatible(
                        member.metadata
                    )
                ),
                "source_issue_codes": list(
                    member.source_issue_codes
                ),
            }
            for member in passage.members
        ],
        "issues": to_json_compatible(
            passage.issues
        ),
    }


def build_manifest_payload(
    result: PassageCorpusBuildResult,
    *,
    artifacts: dict[
        str,
        _WrittenArtifact,
    ],
) -> dict[str, Any]:
    """Build the stable passage corpus manifest."""

    versions = []

    for version in sorted(
        result.versions,
        key=lambda item: item.version_id,
    ):
        artifact = artifacts.get(
            version.version_id
        )

        versions.append(
            {
                "author_id": (
                    version.author_id
                ),
                "work_id": (
                    version.work_id
                ),
                "version_id": (
                    version.version_id
                ),
                "profile": version.profile,
                "include_in_index": (
                    version.include_in_index
                ),
                "is_canonical": (
                    version.is_canonical
                ),
                "source_file": (
                    _source_path(
                        version,
                        input_dir=(
                            result.input_dir
                        ),
                    )
                ),
                "passage_count": (
                    version.passage_count
                ),
                "skipped_unit_count": (
                    version.skipped_unit_count
                ),
                "document_metadata": (
                    to_json_compatible(
                        version.document_metadata
                    )
                    if (
                        version.document_metadata
                        is not None
                    )
                    else None
                ),
                "output": (
                    {
                        "path": (
                            artifact
                            .relative_path
                        ),
                        "line_count": (
                            artifact
                            .line_count
                        ),
                        "byte_count": (
                            artifact
                            .byte_count
                        ),
                        "sha256": (
                            artifact.sha256
                        ),
                    }
                    if artifact is not None
                    else None
                ),
            }
        )

    return {
        "schema_version": (
            PASSAGE_CORPUS_SCHEMA_VERSION
        ),
        "passage_schema_version": (
            PASSAGE_SCHEMA_VERSION
        ),
        "config": to_json_compatible(
            result.config
        ),
        "summary": {
            "version_count": len(
                result.versions
            ),
            "indexable_version_count": len(
                result.indexable_versions
            ),
            "reference_version_count": len(
                result.reference_versions
            ),
            "jsonl_file_count": len(
                artifacts
            ),
            "passage_count": (
                result.passage_count
            ),
            "skipped_unit_count": (
                result.skipped_unit_count
            ),
        },
        "versions": versions,
    }


def build_report_payload(
    result: PassageCorpusBuildResult,
) -> dict[str, Any]:
    """Build a compact passage-build report."""

    kind_counts: Counter[str] = Counter()
    build_issue_counts: Counter[str] = (
        Counter()
    )
    passage_issue_counts: Counter[str] = (
        Counter()
    )
    member_issue_counts: Counter[str] = (
        Counter()
    )

    version_reports = []

    for version in sorted(
        result.versions,
        key=lambda item: item.version_id,
    ):
        version_kind_counts: Counter[str] = (
            Counter()
        )

        for issue in (
            version.build_result.issues
        ):
            build_issue_counts[
                issue.code
            ] += 1

        for passage in (
            version.build_result.passages
        ):
            kind_counts[
                passage.kind.value
            ] += 1

            version_kind_counts[
                passage.kind.value
            ] += 1

            for issue in passage.issues:
                passage_issue_counts[
                    issue.code
                ] += 1

            for member in passage.members:
                for code in (
                    member.source_issue_codes
                ):
                    member_issue_counts[
                        code
                    ] += 1

        version_reports.append(
            {
                "version_id": (
                    version.version_id
                ),
                "profile": version.profile,
                "include_in_index": (
                    version.include_in_index
                ),
                "is_canonical": (
                    version.is_canonical
                ),
                "passage_count": (
                    version.passage_count
                ),
                "skipped_unit_count": (
                    version.skipped_unit_count
                ),
                "kind_counts": dict(
                    sorted(
                        version_kind_counts
                        .items()
                    )
                ),
            }
        )

    return {
        "schema_version": (
            PASSAGE_CORPUS_SCHEMA_VERSION
        ),
        "passage_schema_version": (
            PASSAGE_SCHEMA_VERSION
        ),
        "input_dir": str(
            result.input_dir
        ),
        "runtime_seconds": round(
            result.runtime_seconds,
            6,
        ),
        "config": to_json_compatible(
            result.config
        ),
        "summary": {
            "version_count": len(
                result.versions
            ),
            "indexable_version_count": len(
                result.indexable_versions
            ),
            "reference_version_count": len(
                result.reference_versions
            ),
            "passage_count": (
                result.passage_count
            ),
            "skipped_unit_count": (
                result.skipped_unit_count
            ),
            "kind_counts": dict(
                sorted(
                    kind_counts.items()
                )
            ),
            "build_issue_counts": dict(
                sorted(
                    build_issue_counts.items()
                )
            ),
            "passage_issue_counts": dict(
                sorted(
                    passage_issue_counts
                    .items()
                )
            ),
            "member_source_issue_counts": dict(
                sorted(
                    member_issue_counts.items()
                )
            ),
        },
        "versions": version_reports,
    }


def write_passage_outputs(
    result: PassageCorpusBuildResult,
    *,
    output_dir: str | Path,
) -> PassageOutputPaths:
    """Write JSONL passages, manifest, and report atomically."""

    output_path = Path(
        output_dir
    )

    versions_path = (
        output_path / "versions"
    )

    versions_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts: dict[
        str,
        _WrittenArtifact,
    ] = {}

    written_paths: list[Path] = []
    expected_file_names: set[str] = set()

    for version in sorted(
        result.versions,
        key=lambda item: item.version_id,
    ):
        if not version.include_in_index:
            if version.build_result.passages:
                raise ValueError(
                    "Reference-only versions must "
                    "not contain passages."
                )

            continue

        file_name = (
            f"{version.version_id}.jsonl"
        )

        version_path = (
            versions_path / file_name
        )

        expected_file_names.add(
            file_name
        )

        artifact = _write_jsonl_atomic(
            version.build_result.passages,
            path=version_path,
            relative_path=(
                f"versions/{file_name}"
            ),
        )

        artifacts[
            version.version_id
        ] = artifact

        written_paths.append(
            version_path
        )

    for stale_path in (
        versions_path.glob("*.jsonl")
    ):
        if (
            stale_path.name
            not in expected_file_names
        ):
            stale_path.unlink()

    manifest_path = (
        output_path / "manifest.json"
    )

    report_path = (
        output_path / "build_report.json"
    )

    _write_json_atomic(
        manifest_path,
        build_manifest_payload(
            result,
            artifacts=artifacts,
        ),
    )

    _write_json_atomic(
        report_path,
        build_report_payload(
            result
        ),
    )

    return PassageOutputPaths(
        manifest_path=manifest_path,
        report_path=report_path,
        version_paths=tuple(
            written_paths
        ),
    )


def _write_jsonl_atomic(
    passages: tuple[Passage, ...],
    *,
    path: Path,
    relative_path: str,
) -> _WrittenArtifact:
    """Write stable UTF-8 JSONL without partial files."""

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    digest = sha256()
    byte_count = 0
    line_count = 0

    try:
        with temporary_path.open(
            "wb"
        ) as file:
            for passage in passages:
                payload = build_passage_payload(
                    passage
                )

                serialized = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )

                encoded = (
                    serialized + "\n"
                ).encode("utf-8")

                file.write(encoded)
                digest.update(encoded)

                byte_count += len(encoded)
                line_count += 1

        temporary_path.replace(path)

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise

    return _WrittenArtifact(
        relative_path=relative_path,
        line_count=line_count,
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write stable pretty UTF-8 JSON atomically."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )

    data = (
        serialized + "\n"
    ).encode("utf-8")

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        temporary_path.write_bytes(
            data
        )

        temporary_path.replace(path)

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise


def _source_path(
    version: BuiltPassageVersion,
    *,
    input_dir: Path,
) -> str:
    """Return a portable source-file path."""

    try:
        return (
            version.source_path
            .resolve()
            .relative_to(
                input_dir.resolve()
            )
            .as_posix()
        )
    except ValueError:
        return version.source_path.name


def _type_value(
    value: object,
) -> str:
    """Return a stable Enum or string value."""

    return str(
        getattr(
            value,
            "value",
            value,
        )
    )


__all__ = [
    "PASSAGE_CORPUS_SCHEMA_VERSION",
    "PassageOutputPaths",
    "build_manifest_payload",
    "build_passage_payload",
    "build_report_payload",
    "write_passage_outputs",
]
