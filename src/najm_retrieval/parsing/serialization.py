"""JSON serialization and output writing for parsed corpus results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    fields,
    is_dataclass,
)
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any

from najm_retrieval.parsing.corpus_runner import (
    CorpusParseResult,
    ParsedCorpusVersion,
)


SCHEMA_VERSION = 1

UNSAFE_FILENAME_PATTERN = re.compile(
    r"[^A-Za-z0-9._-]+"
)


@dataclass(frozen=True)
class CorpusOutputPaths:
    """Paths written for one complete corpus parse."""

    report_path: Path
    version_paths: tuple[Path, ...]


def to_json_compatible(
    value: Any,
) -> Any:
    """Recursively convert project objects to JSON values."""

    if isinstance(value, Enum):
        return to_json_compatible(
            value.value
        )

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if is_dataclass(value):
        return {
            field.name: to_json_compatible(
                getattr(value, field.name)
            )
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        return {
            str(key): to_json_compatible(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            to_json_compatible(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            set,
            frozenset,
        ),
    ):
        converted = [
            to_json_compatible(item)
            for item in value
        ]

        return sorted(
            converted,
            key=repr,
        )

    raise TypeError(
        "Object is not JSON serializable: "
        f"{type(value).__name__}"
    )


def build_version_payload(
    version: ParsedCorpusVersion,
) -> dict[str, Any]:
    """Build the complete JSON payload for one version."""

    block_type_counts = Counter(
        block.block_type.value
        for block in version.document.blocks
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "version": {
            "author_id": version.author_id,
            "work_id": version.work_id,
            "version_id": version.version_id,
            "profile": version.profile,
            "include_in_index": (
                version.include_in_index
            ),
            "is_canonical": (
                version.is_canonical
            ),
        },
        "source": {
            "path": str(
                version.source_path
            ),
            "body_char_start": (
                version.source.body_char_start
            ),
            "body_line_start": (
                version.source.body_line_start
            ),
            "body_char_count": len(
                version.source.body_text
            ),
            "body_line_count": len(
                version.source.lines
            ),
        },
        "parser": {
            "name": (
                version.document.parser_name
            ),
            "version": (
                version.document.parser_version
            ),
        },
        "summary": {
            "block_count": len(
                version.document.blocks
            ),
            "block_type_counts": dict(
                sorted(
                    block_type_counts.items()
                )
            ),
            "passes_lossless_gate": (
                version.passes_lossless_gate
            ),
        },
        "document": to_json_compatible(
            version.document
        ),
        "metrics": to_json_compatible(
            version.metrics
        ),
    }


def build_report_payload(
    result: CorpusParseResult,
) -> dict[str, Any]:
    """Build the compact top-level corpus report."""

    versions: list[dict[str, Any]] = []

    for version in result.versions:
        block_type_counts = Counter(
            block.block_type.value
            for block in version.document.blocks
        )

        versions.append(
            {
                "author_id": version.author_id,
                "work_id": version.work_id,
                "version_id": version.version_id,
                "profile": version.profile,
                "source_path": str(
                    version.source_path
                ),
                "include_in_index": (
                    version.include_in_index
                ),
                "is_canonical": (
                    version.is_canonical
                ),
                "parser_name": (
                    version.document.parser_name
                ),
                "parser_version": (
                    version.document.parser_version
                ),
                "block_count": len(
                    version.document.blocks
                ),
                "block_type_counts": dict(
                    sorted(
                        block_type_counts.items()
                    )
                ),
                "passes_lossless_gate": (
                    version.passes_lossless_gate
                ),
                "metrics": to_json_compatible(
                    version.metrics
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_root": str(
            result.corpus_root
        ),
        "summary": {
            "version_count": (
                result.version_count
            ),
            "indexable_version_count": len(
                result.indexable_versions
            ),
            "canonical_version_count": len(
                result.canonical_versions
            ),
            "reference_version_count": len(
                result.reference_versions
            ),
            "all_lossless": (
                result.all_lossless
            ),
            "total_body_chars": (
                result.total_body_chars
            ),
            "total_blocks": (
                result.total_blocks
            ),
            "runtime_seconds": (
                result.runtime_seconds
            ),
        },
        "versions": versions,
    }


def write_corpus_outputs(
    result: CorpusParseResult,
    *,
    output_dir: str | Path,
) -> CorpusOutputPaths:
    """Write one report and one JSON file per version."""

    output_path = Path(
        output_dir
    )

    version_directory = (
        output_path / "versions"
    )

    version_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    version_paths: list[Path] = []

    for version in result.versions:
        filename = (
            _safe_filename(
                version.version_id
            )
            + ".json"
        )

        version_path = (
            version_directory
            / filename
        )

        _write_json_atomic(
            version_path,
            build_version_payload(
                version
            ),
        )

        version_paths.append(
            version_path
        )

    report_path = (
        output_path / "parse_report.json"
    )

    _write_json_atomic(
        report_path,
        build_report_payload(
            result
        ),
    )

    return CorpusOutputPaths(
        report_path=report_path,
        version_paths=tuple(
            version_paths
        ),
    )


def _safe_filename(
    version_id: str,
) -> str:
    """Convert one version ID to a safe filename."""

    safe_value = (
        UNSAFE_FILENAME_PATTERN.sub(
            "_",
            version_id,
        )
        .strip("._")
    )

    if not safe_value:
        raise ValueError(
            "Version ID does not produce "
            "a valid output filename."
        )

    return safe_value


def _write_json_atomic(
    path: Path,
    payload: Any,
) -> None:
    """Write UTF-8 JSON without leaving partial files."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    serialized = json.dumps(
        to_json_compatible(payload),
        ensure_ascii=False,
        indent=2,
    )

    temporary_path.write_text(
        serialized + "\n",
        encoding="utf-8",
        newline="\n",
    )

    temporary_path.replace(
        path
    )
