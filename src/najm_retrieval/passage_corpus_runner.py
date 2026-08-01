"""Build and serialize passages from parsed corpus JSON files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import time

from najm_retrieval.passage_builder import (
    build_passages,
)
from najm_retrieval.passage_corpus_models import (
    BuiltPassageVersion,
    PassageCorpusBuildResult,
)
from najm_retrieval.passage_serialization import (
    PassageOutputPaths,
    write_passage_outputs,
)
from najm_retrieval.passages import (
    PassageConfig,
)
from najm_retrieval.text_preparation.assembly import (
    assemble_logical_units,
)
from najm_retrieval.text_preparation.context_resolver import (
    resolve_logical_unit_contexts,
)
from najm_retrieval.text_preparation.structural_cleanup import (
    clean_logical_units,
)


class PassageCorpusRunnerError(RuntimeError):
    """Raised when parsed corpus input cannot be processed."""


@dataclass(frozen=True)
class PassageCorpusRunResult:
    """In-memory passage result plus written output paths."""

    build_result: PassageCorpusBuildResult
    output_paths: PassageOutputPaths


@dataclass(frozen=True)
class _ParsedVersionInput:
    """Validated fields read from one parser JSON file."""

    path: Path

    author_id: str
    work_id: str
    version_id: str

    profile: str
    include_in_index: bool
    is_canonical: bool

    blocks: tuple[
        Mapping[str, Any],
        ...,
    ]


def build_passage_corpus(
    *,
    input_dir: str | Path,
    config: PassageConfig | None = None,
) -> PassageCorpusBuildResult:
    """Build passages from all parser version JSON files."""

    input_path = Path(
        input_dir
    )

    if not input_path.exists():
        raise PassageCorpusRunnerError(
            "Parser input directory does not exist: "
            f"{input_path}"
        )

    if not input_path.is_dir():
        raise PassageCorpusRunnerError(
            "Parser input path is not a directory: "
            f"{input_path}"
        )

    version_paths = sorted(
        input_path.glob("*.json")
    )

    if not version_paths:
        raise PassageCorpusRunnerError(
            "No parser version JSON files were found in: "
            f"{input_path}"
        )

    resolved_config = (
        config
        if config is not None
        else PassageConfig()
    )

    started_at = time.perf_counter()

    versions: list[
        BuiltPassageVersion
    ] = []

    seen_version_ids: set[str] = set()

    for path in version_paths:
        parsed_input = _load_version_input(
            path
        )

        if (
            parsed_input.version_id
            in seen_version_ids
        ):
            raise PassageCorpusRunnerError(
                "Duplicate parser version ID: "
                f"{parsed_input.version_id}"
            )

        seen_version_ids.add(
            parsed_input.version_id
        )

        units = assemble_logical_units(
            version_id=(
                parsed_input.version_id
            ),
            blocks=parsed_input.blocks,
        )

        contextual_units = (
            resolve_logical_unit_contexts(
                units=units,
                blocks=parsed_input.blocks,
            )
        )

        cleaned_units = clean_logical_units(
            units
        )

        if not (
            len(units)
            == len(contextual_units)
            == len(cleaned_units)
        ):
            raise PassageCorpusRunnerError(
                "Text-preparation count mismatch for "
                f"{parsed_input.version_id}."
            )

        build_result = build_passages(
            version_id=(
                parsed_input.version_id
            ),
            profile=parsed_input.profile,
            include_in_index=(
                parsed_input.include_in_index
            ),
            contextual_units=(
                contextual_units
            ),
            cleaned_units=cleaned_units,
            config=resolved_config,
        )

        versions.append(
            BuiltPassageVersion(
                author_id=(
                    parsed_input.author_id
                ),
                work_id=(
                    parsed_input.work_id
                ),
                version_id=(
                    parsed_input.version_id
                ),
                profile=(
                    parsed_input.profile
                ),
                include_in_index=(
                    parsed_input
                    .include_in_index
                ),
                is_canonical=(
                    parsed_input.is_canonical
                ),
                source_path=path,
                build_result=build_result,
                document_metadata=None,
            )
        )

    runtime_seconds = (
        time.perf_counter()
        - started_at
    )

    return PassageCorpusBuildResult(
        input_dir=input_path,
        config=resolved_config,
        versions=tuple(versions),
        runtime_seconds=runtime_seconds,
    )


def run_passage_corpus(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    config: PassageConfig | None = None,
) -> PassageCorpusRunResult:
    """Build the corpus and write JSONL outputs."""

    build_result = build_passage_corpus(
        input_dir=input_dir,
        config=config,
    )

    output_paths = write_passage_outputs(
        build_result,
        output_dir=output_dir,
    )

    return PassageCorpusRunResult(
        build_result=build_result,
        output_paths=output_paths,
    )


def _load_version_input(
    path: Path,
) -> _ParsedVersionInput:
    """Load and validate one parser version payload."""

    try:
        text = path.read_text(
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError as error:
        raise PassageCorpusRunnerError(
            "Parser JSON is not valid UTF-8: "
            f"{path}"
        ) from error

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise PassageCorpusRunnerError(
            "Invalid parser JSON file "
            f"{path}: {error}"
        ) from error

    root = _require_mapping(
        payload,
        label="payload",
        path=path,
    )

    version = _require_mapping(
        root.get("version"),
        label="version",
        path=path,
    )

    document = _require_mapping(
        root.get("document"),
        label="document",
        path=path,
    )

    author_id = _require_string(
        version,
        "author_id",
        path=path,
    )

    work_id = _require_string(
        version,
        "work_id",
        path=path,
    )

    version_id = _require_string(
        version,
        "version_id",
        path=path,
    )

    profile = _require_string(
        version,
        "profile",
        path=path,
    )

    include_in_index = _require_bool(
        version,
        "include_in_index",
        path=path,
    )

    is_canonical = _require_bool(
        version,
        "is_canonical",
        path=path,
    )

    document_version_id = (
        _require_string(
            document,
            "version_id",
            path=path,
        )
    )

    document_profile = _require_string(
        document,
        "profile",
        path=path,
    )

    if document_version_id != version_id:
        raise PassageCorpusRunnerError(
            "Version ID mismatch between version "
            f"and document sections in {path}."
        )

    if document_profile != profile:
        raise PassageCorpusRunnerError(
            "Profile mismatch between version "
            f"and document sections in {path}."
        )

    if path.stem != version_id:
        raise PassageCorpusRunnerError(
            "Parser filename must match version_id: "
            f"{path.name} != {version_id}.json"
        )

    raw_blocks = document.get(
        "blocks"
    )

    if not isinstance(
        raw_blocks,
        list,
    ):
        raise PassageCorpusRunnerError(
            "document.blocks must be a list in "
            f"{path}."
        )

    blocks: list[
        Mapping[str, Any]
    ] = []

    for index, block in enumerate(
        raw_blocks
    ):
        if not isinstance(
            block,
            Mapping,
        ):
            raise PassageCorpusRunnerError(
                "Every document block must be an "
                f"object in {path}; invalid index "
                f"{index}."
            )

        blocks.append(block)

    return _ParsedVersionInput(
        path=path,
        author_id=author_id,
        work_id=work_id,
        version_id=version_id,
        profile=profile,
        include_in_index=(
            include_in_index
        ),
        is_canonical=is_canonical,
        blocks=tuple(blocks),
    )


def _require_mapping(
    value: Any,
    *,
    label: str,
    path: Path,
) -> Mapping[str, Any]:
    """Require one JSON object."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise PassageCorpusRunnerError(
            f"{label} must be an object in {path}."
        )

    return value


def _require_string(
    mapping: Mapping[str, Any],
    field_name: str,
    *,
    path: Path,
) -> str:
    """Require one nonempty string field."""

    value = mapping.get(
        field_name
    )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise PassageCorpusRunnerError(
            f"{field_name} must be a nonempty "
            f"string in {path}."
        )

    return value


def _require_bool(
    mapping: Mapping[str, Any],
    field_name: str,
    *,
    path: Path,
) -> bool:
    """Require one Boolean field."""

    value = mapping.get(
        field_name
    )

    if not isinstance(
        value,
        bool,
    ):
        raise PassageCorpusRunnerError(
            f"{field_name} must be Boolean "
            f"in {path}."
        )

    return value


__all__ = [
    "PassageCorpusRunResult",
    "PassageCorpusRunnerError",
    "build_passage_corpus",
    "run_passage_corpus",
]