"""JSONL input and output for retrieval evaluation sets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

from najm_retrieval.evaluation.models import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationSet,
    QueryType,
    RelevanceJudgment,
    RetrievalQuery,
)


class EvaluationDataError(ValueError):
    """Raised when an evaluation JSONL file is invalid."""


def load_evaluation_jsonl(
    path: str | Path,
) -> EvaluationSet:
    """Load and validate a retrieval evaluation JSONL file."""

    input_path = Path(path)

    if not input_path.exists():
        raise EvaluationDataError(
            "Evaluation file does not exist: "
            f"{input_path}"
        )

    if not input_path.is_file():
        raise EvaluationDataError(
            "Evaluation path is not a file: "
            f"{input_path}"
        )

    try:
        text = input_path.read_text(
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError as error:
        raise EvaluationDataError(
            "Evaluation file is not valid UTF-8: "
            f"{input_path}"
        ) from error

    queries: list[RetrievalQuery] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationDataError(
                "Invalid JSON on evaluation line "
                f"{line_number}: {error}"
            ) from error

        try:
            query = _query_from_payload(
                value
            )
        except (
            TypeError,
            ValueError,
            KeyError,
        ) as error:
            raise EvaluationDataError(
                "Invalid evaluation query on line "
                f"{line_number}: {error}"
            ) from error

        queries.append(query)

    try:
        return EvaluationSet(
            queries=tuple(queries)
        )
    except ValueError as error:
        raise EvaluationDataError(
            str(error)
        ) from error


def write_evaluation_jsonl(
    evaluation_set: EvaluationSet,
    *,
    path: str | Path,
) -> Path:
    """Write stable UTF-8 evaluation JSONL atomically."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for query in evaluation_set.queries:
                payload = query_to_payload(
                    query
                )

                serialized = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )

                file.write(
                    serialized + "\n"
                )

        temporary_path.replace(
            output_path
        )

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise

    return output_path


def query_to_payload(
    query: RetrievalQuery,
) -> dict[str, Any]:
    """Convert one query to a stable JSON-compatible record."""

    return {
        "schema_version": (
            query.schema_version
        ),
        "query_id": query.query_id,
        "query_text": query.query_text,
        "query_type": (
            query.query_type.value
        ),
        "judgments": [
            {
                "passage_id": (
                    judgment.passage_id
                ),
                "grade": judgment.grade,
                "rationale": (
                    judgment.rationale
                ),
            }
            for judgment in query.judgments
        ],
        "expected_version_ids": list(
            query.expected_version_ids
        ),
        "tags": list(query.tags),
        "include_in_metrics": (
            query.include_in_metrics
        ),
        "notes": query.notes,
    }


def _query_from_payload(
    value: Any,
) -> RetrievalQuery:
    """Build one query from a decoded JSON object."""

    payload = _require_mapping(
        value,
        label="query",
    )

    schema_version = _require_string(
        payload,
        "schema_version",
    )

    if (
        schema_version
        != EVALUATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported schema_version: "
            f"{schema_version}"
        )

    query_id = _require_string(
        payload,
        "query_id",
    )

    query_text = _require_string(
        payload,
        "query_text",
    )

    query_type_value = _require_string(
        payload,
        "query_type",
    )

    try:
        query_type = QueryType(
            query_type_value
        )
    except ValueError as error:
        raise ValueError(
            "Unsupported query_type: "
            f"{query_type_value}"
        ) from error

    raw_judgments = payload.get(
        "judgments",
        [],
    )

    if not isinstance(
        raw_judgments,
        list,
    ):
        raise ValueError(
            "judgments must be a list."
        )

    judgments = tuple(
        _judgment_from_payload(
            item
        )
        for item in raw_judgments
    )

    expected_version_ids = (
        _require_string_tuple(
            payload.get(
                "expected_version_ids",
                [],
            ),
            label="expected_version_ids",
        )
    )

    tags = _require_string_tuple(
        payload.get(
            "tags",
            [],
        ),
        label="tags",
    )

    include_in_metrics = payload.get(
        "include_in_metrics",
        True,
    )

    if not isinstance(
        include_in_metrics,
        bool,
    ):
        raise ValueError(
            "include_in_metrics must be Boolean."
        )

    notes = payload.get(
        "notes"
    )

    if (
        notes is not None
        and not isinstance(notes, str)
    ):
        raise ValueError(
            "notes must be a string or null."
        )

    return RetrievalQuery(
        query_id=query_id,
        query_text=query_text,
        query_type=query_type,
        judgments=judgments,
        expected_version_ids=(
            expected_version_ids
        ),
        tags=tags,
        include_in_metrics=(
            include_in_metrics
        ),
        notes=notes,
        schema_version=schema_version,
    )


def _judgment_from_payload(
    value: Any,
) -> RelevanceJudgment:
    """Build one relevance judgment."""

    payload = _require_mapping(
        value,
        label="judgment",
    )

    passage_id = _require_string(
        payload,
        "passage_id",
    )

    grade = payload.get(
        "grade",
        1,
    )

    rationale = payload.get(
        "rationale"
    )

    if (
        rationale is not None
        and not isinstance(rationale, str)
    ):
        raise ValueError(
            "judgment rationale must be "
            "a string or null."
        )

    return RelevanceJudgment(
        passage_id=passage_id,
        grade=grade,
        rationale=rationale,
    )


def _require_mapping(
    value: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Require a JSON object."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{label} must be an object."
        )

    return value


def _require_string(
    mapping: Mapping[str, Any],
    field_name: str,
) -> str:
    """Require one nonempty string field."""

    value = mapping.get(
        field_name
    )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a "
            "nonempty string."
        )

    return value


def _require_string_tuple(
    value: Any,
    *,
    label: str,
) -> tuple[str, ...]:
    """Require a list of nonempty strings."""

    if not isinstance(
        value,
        list,
    ):
        raise ValueError(
            f"{label} must be a list."
        )

    items: list[str] = []

    for item in value:
        if (
            not isinstance(item, str)
            or not item.strip()
        ):
            raise ValueError(
                f"{label} must contain only "
                "nonempty strings."
            )

        items.append(item)

    return tuple(items)


__all__ = [
    "EvaluationDataError",
    "load_evaluation_jsonl",
    "query_to_payload",
    "write_evaluation_jsonl",
]