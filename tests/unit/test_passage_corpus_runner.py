"""Tests for parser-JSON passage corpus execution."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

import najm_retrieval.passage_corpus_runner as runner
from najm_retrieval.parsing.models import (
    BlockType,
)
from najm_retrieval.passages import (
    Passage,
    PassageBuildResult,
    PassageConfig,
    PassageKind,
    PassageMember,
)


def _block_type(
    value: str,
) -> BlockType:
    return next(
        item
        for item in BlockType
        if item.value == value
    )


PARAGRAPH = _block_type(
    "paragraph"
)


def _write_payload(
    directory: Path,
    *,
    version_id: str,
    author_id: str = "author",
    work_id: str = "author.work",
    profile: str = "mixed_prose_ocr",
    include_in_index: bool = True,
    is_canonical: bool = True,
    filename: str | None = None,
) -> Path:
    """Write one minimal parser-version payload."""

    payload = {
        "schema_version": 1,
        "version": {
            "author_id": author_id,
            "work_id": work_id,
            "version_id": version_id,
            "profile": profile,
            "include_in_index": (
                include_in_index
            ),
            "is_canonical": (
                is_canonical
            ),
        },
        "document": {
            "version_id": version_id,
            "profile": profile,
            "source_path": (
                f"raw/{version_id}"
            ),
            "blocks": [],
        },
        "source": {
            "path": f"raw/{version_id}",
        },
        "summary": {},
        "metrics": {},
        "parser": {
            "name": "test",
            "version": "1",
        },
    }

    path = directory / (
        filename
        if filename is not None
        else f"{version_id}.json"
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def _make_build_result(
    *,
    version_id: str,
    profile: str,
    include_in_index: bool,
    config: PassageConfig,
) -> PassageBuildResult:
    """Create a deterministic mocked passage result."""

    if not include_in_index:
        return PassageBuildResult(
            config=config,
            passages=(),
            issues=(),
        )

    member = PassageMember(
        unit_id=(
            f"{version_id}:"
            "paragraph_0001"
        ),
        unit_type=PARAGRAPH,
        display_text="متن نمایشی",
        retrieval_text="متن بازیابی",
        search_alias_text="متن بازیابی",
    )

    passage = Passage(
        passage_id=(
            f"{version_id}:"
            "passage_000001"
        ),
        version_id=version_id,
        profile=profile,
        kind=PassageKind.MIXED_PROSE,
        ordinal=1,
        include_in_index=True,
        members=(
            member,
        ),
    )

    return PassageBuildResult(
        config=config,
        passages=(
            passage,
        ),
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch expensive preparation while testing runner logic."""

    monkeypatch.setattr(
        runner,
        "assemble_logical_units",
        lambda **kwargs: (
            kwargs["version_id"],
        ),
    )

    monkeypatch.setattr(
        runner,
        "resolve_logical_unit_contexts",
        lambda **kwargs: kwargs["units"],
    )

    monkeypatch.setattr(
        runner,
        "clean_logical_units",
        lambda units: units,
    )

    monkeypatch.setattr(
        runner,
        "build_passages",
        lambda **kwargs: _make_build_result(
            version_id=kwargs[
                "version_id"
            ],
            profile=kwargs["profile"],
            include_in_index=kwargs[
                "include_in_index"
            ],
            config=kwargs["config"],
        ),
    )


def test_build_passage_corpus_reads_versions_in_stable_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version files are validated and processed deterministically."""

    _patch_pipeline(
        monkeypatch
    )

    _write_payload(
        tmp_path,
        version_id="z.version",
    )

    _write_payload(
        tmp_path,
        version_id="a.version",
    )

    result = runner.build_passage_corpus(
        input_dir=tmp_path
    )

    assert [
        version.version_id
        for version in result.versions
    ] == [
        "a.version",
        "z.version",
    ]

    assert result.passage_count == 2

    assert all(
        version.source_path.parent
        == tmp_path
        for version in result.versions
    )


def test_run_passage_corpus_writes_jsonl_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner connects corpus building to serialization."""

    _patch_pipeline(
        monkeypatch
    )

    input_dir = tmp_path / "parser"
    input_dir.mkdir()

    _write_payload(
        input_dir,
        version_id="book.main",
    )

    _write_payload(
        input_dir,
        version_id="book.reference",
        profile="raw_ocr_reference",
        include_in_index=False,
        is_canonical=False,
    )

    output_dir = tmp_path / "passages"

    result = runner.run_passage_corpus(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    assert result.build_result.passage_count == 1

    assert len(
        result.output_paths.version_paths
    ) == 1

    assert (
        result.output_paths
        .manifest_path
        .exists()
    )

    assert (
        result.output_paths
        .report_path
        .exists()
    )

    jsonl_path = (
        result.output_paths
        .version_paths[0]
    )

    assert jsonl_path.name == (
        "book.main.jsonl"
    )

    record = json.loads(
        jsonl_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    assert record["version_id"] == (
        "book.main"
    )


def test_build_passage_corpus_rejects_filename_mismatch(
    tmp_path: Path,
) -> None:
    """Parser JSON filename must equal its version ID."""

    _write_payload(
        tmp_path,
        version_id="book.actual",
        filename="wrong-name.json",
    )

    with pytest.raises(
        runner.PassageCorpusRunnerError,
        match="filename must match",
    ):
        runner.build_passage_corpus(
            input_dir=tmp_path
        )


def test_build_passage_corpus_rejects_profile_mismatch(
    tmp_path: Path,
) -> None:
    """Version and document profiles must agree."""

    path = _write_payload(
        tmp_path,
        version_id="book.version",
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload["document"]["profile"] = (
        "structured_poetry"
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        runner.PassageCorpusRunnerError,
        match="Profile mismatch",
    ):
        runner.build_passage_corpus(
            input_dir=tmp_path
        )