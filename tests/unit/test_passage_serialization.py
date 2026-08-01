"""Tests for stable passage corpus serialization."""

from __future__ import annotations

from pathlib import Path
import json

from najm_retrieval.parsing.models import (
    BlockType,
)
from najm_retrieval.passage_corpus_models import (
    BuiltPassageVersion,
    PassageCorpusBuildResult,
)
from najm_retrieval.passage_serialization import (
    build_passage_payload,
    write_passage_outputs,
)
from najm_retrieval.passages import (
    Passage,
    PassageBuildResult,
    PassageConfig,
    PassageIssue,
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


VERSE = _block_type("verse")
RAW = _block_type("raw")


def _make_passage(
    version_id: str,
    *,
    ordinal: int = 1,
) -> Passage:
    member = PassageMember(
        unit_id=(
            f"{version_id}:verse_0001"
        ),
        unit_type=VERSE,
        display_text="متن نمایشی",
        retrieval_text="متن بازیابی",
        search_alias_text="متن بازیابی",
        metadata=(
            ("genre", "Gh"),
        ),
    )

    return Passage(
        passage_id=(
            f"{version_id}:"
            f"passage_{ordinal:06d}"
        ),
        version_id=version_id,
        profile="structured_poetry",
        kind=PassageKind.DIWAN,
        ordinal=ordinal,
        include_in_index=True,
        members=(
            member,
        ),
        issues=(
            PassageIssue(
                code="sample_issue",
                message="Sample diagnostic.",
                source_unit_ids=(
                    member.unit_id,
                ),
            ),
        ),
    )


def _make_corpus_result(
    input_dir: Path,
) -> PassageCorpusBuildResult:
    config = PassageConfig()

    indexable_id = (
        "0667BabaAfzal."
        "Diwan.PDL00046-per1"
    )

    reference_id = (
        "0670IbnAscadHanati."
        "Masalik.Kraken220107010708-per1"
    )

    indexable = BuiltPassageVersion(
        author_id="0667BabaAfzal",
        work_id="0667BabaAfzal.Diwan",
        version_id=indexable_id,
        profile="structured_poetry",
        include_in_index=True,
        is_canonical=True,
        source_path=(
            input_dir
            / f"{indexable_id}.json"
        ),
        build_result=PassageBuildResult(
            config=config,
            passages=(
                _make_passage(
                    indexable_id
                ),
            ),
        ),
    )

    reference = BuiltPassageVersion(
        author_id="0670IbnAscadHanati",
        work_id=(
            "0670IbnAscadHanati.Masalik"
        ),
        version_id=reference_id,
        profile="raw_ocr_reference",
        include_in_index=False,
        is_canonical=False,
        source_path=(
            input_dir
            / f"{reference_id}.json"
        ),
        build_result=PassageBuildResult(
            config=config,
            passages=(),
            skipped_unit_ids=(
                f"{reference_id}:raw_0001",
            ),
            issues=(
                PassageIssue(
                    code=(
                        "version_excluded_from_index"
                    ),
                    message=(
                        "Reference-only version."
                    ),
                ),
            ),
        ),
    )

    return PassageCorpusBuildResult(
        input_dir=input_dir,
        config=config,
        versions=(
            indexable,
            reference,
        ),
        runtime_seconds=1.25,
    )


def test_passage_payload_contains_search_and_provenance() -> None:
    """One record keeps text, members, and source identity."""

    passage = _make_passage(
        "book"
    )

    payload = build_passage_payload(
        passage
    )

    assert payload["passage_id"] == (
        "book:passage_000001"
    )

    assert payload["kind"] == "diwan"

    assert payload["text"]["display"] == (
        "متن نمایشی"
    )

    assert payload["text"]["retrieval"] == (
        "متن بازیابی"
    )

    assert payload["source"]["unit_ids"] == [
        "book:verse_0001"
    ]

    assert payload["members"][0][
        "metadata"
    ] == [
        ["genre", "Gh"]
    ]


def test_write_outputs_creates_jsonl_manifest_and_report(
    tmp_path: Path,
) -> None:
    """Serializer writes six-style index files and metadata outputs."""

    input_dir = tmp_path / "parser"
    input_dir.mkdir()

    result = _make_corpus_result(
        input_dir
    )

    output_dir = tmp_path / "passages"

    paths = write_passage_outputs(
        result,
        output_dir=output_dir,
    )

    assert paths.manifest_path.exists()
    assert paths.report_path.exists()

    assert len(paths.version_paths) == 1

    jsonl_text = (
        paths.version_paths[0]
        .read_text(
            encoding="utf-8"
        )
    )

    assert "متن نمایشی" in jsonl_text
    assert "\\u0645" not in jsonl_text

    records = [
        json.loads(line)
        for line in jsonl_text.splitlines()
    ]

    assert len(records) == 1

    manifest = json.loads(
        paths.manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["summary"][
            "version_count"
        ]
        == 2
    )

    assert (
        manifest["summary"][
            "jsonl_file_count"
        ]
        == 1
    )

    versions = {
        item["version_id"]: item
        for item in manifest["versions"]
    }

    indexable = next(
        item
        for item in versions.values()
        if item["include_in_index"]
    )

    reference = next(
        item
        for item in versions.values()
        if not item["include_in_index"]
    )

    assert indexable["output"] is not None
    assert indexable["output"]["line_count"] == 1
    assert len(indexable["output"]["sha256"]) == 64

    assert reference["output"] is None

    report = json.loads(
        paths.report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report["summary"][
        "passage_count"
    ] == 1

    assert report["summary"][
        "kind_counts"
    ] == {
        "diwan": 1
    }


def test_repeated_write_replaces_files_and_removes_stale_jsonl(
    tmp_path: Path,
) -> None:
    """Repeated serialization is deterministic and removes stale files."""

    input_dir = tmp_path / "parser"
    input_dir.mkdir()

    result = _make_corpus_result(
        input_dir
    )

    output_dir = tmp_path / "passages"

    first = write_passage_outputs(
        result,
        output_dir=output_dir,
    )

    first_jsonl = (
        first.version_paths[0]
        .read_bytes()
    )

    first_manifest = (
        first.manifest_path
        .read_bytes()
    )

    stale_path = (
        output_dir
        / "versions"
        / "stale.jsonl"
    )

    stale_path.write_text(
        "stale",
        encoding="utf-8",
    )

    first.version_paths[0].write_text(
        "corrupted",
        encoding="utf-8",
    )

    second = write_passage_outputs(
        result,
        output_dir=output_dir,
    )

    assert not stale_path.exists()

    assert (
        second.version_paths[0]
        .read_bytes()
        == first_jsonl
    )

    assert (
        second.manifest_path
        .read_bytes()
        == first_manifest
    )
