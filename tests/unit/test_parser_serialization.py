"""Tests for parser JSON serialization and output files."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path

from najm_retrieval.parsing.corpus_runner import (
    CorpusParseResult,
    ParsedCorpusVersion,
)
from najm_retrieval.parsing.core import (
    split_openiti_source,
)
from najm_retrieval.parsing.dispatcher import (
    parse_source,
)
from najm_retrieval.parsing.models import (
    BlockType,
    SourceSpan,
)
from najm_retrieval.parsing.profiles import (
    STRUCTURED_POETRY,
)
from najm_retrieval.parsing.serialization import (
    to_json_compatible,
    write_corpus_outputs,
)


VERSION_ID = (
    "0001Author.Poetry.Test-per1"
)


class ExampleEnum(str, Enum):
    """Small enum used to test conversion."""

    VALUE = "example"


def make_corpus_result(
    tmp_path: Path,
) -> CorpusParseResult:
    """Create one fully parsed synthetic corpus result."""

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Serialization Test\r\n"
        "#META#Header#End#\r\n"
        "~~PageV01P001\r\n"
        "# بیت نخست %~% بیت دوم\r\n"
    )

    source_path = (
        tmp_path / VERSION_ID
    )

    source = split_openiti_source(
        source_text,
        source_path=source_path,
    )

    document, metrics = parse_source(
        source=source,
        profile=STRUCTURED_POETRY,
    )

    version = ParsedCorpusVersion(
        author_id="0001Author",
        work_id="0001Author.Poetry",
        version_id=VERSION_ID,
        profile=STRUCTURED_POETRY,
        include_in_index=True,
        is_canonical=True,
        source_path=source_path,
        source=source,
        document=document,
        metrics=metrics,
    )

    return CorpusParseResult(
        corpus_root=tmp_path,
        versions=(version,),
        runtime_seconds=0.01,
    )


def test_json_conversion_handles_project_types() -> None:
    """Paths, enums, dataclasses, and tuples must convert."""

    payload = to_json_compatible(
        {
            "path": Path("data/example"),
            "enum": ExampleEnum.VALUE,
            "block_type": BlockType.VERSE,
            "span": SourceSpan(
                line_start=10,
                line_end=10,
                char_start=100,
                char_end=110,
            ),
            "values": (
                "one",
                "two",
            ),
        }
    )

    assert payload == {
        "path": str(
            Path("data/example")
        ),
        "enum": "example",
        "block_type": "verse",
        "span": {
            "line_start": 10,
            "line_end": 10,
            "char_start": 100,
            "char_end": 110,
        },
        "values": [
            "one",
            "two",
        ],
    }

    json.dumps(
        payload,
        ensure_ascii=False,
    )


def test_writer_creates_report_and_version_json(
    tmp_path: Path,
) -> None:
    """A parse result must produce valid UTF-8 JSON files."""

    result = make_corpus_result(
        tmp_path
    )

    output_paths = write_corpus_outputs(
        result,
        output_dir=(
            tmp_path / "output"
        ),
    )

    assert (
        output_paths.report_path.is_file()
    )

    assert len(
        output_paths.version_paths
    ) == 1

    version_path = (
        output_paths.version_paths[0]
    )

    assert version_path.is_file()

    report = json.loads(
        output_paths.report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report["schema_version"] == 1

    assert (
        report["summary"]["version_count"]
        == 1
    )

    assert (
        report["summary"]["all_lossless"]
        is True
    )

    assert (
        report["summary"][
            "indexable_version_count"
        ]
        == 1
    )

    version_payload = json.loads(
        version_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        version_payload["version"][
            "version_id"
        ]
        == VERSION_ID
    )

    assert (
        version_payload["summary"][
            "passes_lossless_gate"
        ]
        is True
    )

    block_types = {
        block["block_type"]
        for block in version_payload[
            "document"
        ]["blocks"]
    }

    assert "page_marker" in block_types
    assert "verse" in block_types

    raw_text = "".join(
        block["raw_text"]
        for block in version_payload[
            "document"
        ]["blocks"]
    )

    assert raw_text == (
        result.versions[0].source.body_text
    )


def test_writer_can_replace_existing_outputs(
    tmp_path: Path,
) -> None:
    """Repeated runs must deterministically replace outputs."""

    result = make_corpus_result(
        tmp_path
    )

    output_dir = (
        tmp_path / "output"
    )

    first = write_corpus_outputs(
        result,
        output_dir=output_dir,
    )

    first_report = (
        first.report_path.read_text(
            encoding="utf-8"
        )
    )

    second = write_corpus_outputs(
        result,
        output_dir=output_dir,
    )

    second_report = (
        second.report_path.read_text(
            encoding="utf-8"
        )
    )

    assert first_report == second_report
