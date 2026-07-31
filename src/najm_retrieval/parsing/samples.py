"""Utilities for reproducible parser golden samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from najm_retrieval.corpus.metadata import HEADER_END_MARKER


ALLOWED_SPLITS = {
    "development",
    "holdout",
}


@dataclass(frozen=True)
class SampleLine:
    """One exact source line included in a parser sample."""

    line_number: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class ParserSample:
    """A reproducible source excerpt for parser evaluation."""

    schema_version: int
    sample_id: str
    split: str

    version_id: str
    profile: str
    source_path: str

    body_line_start: int

    line_start: int
    line_end: int

    char_start: int
    char_end: int

    raw_text: str
    lines: tuple[SampleLine, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split,
            "version_id": self.version_id,
            "profile": self.profile,
            "source_path": self.source_path,
            "body_line_start": self.body_line_start,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "raw_text": self.raw_text,
            "lines": [
                {
                    "line_number": line.line_number,
                    "char_start": line.char_start,
                    "char_end": line.char_end,
                    "text": line.text,
                }
                for line in self.lines
            ],
            "annotations": {
                "blocks": [],
                "notes": "",
            },
        }


def _read_source_text(path: Path) -> str:
    """Read decoded source text while preserving newline characters."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Source text does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return handle.read()


def _find_body_line_start(
    lines: tuple[str, ...],
) -> int:
    """Return the one-based first body line after the OpenITI header."""

    for index, line in enumerate(lines):
        if line.strip() == HEADER_END_MARKER:
            return index + 2

    raise ValueError(
        f"OpenITI header end marker was not found: "
        f"{HEADER_END_MARKER}"
    )


def extract_parser_sample(
    *,
    sample_id: str,
    split: str,
    version_id: str,
    profile: str,
    source_path: Path,
    source_label: str,
    line_start: int,
    line_end: int,
) -> ParserSample:
    """Extract an exact inclusive source-line range."""

    if not sample_id.strip():
        raise ValueError("sample_id must not be empty.")

    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. "
            f"Expected one of: {sorted(ALLOWED_SPLITS)}"
        )

    text = _read_source_text(source_path)

    lines = tuple(
        text.splitlines(keepends=True)
    )

    if not lines:
        raise ValueError(
            f"Source text is empty: {source_path}"
        )

    body_line_start = _find_body_line_start(lines)

    if line_start < body_line_start:
        raise ValueError(
            f"Requested line_start={line_start}, but the body "
            f"starts at line {body_line_start}."
        )

    if line_end < line_start:
        raise ValueError(
            "line_end must be greater than or equal to line_start."
        )

    if line_end > len(lines):
        raise ValueError(
            f"Requested line_end={line_end}, but the source "
            f"contains only {len(lines)} lines."
        )

    line_records: list[SampleLine] = []
    char_cursor = 0

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        char_start = char_cursor
        char_end = char_start + len(line)

        if line_start <= line_number <= line_end:
            line_records.append(
                SampleLine(
                    line_number=line_number,
                    char_start=char_start,
                    char_end=char_end,
                    text=line,
                )
            )

        char_cursor = char_end

    selected_lines = tuple(line_records)

    if not selected_lines:
        raise ValueError(
            "The requested range did not produce any lines."
        )

    raw_text = "".join(
        line.text
        for line in selected_lines
    )

    return ParserSample(
        schema_version=1,
        sample_id=sample_id,
        split=split,
        version_id=version_id,
        profile=profile,
        source_path=source_label,
        body_line_start=body_line_start,
        line_start=line_start,
        line_end=line_end,
        char_start=selected_lines[0].char_start,
        char_end=selected_lines[-1].char_end,
        raw_text=raw_text,
        lines=selected_lines,
    )


def write_parser_sample(
    sample: ParserSample,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write one parser sample as stable UTF-8 JSON."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = json.dumps(
        sample.to_dict(),
        ensure_ascii=False,
        indent=2,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(serialized)
        handle.write("\n")