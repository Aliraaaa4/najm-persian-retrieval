"""Core utilities for exact, loss-preserving OpenITI parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OPENITI_MAGIC_LINE = "######OpenITI#"
OPENITI_HEADER_END = "#META#Header#End#"


@dataclass(frozen=True)
class SourceLineRecord:
    """One exact physical line from an OpenITI source body."""

    line_number: int
    char_start: int
    char_end: int
    text: str

    def __post_init__(self) -> None:
        if self.line_number < 1:
            raise ValueError(
                "line_number must be at least 1."
            )

        if self.char_start < 0:
            raise ValueError(
                "char_start must not be negative."
            )

        if self.char_end < self.char_start:
            raise ValueError(
                "char_end must not precede char_start."
            )

        if len(self.text) != (
            self.char_end - self.char_start
        ):
            raise ValueError(
                "Source line text length does not match "
                "its character span."
            )


@dataclass(frozen=True)
class OpenITISource:
    """Exact decoded OpenITI source with separated header and body."""

    source_path: Path
    source_text: str

    header_text: str
    body_text: str

    body_char_start: int
    body_line_start: int

    lines: tuple[SourceLineRecord, ...]

    @property
    def version_id(self) -> str:
        """Return the source version identifier."""

        return self.source_path.name

    def reconstruct_body(self) -> str:
        """Reconstruct the body from physical line records."""

        return "".join(
            line.text
            for line in self.lines
        )


def load_openiti_source(
    path: str | Path,
) -> OpenITISource:
    """Load one OpenITI file without newline normalization."""

    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"OpenITI source file not found: "
            f"{source_path}"
        )

    try:
        with source_path.open(
            mode="r",
            encoding="utf-8-sig",
            errors="strict",
            newline="",
        ) as handle:
            source_text = handle.read()

    except UnicodeDecodeError as error:
        raise ValueError(
            "OpenITI source is not valid UTF-8: "
            f"{source_path}"
        ) from error

    except OSError as error:
        raise OSError(
            f"Could not read OpenITI source "
            f"{source_path}: {error}"
        ) from error

    return split_openiti_source(
        source_text,
        source_path=source_path,
    )


def split_openiti_source(
    source_text: str,
    *,
    source_path: Path,
) -> OpenITISource:
    """Validate and split an exact decoded OpenITI source."""

    if not isinstance(source_text, str):
        raise TypeError(
            "source_text must be a string."
        )

    physical_lines = source_text.splitlines(
        keepends=True
    )

    if not physical_lines:
        raise ValueError(
            "OpenITI source is empty."
        )

    first_line = physical_lines[0].rstrip(
        "\r\n"
    )

    if first_line != OPENITI_MAGIC_LINE:
        raise ValueError(
            "Invalid OpenITI magic line. "
            f"Expected {OPENITI_MAGIC_LINE!r}, "
            f"found {first_line!r}."
        )

    cursor = 0

    for line_number, line_text in enumerate(
        physical_lines,
        start=1,
    ):
        line_end = cursor + len(line_text)

        logical_line = line_text.rstrip(
            "\r\n"
        )

        if logical_line == OPENITI_HEADER_END:
            body_char_start = line_end
            body_line_start = line_number + 1

            header_text = source_text[
                :body_char_start
            ]

            body_text = source_text[
                body_char_start:
            ]

            lines = build_source_line_records(
                body_text,
                body_char_start=body_char_start,
                body_line_start=body_line_start,
            )

            source = OpenITISource(
                source_path=source_path,
                source_text=source_text,
                header_text=header_text,
                body_text=body_text,
                body_char_start=body_char_start,
                body_line_start=body_line_start,
                lines=lines,
            )

            if source.reconstruct_body() != body_text:
                raise ValueError(
                    "Source line construction failed "
                    "exact body reconstruction."
                )

            return source

        cursor = line_end

    raise ValueError(
        "Header end marker "
        f"{OPENITI_HEADER_END!r} was not found "
        f"in {source_path}."
    )


def build_source_line_records(
    body_text: str,
    *,
    body_char_start: int,
    body_line_start: int,
) -> tuple[SourceLineRecord, ...]:
    """Build exact absolute line and character ranges for a body."""

    if not isinstance(body_text, str):
        raise TypeError(
            "body_text must be a string."
        )

    if body_char_start < 0:
        raise ValueError(
            "body_char_start must not be negative."
        )

    if body_line_start < 1:
        raise ValueError(
            "body_line_start must be at least 1."
        )

    records: list[SourceLineRecord] = []
    cursor = body_char_start

    line_texts = body_text.splitlines(
        keepends=True
    )

    for offset, line_text in enumerate(
        line_texts
    ):
        char_start = cursor
        char_end = (
            char_start + len(line_text)
        )

        records.append(
            SourceLineRecord(
                line_number=(
                    body_line_start + offset
                ),
                char_start=char_start,
                char_end=char_end,
                text=line_text,
            )
        )

        cursor = char_end

    expected_end = (
        body_char_start + len(body_text)
    )

    if cursor != expected_end:
        raise ValueError(
            "Source line offsets do not cover "
            "the complete body."
        )

    reconstructed = "".join(
        record.text
        for record in records
    )

    if reconstructed != body_text:
        raise ValueError(
            "Source line records do not reconstruct "
            "the exact body."
        )

    return tuple(records)