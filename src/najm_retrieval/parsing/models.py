"""Shared data models for deterministic, loss-preserving parsing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class BlockType(str, Enum):
    """Structural block types emitted by every parser candidate."""

    PAGE_MARKER = "page_marker"
    IMAGE_REFERENCE = "image_reference"
    MILESTONE = "milestone"
    HEADING = "heading"
    SECTION = "section"
    VERSE = "verse"
    PARAGRAPH = "paragraph"
    BLANK = "blank"
    RAW = "raw"


class IssueSeverity(str, Enum):
    """Severity assigned to one parser diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class SourceSpan:
    """Location of a block in the original source file.

    Line numbers are one-based and inclusive.
    Character offsets are zero-based and half-open.
    """

    line_start: int
    line_end: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.line_start < 1:
            raise ValueError("line_start must be at least 1.")

        if self.line_end < self.line_start:
            raise ValueError(
                "line_end must be greater than or equal to line_start."
            )

        if self.char_start < 0:
            raise ValueError("char_start must not be negative.")

        if self.char_end < self.char_start:
            raise ValueError(
                "char_end must be greater than or equal to char_start."
            )

    @property
    def line_count(self) -> int:
        """Return the number of source lines covered by the span."""

        return self.line_end - self.line_start + 1

    @property
    def char_count(self) -> int:
        """Return the number of source characters covered by the span."""

        return self.char_end - self.char_start


@dataclass(frozen=True)
class PageReference:
    """A logical OpenITI page reference."""

    raw_marker: str
    volume: int | None
    page: int | None
    source_line: int


@dataclass(frozen=True)
class ImageReference:
    """Reference connecting OCR text to a source page image."""

    raw_marker: str
    image_id: str | None
    image_url: str | None
    source_line: int


@dataclass(frozen=True)
class ParseIssue:
    """A diagnostic produced while parsing a source document."""

    code: str
    message: str
    severity: IssueSeverity
    span: SourceSpan | None = None
    raw_text: str | None = None


@dataclass(frozen=True)
class ParsedBlock:
    """One loss-preserving structural unit from a source document."""

    block_id: str
    block_type: BlockType
    span: SourceSpan

    raw_text: str
    display_text: str
    retrieval_text: str

    page: PageReference | None = None
    image: ImageReference | None = None

    section_path: tuple[str, ...] = ()
    attributes: tuple[tuple[str, Any], ...] = ()
    issues: tuple[ParseIssue, ...] = ()

    def get_attribute(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """Return a block attribute by name."""

        for key, value in self.attributes:
            if key == name:
                return value

        return default

    @property
    def is_content(self) -> bool:
        """Return whether the block may contribute searchable text."""

        return self.block_type in {
            BlockType.HEADING,
            BlockType.SECTION,
            BlockType.VERSE,
            BlockType.PARAGRAPH,
            BlockType.RAW,
        }

    @property
    def is_marker(self) -> bool:
        """Return whether the block is a structural marker."""

        return self.block_type in {
            BlockType.PAGE_MARKER,
            BlockType.IMAGE_REFERENCE,
            BlockType.MILESTONE,
        }


@dataclass(frozen=True)
class ParsedDocument:
    """Complete ordered parser output for one source version."""

    version_id: str
    profile: str
    parser_name: str
    parser_version: str
    source_path: Path

    body_char_start: int
    body_line_start: int

    blocks: tuple[ParsedBlock, ...]
    issues: tuple[ParseIssue, ...]

    @property
    def block_counts(self) -> dict[str, int]:
        """Return counts grouped by block type."""

        counts = Counter(
            block.block_type.value
            for block in self.blocks
        )

        return dict(sorted(counts.items()))

    @property
    def content_blocks(self) -> tuple[ParsedBlock, ...]:
        """Return blocks that may later contribute to passages."""

        return tuple(
            block
            for block in self.blocks
            if block.is_content
        )

    @property
    def marker_blocks(self) -> tuple[ParsedBlock, ...]:
        """Return page, image, and milestone marker blocks."""

        return tuple(
            block
            for block in self.blocks
            if block.is_marker
        )

    @property
    def all_issues(self) -> tuple[ParseIssue, ...]:
        """Return document-level and block-level diagnostics."""

        block_issues = tuple(
            issue
            for block in self.blocks
            for issue in block.issues
        )

        return self.issues + block_issues

    def reconstruct_body(self) -> str:
        """Reconstruct the parsed body from exact raw block text."""

        return "".join(
            block.raw_text
            for block in self.blocks
        )

@dataclass(frozen=True)
class ParseMetrics:
    """Structural and loss-preservation metrics."""

    total_body_lines: int
    covered_lines: int
    uncovered_lines: int

    total_body_chars: int
    covered_chars: int
    uncovered_chars: int
    overlapping_chars: int

    reconstructed_chars: int
    reconstruction_matches_source: bool

    marker_count: int
    raw_line_count: int
    issue_count: int

    runtime_seconds: float
    peak_memory_bytes: int

    @property
    def line_coverage(self) -> float:
        """Return the fraction of body lines touched by blocks."""

        if self.total_body_lines == 0:
            return 1.0

        return self.covered_lines / self.total_body_lines

    @property
    def char_coverage(self) -> float:
        """Return the fraction of source characters covered once."""

        if self.total_body_chars == 0:
            return 1.0

        return self.covered_chars / self.total_body_chars

    @property
    def reconstruction_ratio(self) -> float:
        """Return reconstructed-character length coverage."""

        if self.total_body_chars == 0:
            return 1.0

        return (
            self.reconstructed_chars
            / self.total_body_chars
        )

    @property
    def raw_line_rate(self) -> float:
        """Return the fraction of body lines classified as RAW."""

        if self.total_body_lines == 0:
            return 0.0

        return (
            self.raw_line_count
            / self.total_body_lines
        )

    @property
    def passes_lossless_gate(self) -> bool:
        """Return whether strict lossless checks pass."""

        return (
            self.char_coverage == 1.0
            and self.uncovered_chars == 0
            and self.overlapping_chars == 0
            and self.reconstruction_ratio == 1.0
            and self.reconstruction_matches_source
        )