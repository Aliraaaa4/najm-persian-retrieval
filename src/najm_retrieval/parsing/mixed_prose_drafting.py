"""Drafting helpers for mixed-prose OCR golden annotations.

The functions in this module create deterministic initial drafts for
mixed prose/OCR samples. Every generated draft must be reviewed
manually before its annotation status is changed to ``complete``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from najm_retrieval.parsing.goldens import (
    MILESTONE_PATTERN,
    PAGE_PATTERN,
)


IMAGE_REFERENCE_PATTERN = re.compile(
    r"^\s*#?\s*!"
    r"\[(?P<alt_text>[^\]]*)\]"
    r"\((?P<target>[^)]+)\)"
    r"\s*$"
)


MAJLIS_SECTION_PATTERN = re.compile(
    r"^\s*#\s*~{0,2}\s*"
    r"(?P<title>مجلس\s+.+?)\s*$"
)


@dataclass
class _DraftState:
    """Mutable drafting state for one mixed-prose sample."""

    paragraph_count: int = 0
    heading_count: int = 0
    section_count: int = 0
    verse_count: int = 0

    current_paragraph_group: str | None = None
    current_paragraph_is_orphan: bool = False


@dataclass(frozen=True)
class _MarkerMatch:
    """One page or milestone marker found inside a line."""

    kind: str
    start: int
    end: int
    raw_text: str
    attributes: dict[str, Any]


@dataclass
class _DraftSpan:
    """One draft block span relative to a physical line."""

    start: int
    end: int
    block_type: str
    group_id: str | None
    attributes: dict[str, Any]


def draft_mixed_prose_blocks(
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create draft blocks for one mixed-prose OCR sample."""

    profile = sample.get("profile")

    if profile != "mixed_prose_ocr":
        raise ValueError(
            "Mixed-prose drafting requires "
            "profile='mixed_prose_ocr'."
        )

    line_records = sample.get("lines")

    if not isinstance(line_records, list):
        raise ValueError(
            "Sample lines must be an array."
        )

    state = _DraftState()
    draft_blocks: list[dict[str, Any]] = []

    for line_record in line_records:
        if not isinstance(line_record, dict):
            raise ValueError(
                "Every sample line must be an object."
            )

        line_number = line_record.get(
            "line_number"
        )
        line_char_start = line_record.get(
            "char_start"
        )
        line_char_end = line_record.get(
            "char_end"
        )
        line_text = line_record.get("text")

        if not isinstance(line_number, int):
            raise ValueError(
                "Line number must be an integer."
            )

        if not isinstance(line_char_start, int):
            raise ValueError(
                "Line char_start must be an integer."
            )

        if not isinstance(line_char_end, int):
            raise ValueError(
                "Line char_end must be an integer."
            )

        if not isinstance(line_text, str):
            raise ValueError(
                "Line text must be a string."
            )

        if (
            line_char_end - line_char_start
            != len(line_text)
        ):
            raise ValueError(
                f"Line {line_number} character range "
                "does not match its text length."
            )

        spans = _draft_line_spans(
            line_text=line_text,
            state=state,
        )

        for span in spans:
            raw_text = line_text[
                span.start:span.end
            ]

            draft_blocks.append(
                {
                    "block_id": "",
                    "block_type": span.block_type,
                    "char_start": (
                        line_char_start + span.start
                    ),
                    "char_end": (
                        line_char_start + span.end
                    ),
                    "line_start": line_number,
                    "line_end": line_number,
                    "raw_text": raw_text,
                    "group_id": span.group_id,
                    "attributes": deepcopy(
                        span.attributes
                    ),
                }
            )

    for index, block in enumerate(
        draft_blocks,
        start=1,
    ):
        block["block_id"] = (
            f"b{index:04d}"
        )

    return draft_blocks


def apply_mixed_prose_draft(
    sample: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Return a copy of a sample with mixed-prose draft blocks."""

    merged = deepcopy(sample)

    annotations = merged.get(
        "annotations"
    )

    if not isinstance(annotations, dict):
        raise ValueError(
            "Sample annotations must be an object."
        )

    existing_blocks = annotations.get(
        "blocks"
    )

    if not isinstance(existing_blocks, list):
        raise ValueError(
            "Annotation blocks must be an array."
        )

    if existing_blocks and not force:
        raise ValueError(
            "Annotation already contains blocks. "
            "Use force=True only when replacement "
            "is intentional."
        )

    annotations["schema_version"] = (
        annotations.get(
            "schema_version",
            1,
        )
    )

    annotations["status"] = "draft"

    annotations["blocks"] = (
        draft_mixed_prose_blocks(
            merged
        )
    )

    return merged


def _draft_line_spans(
    *,
    line_text: str,
    state: _DraftState,
) -> list[_DraftSpan]:
    """Create all draft spans for one physical line."""

    if line_text.strip(
        " \t\r\n"
    ) == "":
        return [
            _DraftSpan(
                start=0,
                end=len(line_text),
                block_type="blank",
                group_id=None,
                attributes={},
            )
        ]

    image_attributes = (
        _match_image_reference(
            line_text
        )
    )

    if image_attributes is not None:
        return [
            _DraftSpan(
                start=0,
                end=len(line_text),
                block_type="image_reference",
                group_id=None,
                attributes=image_attributes,
            )
        ]

    markers = _collect_markers(
        line_text
    )

    if (
        markers
        and _is_marker_only_line(
            line_text,
            markers,
        )
    ):
        return _draft_marker_only_line(
            line_text=line_text,
            markers=markers,
        )

    (
        block_type,
        group_id,
        attributes,
    ) = _classify_content_line(
        line_text=line_text,
        state=state,
    )

    if not markers:
        return [
            _DraftSpan(
                start=0,
                end=len(line_text),
                block_type=block_type,
                group_id=group_id,
                attributes=attributes,
            )
        ]

    spans: list[_DraftSpan] = []
    cursor = 0

    for marker in markers:
        if cursor < marker.start:
            spans.append(
                _DraftSpan(
                    start=cursor,
                    end=marker.start,
                    block_type=block_type,
                    group_id=group_id,
                    attributes=deepcopy(
                        attributes
                    ),
                )
            )

        spans.append(
            _DraftSpan(
                start=marker.start,
                end=marker.end,
                block_type=marker.kind,
                group_id=None,
                attributes=deepcopy(
                    marker.attributes
                ),
            )
        )

        cursor = marker.end

    if cursor < len(line_text):
        spans.append(
            _DraftSpan(
                start=cursor,
                end=len(line_text),
                block_type=block_type,
                group_id=group_id,
                attributes=deepcopy(
                    attributes
                ),
            )
        )

    return _merge_whitespace_spans(
        spans=spans,
        line_text=line_text,
    )


def _classify_content_line(
    *,
    line_text: str,
    state: _DraftState,
) -> tuple[
    str,
    str | None,
    dict[str, Any],
]:
    """Classify nonblank and non-image prose content."""

    content = line_text.rstrip(
        "\r\n"
    )

    stripped = content.lstrip()

    if stripped.startswith("###"):
        state.heading_count += 1
        state.current_paragraph_group = None
        state.current_paragraph_is_orphan = False

        return (
            "heading",
            f"heading_{state.heading_count:04d}",
            {
                "level": 3,
            },
        )

    section_match = (
        MAJLIS_SECTION_PATTERN.fullmatch(
            content
        )
    )

    if section_match is not None:
        state.section_count += 1
        state.current_paragraph_group = None
        state.current_paragraph_is_orphan = False

        title = section_match.group(
            "title"
        ).strip()

        return (
            "section",
            f"section_{state.section_count:04d}",
            {
                "section_type": "majlis",
                "title": title,
            },
        )

    if (
        stripped.startswith("#")
        and "%~%" in line_text
    ):
        state.verse_count += 1
        state.current_paragraph_group = None
        state.current_paragraph_is_orphan = False

        return (
            "verse",
            f"verse_{state.verse_count:04d}",
            {
                "has_hemistich_separator": True,
                "continuation": False,
                "embedded_in_prose": True,
            },
        )

    if stripped.startswith("~~"):
        if (
            state.current_paragraph_group
            is None
        ):
            state.paragraph_count += 1

            state.current_paragraph_group = (
                f"paragraph_"
                f"{state.paragraph_count:04d}"
            )

            state.current_paragraph_is_orphan = (
                True
            )

        return (
            "paragraph",
            state.current_paragraph_group,
            {
                "continuation": True,
                "orphan_continuation": (
                    state.current_paragraph_is_orphan
                ),
            },
        )

    if stripped.startswith("#"):
        state.paragraph_count += 1

        state.current_paragraph_group = (
            f"paragraph_"
            f"{state.paragraph_count:04d}"
        )

        state.current_paragraph_is_orphan = False

        return (
            "paragraph",
            state.current_paragraph_group,
            {
                "continuation": False,
                "orphan_continuation": False,
            },
        )

    state.current_paragraph_group = None
    state.current_paragraph_is_orphan = False

    return (
        "raw",
        None,
        {},
    )

def _match_image_reference(
    line_text: str,
) -> dict[str, Any] | None:
    """Return attributes when a line is only an image reference."""

    content = line_text.rstrip(
        "\r\n"
    )

    match = IMAGE_REFERENCE_PATTERN.fullmatch(
        content
    )

    if match is None:
        return None

    return {
        "target": match.group("target"),
        "alt_text": match.group(
            "alt_text"
        ),
    }


def _collect_markers(
    line_text: str,
) -> list[_MarkerMatch]:
    """Collect page and milestone markers from one line."""

    markers: list[_MarkerMatch] = []

    for match in PAGE_PATTERN.finditer(
        line_text
    ):
        markers.append(
            _MarkerMatch(
                kind="page_marker",
                start=match.start(),
                end=match.end(),
                raw_text=match.group(0),
                attributes={
                    "volume": int(
                        match.group("volume")
                    ),
                    "page": int(
                        match.group("page")
                    ),
                },
            )
        )

    for match in MILESTONE_PATTERN.finditer(
        line_text
    ):
        markers.append(
            _MarkerMatch(
                kind="milestone",
                start=match.start(),
                end=match.end(),
                raw_text=match.group(0),
                attributes={
                    "number": int(
                        match.group("number")
                    ),
                },
            )
        )

    markers.sort(
        key=lambda marker: (
            marker.start,
            marker.end,
        )
    )

    previous_end = -1

    for marker in markers:
        if marker.start < previous_end:
            raise ValueError(
                "Overlapping page and milestone "
                "markers were detected."
            )

        previous_end = marker.end

    return markers


def _is_marker_only_line(
    line_text: str,
    markers: list[_MarkerMatch],
) -> bool:
    """Return whether a line contains only marker syntax."""

    characters = list(
        line_text.rstrip("\r\n")
    )

    for marker in markers:
        marker_end = min(
            marker.end,
            len(characters),
        )

        for position in range(
            marker.start,
            marker_end,
        ):
            characters[position] = " "

    remainder = "".join(
        characters
    ).strip()

    remainder = (
        remainder
        .replace("#", "")
        .replace("~", "")
        .replace("|", "")
        .strip()
    )

    return remainder == ""


def _draft_marker_only_line(
    *,
    line_text: str,
    markers: list[_MarkerMatch],
) -> list[_DraftSpan]:
    """Draft a physical line containing only markers."""

    spans: list[_DraftSpan] = []

    for index, marker in enumerate(markers):
        start = (
            0
            if index == 0
            else marker.start
        )

        end = (
            markers[index + 1].start
            if index + 1 < len(markers)
            else len(line_text)
        )

        spans.append(
            _DraftSpan(
                start=start,
                end=end,
                block_type=marker.kind,
                group_id=None,
                attributes=deepcopy(
                    marker.attributes
                ),
            )
        )

    return spans


def _merge_whitespace_spans(
    *,
    spans: list[_DraftSpan],
    line_text: str,
) -> list[_DraftSpan]:
    """Attach separator whitespace to an adjacent real block."""

    merged: list[_DraftSpan] = []
    pending_start: int | None = None

    for span in spans:
        span_text = line_text[
            span.start:span.end
        ]

        is_whitespace_content = (
            span.block_type
            not in {
                "page_marker",
                "milestone",
            }
            and span_text.strip(
                " \t\r\n"
            )
            == ""
        )

        if is_whitespace_content:
            if merged:
                merged[-1].end = span.end
            elif pending_start is None:
                pending_start = span.start

            continue

        if pending_start is not None:
            span.start = pending_start
            pending_start = None

        merged.append(span)

    if pending_start is not None:
        return [
            _DraftSpan(
                start=0,
                end=len(line_text),
                block_type="blank",
                group_id=None,
                attributes={},
            )
        ]

    return merged