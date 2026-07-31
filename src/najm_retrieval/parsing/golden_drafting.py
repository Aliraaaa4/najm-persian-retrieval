"""Drafting helpers for structured-poetry golden annotations.

The functions in this module create an initial deterministic draft.
The generated blocks must still be reviewed manually before an
annotation is marked complete.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any

from najm_retrieval.parsing.goldens import (
    MILESTONE_PATTERN,
    PAGE_PATTERN,
)


VERSE_NUMBER_PATTERN = re.compile(
    r"^\s*#\s*(?P<number>\d+)\b"
)

DAFTAR_SECTION_PATTERN = re.compile(
    r"^\s*###\s*\|\s*\[\s*daftar\s+"
    r"(?P<number>\d+)\s*\]\s*$",
    flags=re.IGNORECASE,
)
@dataclass
class _DraftState:
    """Mutable drafting state for one structured-poetry sample."""

    heading_count: int = 0
    section_count: int = 0
    verse_count: int = 0

    current_verse_group: str | None = None

    current_verse_attributes: dict[str, Any] = field(
        default_factory=dict
    )


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


def draft_structured_poetry_blocks(
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create draft blocks for one structured-poetry sample."""

    profile = sample.get("profile")

    if profile != "structured_poetry":
        raise ValueError(
            "Structured-poetry drafting requires "
            "profile='structured_poetry'."
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
        block["block_id"] = f"b{index:04d}"

    return draft_blocks


def apply_structured_poetry_draft(
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy containing draft structured-poetry blocks."""

    updated = deepcopy(sample)

    existing_annotations = updated.get(
        "annotations"
    )

    if not isinstance(
        existing_annotations,
        dict,
    ):
        existing_annotations = {}

    existing_notes = existing_annotations.get(
        "notes",
        "",
    )

    if not isinstance(existing_notes, str):
        existing_notes = ""

    drafting_note = (
        "Automatically drafted using deterministic "
        "structured-poetry rules. Manual review is required."
    )

    if drafting_note not in existing_notes:
        if existing_notes.strip():
            notes = (
                existing_notes.rstrip()
                + "\n"
                + drafting_note
            )
        else:
            notes = drafting_note
    else:
        notes = existing_notes

    updated["annotations"] = {
        "schema_version": 1,
        "status": "draft",
        "blocks": draft_structured_poetry_blocks(
            updated
        ),
        "notes": notes,
    }

    return updated


def _draft_line_spans(
    *,
    line_text: str,
    state: _DraftState,
) -> list[_DraftSpan]:
    """Draft all block spans inside one physical line."""

    if line_text.strip(" \t\r\n") == "":
        state.current_verse_group = None
        state.current_verse_attributes = {}

        return [
            _DraftSpan(
                start=0,
                end=len(line_text),
                block_type="blank",
                group_id=None,
                attributes={},
            )
        ]

    marker_matches = _find_markers(
        line_text
    )

    if (
        marker_matches
        and _is_marker_only_line(
            line_text,
            marker_matches,
        )
    ):
        return _draft_marker_only_line(
            line_text=line_text,
            markers=marker_matches,
        )

    (
        content_type,
        content_group_id,
        content_attributes,
    ) = _classify_content_line(
        line_text=line_text,
        state=state,
    )

    spans: list[_DraftSpan] = []
    cursor = 0

    for marker in marker_matches:
        if cursor < marker.start:
            spans.append(
                _DraftSpan(
                    start=cursor,
                    end=marker.start,
                    block_type=content_type,
                    group_id=content_group_id,
                    attributes=deepcopy(
                        content_attributes
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
                block_type=content_type,
                group_id=content_group_id,
                attributes=deepcopy(
                    content_attributes
                ),
            )
        )

    if not spans:
        spans.append(
            _DraftSpan(
                start=0,
                end=len(line_text),
                block_type=content_type,
                group_id=content_group_id,
                attributes=deepcopy(
                    content_attributes
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
    """Classify non-marker text from one physical line."""

    stripped = line_text.lstrip()
    
    daftar_match = DAFTAR_SECTION_PATTERN.match(
        line_text
    )

    if daftar_match is not None:
        state.section_count += 1
        state.current_verse_group = None
        state.current_verse_attributes = {}

        section_number = int(
            daftar_match.group("number")
        )

        return (
            "section",
            f"section_{state.section_count:04d}",
            {
                "section_type": "daftar",
                "number": section_number,
            },
        )

    if stripped.startswith("###"):
        state.heading_count += 1
        state.current_verse_group = None
        state.current_verse_attributes = {}

        return (
            "heading",
            f"heading_{state.heading_count:04d}",
            {
                "level": 3,
            },
        )

    if (
        stripped.startswith("#")
        and "%~%" in line_text
    ):
        state.verse_count += 1

        group_id = (
            f"verse_{state.verse_count:04d}"
        )

        verse_attributes: dict[str, Any] = {
            "has_hemistich_separator": True,
            "continuation": False,
        }

        verse_number_match = (
            VERSE_NUMBER_PATTERN.match(
                line_text
            )
        )

        if verse_number_match is not None:
            verse_attributes["verse_number"] = int(
                verse_number_match.group(
                    "number"
                )
            )

        state.current_verse_group = group_id

        state.current_verse_attributes = deepcopy(
            verse_attributes
        )

        return (
            "verse",
            group_id,
            verse_attributes,
        )

    if stripped.startswith("~~"):
        if state.current_verse_group is not None:
            continuation_attributes = deepcopy(
                state.current_verse_attributes
            )

            continuation_attributes[
                "continuation"
            ] = True

            return (
                "verse",
                state.current_verse_group,
                continuation_attributes,
            )

        return (
            "raw",
            None,
            {},
        )

    state.current_verse_group = None
    state.current_verse_attributes = {}

    return (
        "raw",
        None,
        {},
    )


def _find_markers(
    line_text: str,
) -> list[_MarkerMatch]:
    """Find ordered non-overlapping markers in a line."""

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
            else:
                if pending_start is None:
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