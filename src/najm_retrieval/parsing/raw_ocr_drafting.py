"""Conservative golden drafting for raw OCR reference texts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from najm_retrieval.parsing.goldens import (
    MILESTONE_PATTERN,
    PAGE_PATTERN,
)


@dataclass(frozen=True)
class _MarkerSpan:
    """One marker found inside a physical source line."""

    start: int
    end: int
    block_type: str
    attributes: dict[str, Any]


def draft_raw_ocr_blocks(
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """Draft lossless blocks for a raw OCR reference sample."""

    profile = sample.get("profile")

    if profile != "raw_ocr_reference":
        raise ValueError(
            "Raw OCR drafting requires profile "
            "'raw_ocr_reference', "
            f"found {profile!r}."
        )

    lines = sample.get("lines")

    if not isinstance(lines, list) or not lines:
        raise ValueError(
            "Sample must contain a non-empty lines list."
        )

    blocks: list[dict[str, Any]] = []
    raw_group_count = 0

    for line in lines:
        if not isinstance(line, dict):
            raise ValueError(
                "Each sample line must be an object."
            )

        line_number = line.get("line_number")
        char_start = line.get("char_start")
        char_end = line.get("char_end")
        line_text = line.get("text")

        if not isinstance(line_number, int):
            raise ValueError(
                "Line number must be an integer."
            )

        if not isinstance(char_start, int):
            raise ValueError(
                "Line char_start must be an integer."
            )

        if not isinstance(char_end, int):
            raise ValueError(
                "Line char_end must be an integer."
            )

        if not isinstance(line_text, str):
            raise ValueError(
                "Line text must be a string."
            )

        if char_end - char_start != len(line_text):
            raise ValueError(
                f"Character span mismatch on line "
                f"{line_number}."
            )

        if line_text.strip() == "":
            _append_block(
                blocks=blocks,
                block_type="blank",
                char_start=char_start,
                char_end=char_end,
                line_number=line_number,
                raw_text=line_text,
                group_id=None,
                attributes={},
            )
            continue

        markers = _collect_markers(line_text)

        if (
            len(markers) == 1
            and _is_marker_only_line(
                line_text=line_text,
                marker=markers[0],
            )
        ):
            marker = markers[0]

            _append_block(
                blocks=blocks,
                block_type=marker.block_type,
                char_start=char_start,
                char_end=char_end,
                line_number=line_number,
                raw_text=line_text,
                group_id=None,
                attributes=marker.attributes,
            )
            continue

        raw_group_count += 1
        group_id = (
            f"raw_ocr_{raw_group_count:04d}"
        )

        if not markers:
            _append_block(
                blocks=blocks,
                block_type="raw",
                char_start=char_start,
                char_end=char_end,
                line_number=line_number,
                raw_text=line_text,
                group_id=group_id,
                attributes={},
            )
            continue

        cursor = 0

        for marker in markers:
            if marker.start > cursor:
                _append_block(
                    blocks=blocks,
                    block_type="raw",
                    char_start=(
                        char_start + cursor
                    ),
                    char_end=(
                        char_start + marker.start
                    ),
                    line_number=line_number,
                    raw_text=line_text[
                        cursor:marker.start
                    ],
                    group_id=group_id,
                    attributes={},
                )

            _append_block(
                blocks=blocks,
                block_type=marker.block_type,
                char_start=(
                    char_start + marker.start
                ),
                char_end=(
                    char_start + marker.end
                ),
                line_number=line_number,
                raw_text=line_text[
                    marker.start:marker.end
                ],
                group_id=None,
                attributes=marker.attributes,
            )

            cursor = marker.end

        if cursor < len(line_text):
            _append_block(
                blocks=blocks,
                block_type="raw",
                char_start=char_start + cursor,
                char_end=char_end,
                line_number=line_number,
                raw_text=line_text[cursor:],
                group_id=group_id,
                attributes={},
            )

    reconstructed = "".join(
        block["raw_text"]
        for block in blocks
    )

    source_text = "".join(
        line["text"]
        for line in lines
    )

    if reconstructed != source_text:
        raise ValueError(
            "Raw OCR drafting failed exact "
            "reconstruction."
        )

    return blocks


def apply_raw_ocr_draft(
    sample: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Return a copy containing raw OCR draft blocks."""

    updated = deepcopy(sample)

    annotations = updated.get("annotations")

    if not isinstance(annotations, dict):
        raise ValueError(
            "Sample annotations must be an object."
        )

    if annotations.get("status") == "complete":
        raise ValueError(
            "Refusing to overwrite a complete "
            "golden annotation."
        )

    existing_blocks = annotations.get(
        "blocks"
    )

    if (
        isinstance(existing_blocks, list)
        and existing_blocks
        and not force
    ):
        raise ValueError(
            "Draft already contains blocks. "
            "Use force=True to replace them."
        )

    annotations.setdefault(
        "schema_version",
        1,
    )

    annotations["status"] = "draft"

    annotations["blocks"] = (
        draft_raw_ocr_blocks(updated)
    )

    return updated


def _append_block(
    *,
    blocks: list[dict[str, Any]],
    block_type: str,
    char_start: int,
    char_end: int,
    line_number: int,
    raw_text: str,
    group_id: str | None,
    attributes: dict[str, Any],
) -> None:
    """Append one block using the golden schema."""

    if char_end <= char_start:
        raise ValueError(
            "Draft block must have positive length."
        )

    if len(raw_text) != char_end - char_start:
        raise ValueError(
            "Draft block raw_text length does not "
            "match its character span."
        )

    blocks.append(
        {
            "block_id": (
                f"b{len(blocks) + 1:04d}"
            ),
            "block_type": block_type,
            "char_start": char_start,
            "char_end": char_end,
            "line_start": line_number,
            "line_end": line_number,
            "raw_text": raw_text,
            "group_id": group_id,
            "attributes": attributes,
        }
    )


def _collect_markers(
    line_text: str,
) -> list[_MarkerSpan]:
    """Collect non-overlapping page and milestone markers."""

    markers: list[_MarkerSpan] = []

    for match in PAGE_PATTERN.finditer(
        line_text
    ):
        markers.append(
            _MarkerSpan(
                start=match.start(),
                end=match.end(),
                block_type="page_marker",
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
            _MarkerSpan(
                start=match.start(),
                end=match.end(),
                block_type="milestone",
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
                "Overlapping markers detected "
                "inside one OCR line."
            )

        previous_end = marker.end

    return markers


def _is_marker_only_line(
    *,
    line_text: str,
    marker: _MarkerSpan,
) -> bool:
    """Check whether a line contains only one marker."""

    remainder = (
        line_text[:marker.start]
        + line_text[marker.end:]
    )

    remainder = re.sub(
        r"^\s*~~\s*",
        "",
        remainder,
        count=1,
    )

    return remainder.strip() == ""