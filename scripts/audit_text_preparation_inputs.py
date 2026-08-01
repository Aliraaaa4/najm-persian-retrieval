"""Audit parsed corpus before text normalization and passage building."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import unicodedata

from najm_retrieval.text_preparation.attributes import attributes_to_dict


INPUT_DIR = Path(
    "data/processed/parser/versions"
)

OUTPUT_DIR = Path(
    "data/processed/qa"
)

JSON_OUTPUT_PATH = (
    OUTPUT_DIR / "text_inventory.json"
)

MARKDOWN_OUTPUT_PATH = (
    OUTPUT_DIR / "text_inventory.md"
)

EXPECTED_VERSION_COUNT = 7
MAX_EXAMPLES_PER_CATEGORY = 25


CONTENT_BLOCK_TYPES = {
    "heading",
    "section",
    "verse",
    "paragraph",
    "raw",
}


NON_CONTENT_BLOCK_TYPES = {
    "blank",
    "page_marker",
    "image_reference",
    "milestone",
}


LATIN_PATTERN = re.compile(
    r"[A-Za-z]"
)

ASCII_DIGIT_PATTERN = re.compile(
    r"[0-9]"
)

PERSIAN_DIGIT_PATTERN = re.compile(
    r"[\u06F0-\u06F9]"
)

ARABIC_INDIC_DIGIT_PATTERN = re.compile(
    r"[\u0660-\u0669]"
)

ANY_DIGIT_CLASS = (
    r"0-9\u06F0-\u06F9\u0660-\u0669"
)

LEADING_PAREN_NUMBER_PATTERN = re.compile(
    rf"^\s*[\(\[]\s*"
    rf"[{ANY_DIGIT_CLASS}]+"
    rf"\s*[\)\]]"
)

LEADING_NUMBER_PATTERN = re.compile(
    rf"^\s*[{ANY_DIGIT_CLASS}]+"
    rf"\s*[\.\-\u2013\u2014\)]"
)

TRAILING_NUMBER_MARKER_PATTERN = re.compile(
    rf"[\(\[]\s*"
    rf"[{ANY_DIGIT_CLASS}]+"
    rf"\s*[\)\]]\s*$"
)

STANDALONE_NUMBER_PATTERN = re.compile(
    rf"^\s*[{ANY_DIGIT_CLASS}]+\s*$"
)

OPENITI_PATTERNS: dict[str, re.Pattern[str]] = {
    "line_hash": re.compile(
        r"(?m)^\s*#+"
    ),
    "continuation_marker": re.compile(
        r"~~"
    ),
    "hemistich_separator": re.compile(
        r"%~%"
    ),
    "page_marker_token": re.compile(
        r"\bPageV\d"
    ),
    "milestone_token": re.compile(
        r"\bms\d+"
    ),
    "genre_marker": re.compile(
        r"\[genre:[^\]]+\]"
    ),
}


SPECIAL_CODEPOINTS = {
    "\u064A": "arabic_yeh",
    "\u06CC": "persian_yeh",
    "\u0649": "alef_maqsura",
    "\u0643": "arabic_kaf",
    "\u06A9": "persian_kaf",
    "\u0629": "teh_marbuta",
    "\u06C0": "heh_with_yeh_above",
    "\u0640": "tatweel",
    "\u200C": "zwnj",
    "\u200D": "zwj",
    "\u200B": "zero_width_space",
    "\u200E": "left_to_right_mark",
    "\u200F": "right_to_left_mark",
    "\u2060": "word_joiner",
    "\uFEFF": "bom",
    "\uFFFD": "replacement_character",
}


ARABIC_DIACRITIC_RANGES = (
    (0x0610, 0x061A),
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06ED),
)


def is_arabic_diacritic(
    character: str,
) -> bool:
    """Return whether a character is an Arabic combining mark."""

    codepoint = ord(character)

    return any(
        start <= codepoint <= end
        for start, end
        in ARABIC_DIACRITIC_RANGES
    )


def shorten(
    text: str,
    limit: int = 220,
) -> str:
    """Create a one-line readable preview."""

    preview = (
        text
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )

    if len(preview) > limit:
        return preview[:limit] + "..."

    return preview


def add_example(
    examples: dict[str, list[dict[str, Any]]],
    category: str,
    example: dict[str, Any],
) -> None:
    """Add a capped example to a category."""

    if (
        len(examples[category])
        < MAX_EXAMPLES_PER_CATEGORY
    ):
        examples[category].append(
            example
        )


def get_block_text(
    block: dict[str, Any],
) -> str:
    """Return exact block text used for auditing."""

    raw_text = block.get("raw_text")

    if isinstance(raw_text, str):
        return raw_text

    return ""


def get_group_id(
    block: dict[str, Any],
) -> str | None:
    """Read group identifier from block attributes."""

    attributes = attributes_to_dict(
        block.get("attributes")
    )

    value = attributes.get("group_id")

    if value is None:
        return None

    return str(value)


def inspect_document(
    path: Path,
) -> dict[str, Any]:
    """Audit one parsed version JSON."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    document = payload.get("document")

    if not isinstance(document, dict):
        raise TypeError(
            f"Missing document object: {path.name}"
        )

    blocks = document.get("blocks")

    if not isinstance(blocks, list):
        raise TypeError(
            f"Missing document.blocks list: {path.name}"
        )

    version_id = str(
        document.get("version_id")
        or payload.get("version")
        or path.stem
    )

    profile = document.get("profile")

    block_types: Counter[str] = Counter()
    digit_scripts: Counter[str] = Counter()
    openiti_counts: Counter[str] = Counter()
    codepoint_counts: Counter[str] = Counter()
    character_categories: Counter[str] = Counter()

    examples: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    content_block_count = 0
    content_with_latin = 0
    content_with_digits = 0
    content_without_page = 0
    content_without_section_path = 0
    group_ids: Counter[str] = Counter()
    blocks_with_group_id = 0
    diacritic_count = 0

    for index, item in enumerate(blocks):
        if not isinstance(item, dict):
            block_types["<non-dict>"] += 1
            continue

        block_type = str(
            item.get("block_type")
            or "<missing>"
        )

        block_types[block_type] += 1

        text = get_block_text(item)

        if not text:
            continue

        is_content = (
            block_type in CONTENT_BLOCK_TYPES
        )

        if is_content:
            content_block_count += 1

            if not item.get("page"):
                content_without_page += 1

            section_path = item.get(
                "section_path"
            )

            if not section_path:
                content_without_section_path += 1

            group_id = get_group_id(item)

            if group_id is not None:
                blocks_with_group_id += 1
                group_ids[group_id] += 1

        has_latin = bool(
            LATIN_PATTERN.search(text)
        )

        if has_latin:
            if is_content:
                content_with_latin += 1

            add_example(
                examples,
                "latin_text",
                {
                    "block_index": index,
                    "block_id": item.get(
                        "block_id"
                    ),
                    "block_type": block_type,
                    "text": shorten(text),
                },
            )

        ascii_digits = len(
            ASCII_DIGIT_PATTERN.findall(text)
        )

        persian_digits = len(
            PERSIAN_DIGIT_PATTERN.findall(text)
        )

        arabic_digits = len(
            ARABIC_INDIC_DIGIT_PATTERN.findall(
                text
            )
        )

        if ascii_digits:
            digit_scripts[
                "ascii"
            ] += ascii_digits

        if persian_digits:
            digit_scripts[
                "persian"
            ] += persian_digits

        if arabic_digits:
            digit_scripts[
                "arabic_indic"
            ] += arabic_digits

        if (
            ascii_digits
            or persian_digits
            or arabic_digits
        ):
            if is_content:
                content_with_digits += 1

            add_example(
                examples,
                "numeric_text",
                {
                    "block_index": index,
                    "block_id": item.get(
                        "block_id"
                    ),
                    "block_type": block_type,
                    "text": shorten(text),
                },
            )

        numeric_patterns = {
            "leading_parenthesized_number":
                LEADING_PAREN_NUMBER_PATTERN.search(
                    text
                ),
            "leading_number":
                LEADING_NUMBER_PATTERN.search(
                    text
                ),
            "trailing_number_marker":
                TRAILING_NUMBER_MARKER_PATTERN.search(
                    text
                ),
            "standalone_number":
                STANDALONE_NUMBER_PATTERN.fullmatch(
                    text
                ),
        }

        for category, match in (
            numeric_patterns.items()
        ):
            if match:
                add_example(
                    examples,
                    category,
                    {
                        "block_index": index,
                        "block_id": item.get(
                            "block_id"
                        ),
                        "block_type": block_type,
                        "text": shorten(text),
                    },
                )

        for marker_name, pattern in (
            OPENITI_PATTERNS.items()
        ):
            matches = pattern.findall(text)

            if matches:
                openiti_counts[
                    marker_name
                ] += len(matches)

                add_example(
                    examples,
                    f"openiti_{marker_name}",
                    {
                        "block_index": index,
                        "block_id": item.get(
                            "block_id"
                        ),
                        "block_type": block_type,
                        "text": shorten(text),
                    },
                )

        for character in text:
            unicode_name = SPECIAL_CODEPOINTS.get(
                character
            )

            if unicode_name is not None:
                codepoint_counts[
                    unicode_name
                ] += 1

            if is_arabic_diacritic(character):
                diacritic_count += 1

            category = unicodedata.category(
                character
            )

            if category.startswith("C"):
                character_categories[
                    category
                ] += 1

                if character not in {
                    "\r",
                    "\n",
                    "\t",
                }:
                    add_example(
                        examples,
                        "control_or_format_character",
                        {
                            "block_index": index,
                            "block_id": item.get(
                                "block_id"
                            ),
                            "block_type": block_type,
                            "codepoint": (
                                f"U+{ord(character):04X}"
                            ),
                            "unicode_name": (
                                unicodedata.name(
                                    character,
                                    "<unknown>",
                                )
                            ),
                            "text": shorten(text),
                        },
                    )

    repeated_group_ids = sum(
        1
        for count in group_ids.values()
        if count > 1
    )

    max_group_size = max(
        group_ids.values(),
        default=0,
    )

    return {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "version_id": version_id,
        "profile": profile,
        "parser_name": document.get(
            "parser_name"
        ),
        "parser_version": document.get(
            "parser_version"
        ),
        "source_path": document.get(
            "source_path"
        ),
        "body_char_start": document.get(
            "body_char_start"
        ),
        "body_line_start": document.get(
            "body_line_start"
        ),
        "block_count": len(blocks),
        "content_block_count":
            content_block_count,
        "block_types": dict(
            block_types.most_common()
        ),
        "digit_scripts": dict(
            digit_scripts
        ),
        "content_with_digits":
            content_with_digits,
        "content_with_latin":
            content_with_latin,
        "openiti_counts": dict(
            openiti_counts
        ),
        "special_codepoints": dict(
            codepoint_counts
        ),
        "arabic_diacritic_count":
            diacritic_count,
        "character_categories": dict(
            character_categories
        ),
        "content_without_page":
            content_without_page,
        "content_without_section_path":
            content_without_section_path,
        "blocks_with_group_id":
            blocks_with_group_id,
        "unique_group_ids":
            len(group_ids),
        "repeated_group_ids":
            repeated_group_ids,
        "max_group_size":
            max_group_size,
        "examples": dict(examples),
    }


def build_global_summary(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine per-document counts."""

    total_block_count = sum(
        document["block_count"]
        for document in documents
    )

    total_content_blocks = sum(
        document["content_block_count"]
        for document in documents
    )

    block_types: Counter[str] = Counter()
    digit_scripts: Counter[str] = Counter()
    openiti_counts: Counter[str] = Counter()
    special_codepoints: Counter[str] = Counter()

    for document in documents:
        block_types.update(
            document["block_types"]
        )

        digit_scripts.update(
            document["digit_scripts"]
        )

        openiti_counts.update(
            document["openiti_counts"]
        )

        special_codepoints.update(
            document["special_codepoints"]
        )

    return {
        "version_count": len(documents),
        "total_block_count":
            total_block_count,
        "total_content_block_count":
            total_content_blocks,
        "block_types": dict(
            block_types.most_common()
        ),
        "digit_scripts": dict(
            digit_scripts
        ),
        "openiti_counts": dict(
            openiti_counts
        ),
        "special_codepoints": dict(
            special_codepoints
        ),
        "content_with_digits": sum(
            document[
                "content_with_digits"
            ]
            for document in documents
        ),
        "content_with_latin": sum(
            document[
                "content_with_latin"
            ]
            for document in documents
        ),
        "arabic_diacritic_count": sum(
            document[
                "arabic_diacritic_count"
            ]
            for document in documents
        ),
    }


def write_markdown(
    report: dict[str, Any],
) -> None:
    """Write a concise human-readable report."""

    global_summary = report[
        "global_summary"
    ]

    lines = [
        "# Text Preparation Input Audit",
        "",
        "## Global summary",
        "",
        (
            f"- Versions: "
            f"{global_summary['version_count']}"
        ),
        (
            f"- Blocks: "
            f"{global_summary['total_block_count']}"
        ),
        (
            f"- Content blocks: "
            f"{global_summary['total_content_block_count']}"
        ),
        (
            f"- Content blocks with digits: "
            f"{global_summary['content_with_digits']}"
        ),
        (
            f"- Content blocks with Latin text: "
            f"{global_summary['content_with_latin']}"
        ),
        (
            f"- Arabic diacritic characters: "
            f"{global_summary['arabic_diacritic_count']}"
        ),
        "",
        "## Documents",
        "",
        (
            "| File | Profile | Blocks | Content | "
            "Digits | Latin | Missing page | "
            "Missing section path |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for document in report["documents"]:
        lines.append(
            "| "
            f"{document['file_name']} | "
            f"{document['profile']} | "
            f"{document['block_count']} | "
            f"{document['content_block_count']} | "
            f"{document['content_with_digits']} | "
            f"{document['content_with_latin']} | "
            f"{document['content_without_page']} | "
            f"{document['content_without_section_path']} |"
        )

    lines.extend(
        [
            "",
            "## Global block types",
            "",
        ]
    )

    for name, count in (
        global_summary["block_types"].items()
    ):
        lines.append(
            f"- `{name}`: {count}"
        )

    lines.extend(
        [
            "",
            "## Digit scripts",
            "",
        ]
    )

    for name, count in (
        global_summary["digit_scripts"].items()
    ):
        lines.append(
            f"- `{name}`: {count}"
        )

    lines.extend(
        [
            "",
            "## OpenITI syntax",
            "",
        ]
    )

    for name, count in (
        global_summary["openiti_counts"].items()
    ):
        lines.append(
            f"- `{name}`: {count}"
        )

    lines.extend(
        [
            "",
            "## Special Unicode code points",
            "",
        ]
    )

    for name, count in (
        global_summary[
            "special_codepoints"
        ].items()
    ):
        lines.append(
            f"- `{name}`: {count}"
        )

    lines.extend(
        [
            "",
            "## Example inventory",
            "",
        ]
    )

    for document in report["documents"]:
        lines.append(
            f"### {document['file_name']}"
        )
        lines.append("")

        examples = document["examples"]

        if not examples:
            lines.append(
                "No examples collected."
            )
            lines.append("")
            continue

        for category, items in examples.items():
            lines.append(
                f"#### {category}"
            )
            lines.append("")

            for item in items[:5]:
                text = str(
                    item.get("text", "")
                ).replace(
                    "`",
                    "\\`",
                )

                lines.append(
                    f"- `{item.get('block_type')}` "
                    f"`{item.get('block_id')}`: "
                    f"`{text}`"
                )

            lines.append("")

    MARKDOWN_OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run corpus audit."""

    version_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    if len(version_files) != (
        EXPECTED_VERSION_COUNT
    ):
        raise SystemExit(
            "Expected exactly "
            f"{EXPECTED_VERSION_COUNT} JSON files, "
            f"found {len(version_files)}."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents: list[
        dict[str, Any]
    ] = []

    for index, path in enumerate(
        version_files,
        start=1,
    ):
        print(
            f"[{index}/{len(version_files)}] "
            f"Auditing {path.name}"
        )

        documents.append(
            inspect_document(path)
        )

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_directory": str(INPUT_DIR),
        "global_summary":
            build_global_summary(documents),
        "documents": documents,
    }

    JSON_OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_markdown(report)

    print()
    print(
        "JSON_REPORT:",
        JSON_OUTPUT_PATH,
    )
    print(
        "MARKDOWN_REPORT:",
        MARKDOWN_OUTPUT_PATH,
    )
    print(
        "AUDIT_COMPLETE"
    )


if __name__ == "__main__":
    main()