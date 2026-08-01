"""Remove OpenITI structural syntax from logical-unit text."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import re
from typing import Any

from najm_retrieval.parsing.models import BlockType
from najm_retrieval.text_preparation.models import LogicalUnit


LEADING_HASH_RE = re.compile(
    r"^\s*#{1,3}\s*"
)

LEADING_CONTINUATION_RE = re.compile(
    r"^\s*~~\s*"
)

LEADING_PIPE_RE = re.compile(
    r"^\|{1,2}\s*"
)

LEADING_NUMBER_RE = re.compile(
    r"^(\d+)\s+"
)

HEMISTICH_SEPARATOR_RE = re.compile(
    r"\s*%~%\s*"
)

HEADING_METADATA_RE = re.compile(
    r"\[\s*"
    r"([A-Za-z][A-Za-z0-9_-]*)"
    r"\s*:\s*"
    r"([^\[\]]+?)"
    r"\s*\]"
)

WHITESPACE_RE = re.compile(
    r"\s+"
)


@dataclass(frozen=True)
class CleanupIssue:
    """One non-fatal issue detected during structural cleanup."""

    code: str
    message: str

    def __post_init__(self) -> None:
        """Validate issue fields."""

        if (
            not isinstance(self.code, str)
            or not self.code.strip()
        ):
            raise ValueError(
                "Cleanup issue code must be a "
                "non-empty string."
            )

        if (
            not isinstance(self.message, str)
            or not self.message.strip()
        ):
            raise ValueError(
                "Cleanup issue message must be a "
                "non-empty string."
            )


@dataclass(frozen=True)
class StructurallyCleanedText:
    """Structural-cleanup result for one logical unit."""

    unit_id: str
    raw_text: str
    display_text: str
    retrieval_text: str

    metadata: tuple[
        tuple[str, Any],
        ...,
    ] = field(
        default_factory=tuple
    )

    issues: tuple[
        CleanupIssue,
        ...,
    ] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate cleanup result fields."""

        if (
            not isinstance(self.unit_id, str)
            or not self.unit_id
        ):
            raise ValueError(
                "Cleanup unit_id must be a "
                "non-empty string."
            )

        for value, field_name in (
            (
                self.raw_text,
                "raw_text",
            ),
            (
                self.display_text,
                "display_text",
            ),
            (
                self.retrieval_text,
                "retrieval_text",
            ),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{field_name} must be a string."
                )

        seen_keys: set[str] = set()

        for item in self.metadata:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
            ):
                raise ValueError(
                    "Every cleanup metadata item "
                    "must be a two-item tuple."
                )

            key, _ = item

            if (
                not isinstance(key, str)
                or not key
            ):
                raise ValueError(
                    "Cleanup metadata keys must be "
                    "non-empty strings."
                )

            if key in seen_keys:
                raise ValueError(
                    f"Duplicate cleanup metadata "
                    f"key: {key!r}."
                )

            seen_keys.add(key)

        for issue in self.issues:
            if not isinstance(
                issue,
                CleanupIssue,
            ):
                raise TypeError(
                    "Every cleanup issue must be "
                    "a CleanupIssue."
                )

    @property
    def is_empty(self) -> bool:
        """Return whether both prepared text views are empty."""

        return (
            not self.display_text
            and not self.retrieval_text
        )


def clean_logical_unit_text(
    unit: LogicalUnit,
) -> StructurallyCleanedText:
    """Create display and retrieval text for one logical unit.

    This function removes OpenITI structural syntax but does not
    perform Unicode or language normalization.
    """

    if not isinstance(
        unit,
        LogicalUnit,
    ):
        raise TypeError(
            "unit must be a LogicalUnit."
        )

    unit_attributes = dict(
        unit.attributes
    )

    expected_verse_number = (
        unit_attributes.get(
            "verse_number"
        )
    )

    metadata_values: dict[
        str,
        Any,
    ] = {}

    issues: list[
        CleanupIssue
    ] = []

    fragments: list[str] = []

    verse_prefix_checked = False

    for raw_line in unit.raw_text.splitlines():
        line = raw_line.lstrip()

        line = _strip_repeated_prefix(
            text=line,
            pattern=LEADING_HASH_RE,
        )

        if unit.unit_type in {
            BlockType.HEADING,
            BlockType.SECTION,
        }:
            line = LEADING_PIPE_RE.sub(
                "",
                line,
                count=1,
            )

        line = _strip_repeated_prefix(
            text=line,
            pattern=LEADING_CONTINUATION_RE,
        )

        if (
            unit.unit_type == BlockType.VERSE
            and not verse_prefix_checked
            and isinstance(
                expected_verse_number,
                int,
            )
            and not isinstance(
                expected_verse_number,
                bool,
            )
        ):
            number_match = (
                LEADING_NUMBER_RE.match(
                    line
                )
            )

            if number_match is None:
                issues.append(
                    CleanupIssue(
                        code=(
                            "missing_verse_number_prefix"
                        ),
                        message=(
                            "The logical unit has a "
                            "verse_number attribute but "
                            "no numeric source prefix."
                        ),
                    )
                )

            else:
                detected_number = int(
                    number_match.group(1)
                )

                line = line[
                    number_match.end():
                ]

                metadata_values[
                    "verse_number"
                ] = detected_number

                if (
                    detected_number
                    != expected_verse_number
                ):
                    issues.append(
                        CleanupIssue(
                            code=(
                                "verse_number_mismatch"
                            ),
                            message=(
                                "The source verse number "
                                "does not match the "
                                "verse_number attribute."
                            ),
                        )
                    )

            verse_prefix_checked = True

        if unit.unit_type == BlockType.HEADING:
            line = _extract_heading_metadata(
                text=line,
                metadata_values=metadata_values,
                issues=issues,
            )

        line = _collapse_whitespace(
            line
        )

        if line:
            fragments.append(
                line
            )

    structural_text = _collapse_whitespace(
        " ".join(fragments)
    )

    display_text = (
        HEMISTICH_SEPARATOR_RE.sub(
            " | ",
            structural_text,
        )
    )

    retrieval_text = (
        HEMISTICH_SEPARATOR_RE.sub(
            " ",
            structural_text,
        )
    )

    display_text = _collapse_whitespace(
        display_text
    )

    retrieval_text = _collapse_whitespace(
        retrieval_text
    )

    if (
        not display_text
        and not retrieval_text
        and not metadata_values
    ):
        issues.append(
            CleanupIssue(
                code="structural_only_unit",
                message=(
                    "The logical unit contains only "
                    "structural syntax and has no "
                    "retrievable text or metadata."
                ),
            )
        )

    return StructurallyCleanedText(
        unit_id=unit.unit_id,
        raw_text=unit.raw_text,
        display_text=display_text,
        retrieval_text=retrieval_text,
        metadata=tuple(
            sorted(
                metadata_values.items(),
                key=lambda item: item[0],
            )
        ),
        issues=tuple(issues),
    )


def clean_logical_units(
    units: Iterable[LogicalUnit],
) -> tuple[StructurallyCleanedText, ...]:
    """Clean a sequence of logical units while preserving order."""

    results: list[
        StructurallyCleanedText
    ] = []

    seen_unit_ids: set[str] = set()

    for unit in units:
        if not isinstance(
            unit,
            LogicalUnit,
        ):
            raise TypeError(
                "Every item must be a LogicalUnit."
            )

        if unit.unit_id in seen_unit_ids:
            raise ValueError(
                f"Duplicate logical unit ID: "
                f"{unit.unit_id!r}."
            )

        seen_unit_ids.add(
            unit.unit_id
        )

        results.append(
            clean_logical_unit_text(
                unit
            )
        )

    return tuple(results)


def _strip_repeated_prefix(
    *,
    text: str,
    pattern: re.Pattern[str],
) -> str:
    """Remove repeated structural prefixes from a line.

    Examples include duplicated OpenITI markers such as ``# #``
    and ``~~~~``. Only prefixes at the beginning of the line are
    removed; identical characters inside the text are preserved.
    """

    result = text

    while True:
        updated = pattern.sub(
            "",
            result,
            count=1,
        )

        if updated == result:
            return result

        result = updated


def _extract_heading_metadata(
    *,
    text: str,
    metadata_values: dict[str, Any],
    issues: list[CleanupIssue],
) -> str:
    """Extract explicit key-value metadata from a heading."""

    def replace_match(
        match: re.Match[str],
    ) -> str:
        key = match.group(1).strip()
        value = match.group(2).strip()

        existing_value = (
            metadata_values.get(key)
        )

        if existing_value is None:
            metadata_values[
                key
            ] = value

        elif existing_value != value:
            issues.append(
                CleanupIssue(
                    code="metadata_conflict",
                    message=(
                        f"Heading metadata key "
                        f"{key!r} has conflicting "
                        "values."
                    ),
                )
            )

        return " "

    return HEADING_METADATA_RE.sub(
        replace_match,
        text,
    )


def _collapse_whitespace(
    text: str,
) -> str:
    """Collapse source formatting whitespace to one space."""

    return WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()
