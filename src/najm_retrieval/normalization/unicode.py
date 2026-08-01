"""Conservative multi-view Arabic-script normalization."""

from __future__ import annotations

import re
import unicodedata

from najm_retrieval.normalization.models import (
    NormalizationChange,
    NormalizationResult,
    NormalizationView,
    ScriptProfile,
)


_CONTROL_CHARS_PATTERN = re.compile(
    r"[\u0000-\u0008\u000B-\u000C"
    r"\u000E-\u001F\u007F-\u009F]"
)

_REMOVABLE_FORMAT_PATTERN = re.compile(
    r"[\u00AD\u061C\u200B\u200E\u200F"
    r"\u202A-\u202E\u2060\u2066-\u2069"
    r"\uFEFF]"
)

_HORIZONTAL_SPACE_PATTERN = re.compile(
    r"[ \t]+"
)

_ALL_WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

# Vowel and recitation marks are removed from retrieval views.
# U+0653, U+0654, and U+0655 are deliberately preserved because
# they encode madda or hamza distinctions rather than short vowels.
_ARABIC_HARAKAT_PATTERN = re.compile(
    r"[\u064B-\u0652\u0656-\u065F\u0670]"
)

_QURANIC_MARK_PATTERN = re.compile(
    r"[\u06D6-\u06ED]"
)

_SPACE_TRANSLATION = str.maketrans(
    {
        "\u00A0": " ",
        "\u2007": " ",
        "\u202F": " ",
    }
)

_RETRIEVAL_CHARACTER_TRANSLATION = (
    str.maketrans(
        {
            "\u064A": "\u06CC",
            "\u0649": "\u06CC",
            "\u0643": "\u06A9",
            **{
                chr(0x0660 + index): str(index)
                for index in range(10)
            },
            **{
                chr(0x06F0 + index): str(index)
                for index in range(10)
            },
        }
    )
)

_ALIAS_CHARACTER_TRANSLATION = (
    str.maketrans(
        {
            "\u0623": "\u0627",
            "\u0625": "\u0627",
            "\u0671": "\u0627",
            "\u0629": "\u0647",
        }
    )
)

_PERSIAN_SIGNAL_CHARACTERS = frozenset(
    "پچژگکی"
)

_ARABIC_VARIANT_CHARACTERS = frozenset(
    "يكىةأإٱؤئ"
)

_STANDARD_HEH_HAMZA = "\u0647\u0654"


def normalize_display(
    text: str,
) -> str:
    """Create a conservative user-facing text representation."""

    normalized, _ = (
        _normalize_display_internal(
            text
        )
    )

    return normalized


def normalize_retrieval(
    text: str,
) -> str:
    """Create the primary representation for search and embedding."""

    display_text, _ = (
        _normalize_display_internal(
            text
        )
    )

    retrieval_text, _ = (
        _normalize_retrieval_from_display(
            display_text
        )
    )

    return retrieval_text


def normalize_search_alias(
    text: str,
    *,
    fold_alef_madda: bool = False,
) -> str:
    """Create a looser lexical-search representation.

    Alef with madda is preserved by default. Its optional folding
    is an evaluation parameter rather than a fixed corpus decision.
    """

    if not isinstance(
        fold_alef_madda,
        bool,
    ):
        raise TypeError(
            "fold_alef_madda must be a boolean."
        )

    display_text, _ = (
        _normalize_display_internal(
            text
        )
    )

    retrieval_text, _ = (
        _normalize_retrieval_from_display(
            display_text
        )
    )

    alias_text, _ = (
        _normalize_alias_from_retrieval(
            retrieval_text,
            fold_alef_madda=(
                fold_alef_madda
            ),
        )
    )

    return alias_text


def normalize_text(
    text: str,
    *,
    fold_alef_madda: bool = False,
) -> NormalizationResult:
    """Produce display, retrieval, and search-alias views."""

    if not isinstance(
        fold_alef_madda,
        bool,
    ):
        raise TypeError(
            "fold_alef_madda must be a boolean."
        )

    display_text, display_changes = (
        _normalize_display_internal(
            text
        )
    )

    retrieval_text, retrieval_changes = (
        _normalize_retrieval_from_display(
            display_text
        )
    )

    alias_text, alias_changes = (
        _normalize_alias_from_retrieval(
            retrieval_text,
            fold_alef_madda=(
                fold_alef_madda
            ),
        )
    )

    return NormalizationResult(
        original_text=text,
        display_text=display_text,
        retrieval_text=retrieval_text,
        search_alias_text=alias_text,
        script_profile=(
            classify_script_profile(
                display_text
            )
        ),
        alias_fold_alef_madda=(
            fold_alef_madda
        ),
        changes=(
            display_changes
            + retrieval_changes
            + alias_changes
        ),
    )


def classify_script_profile(
    text: str,
) -> ScriptProfile:
    """Classify codepoint signals without detecting language."""

    _ensure_text(text)

    has_persian_signal = any(
        character
        in _PERSIAN_SIGNAL_CHARACTERS
        for character in text
    )

    has_arabic_variant = any(
        character
        in _ARABIC_VARIANT_CHARACTERS
        for character in text
    )

    if (
        has_persian_signal
        and has_arabic_variant
    ):
        return ScriptProfile.MIXED_SIGNALS

    if has_persian_signal:
        return (
            ScriptProfile.PERSIAN_SIGNALS_ONLY
        )

    if has_arabic_variant:
        return (
            ScriptProfile.ARABIC_VARIANTS_ONLY
        )

    return (
        ScriptProfile.SHARED_OR_OTHER_ONLY
    )


def _normalize_display_internal(
    text: str,
) -> tuple[
    str,
    tuple[NormalizationChange, ...],
]:
    """Normalize a display view and record applied operations."""

    _ensure_text(text)

    changes: list[
        NormalizationChange
    ] = []

    crlf_count = text.count(
        "\r\n"
    )

    without_crlf = text.replace(
        "\r\n",
        "",
    )

    remaining_cr_count = (
        without_crlf.count("\r")
    )

    line_ending_count = (
        crlf_count
        + remaining_cr_count
    )

    normalized = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    _record_change(
        changes,
        view=NormalizationView.DISPLAY,
        operation="normalize_line_endings",
        count=line_ending_count,
    )

    nfc_text = unicodedata.normalize(
        "NFC",
        normalized,
    )

    if nfc_text != normalized:
        _record_change(
            changes,
            view=NormalizationView.DISPLAY,
            operation="unicode_nfc",
            count=1,
        )

    normalized = nfc_text

    heh_with_yeh_above_count = (
        normalized.count("\u06C0")
    )

    normalized = normalized.replace(
        "\u06C0",
        _STANDARD_HEH_HAMZA,
    )

    _record_change(
        changes,
        view=NormalizationView.DISPLAY,
        operation="standardize_heh_hamza",
        count=heh_with_yeh_above_count,
    )

    space_variant_count = sum(
        normalized.count(character)
        for character in (
            "\u00A0",
            "\u2007",
            "\u202F",
        )
    )

    normalized = normalized.translate(
        _SPACE_TRANSLATION
    )

    _record_change(
        changes,
        view=NormalizationView.DISPLAY,
        operation="normalize_space_variants",
        count=space_variant_count,
    )

    normalized, control_count = (
        _CONTROL_CHARS_PATTERN.subn(
            "",
            normalized,
        )
    )

    _record_change(
        changes,
        view=NormalizationView.DISPLAY,
        operation="remove_control_characters",
        count=control_count,
    )

    normalized, format_count = (
        _REMOVABLE_FORMAT_PATTERN.subn(
            "",
            normalized,
        )
    )

    _record_change(
        changes,
        view=NormalizationView.DISPLAY,
        operation="remove_format_characters",
        count=format_count,
    )

    normalized_lines: list[str] = []
    horizontal_space_count = 0

    for line in normalized.split("\n"):
        (
            normalized_line,
            changed_run_count,
        ) = _collapse_horizontal_space_runs(
            line
        )

        horizontal_space_count += (
            changed_run_count
        )

        normalized_lines.append(
            normalized_line
        )

    joined_text = "\n".join(
        normalized_lines
    )

    normalized = joined_text.strip()

    _record_change(
        changes,
        view=NormalizationView.DISPLAY,
        operation="collapse_horizontal_spaces",
        count=horizontal_space_count,
    )

    _record_change(
        changes,
        view=NormalizationView.DISPLAY,
        operation="trim_outer_whitespace",
        count=int(
            normalized != joined_text
        ),
    )

    return (
        normalized,
        tuple(changes),
    )


def _normalize_retrieval_from_display(
    display_text: str,
) -> tuple[
    str,
    tuple[NormalizationChange, ...],
]:
    """Normalize a display view for retrieval."""

    changes: list[
        NormalizationChange
    ] = []

    normalized = display_text

    yeh_variant_count = sum(
        normalized.count(character)
        for character in (
            "\u064A",
            "\u0649",
        )
    )

    arabic_kaf_count = (
        normalized.count("\u0643")
    )

    digit_variant_count = sum(
        normalized.count(
            chr(0x0660 + index)
        )
        + normalized.count(
            chr(0x06F0 + index)
        )
        for index in range(10)
    )

    normalized = normalized.translate(
        _RETRIEVAL_CHARACTER_TRANSLATION
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="map_yeh_variants",
        count=yeh_variant_count,
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="map_arabic_kaf",
        count=arabic_kaf_count,
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="map_digits_to_ascii",
        count=digit_variant_count,
    )

    normalized, harakat_count = (
        _ARABIC_HARAKAT_PATTERN.subn(
            "",
            normalized,
        )
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="remove_arabic_harakat",
        count=harakat_count,
    )

    normalized, quranic_count = (
        _QURANIC_MARK_PATTERN.subn(
            "",
            normalized,
        )
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="remove_quranic_marks",
        count=quranic_count,
    )

    tatweel_count = normalized.count(
        "\u0640"
    )

    normalized = normalized.replace(
        "\u0640",
        "",
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="remove_tatweel",
        count=tatweel_count,
    )

    zwnj_count = normalized.count(
        "\u200C"
    )

    normalized = normalized.replace(
        "\u200C",
        " ",
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="replace_zwnj_with_space",
        count=zwnj_count,
    )

    zwj_count = normalized.count(
        "\u200D"
    )

    normalized = normalized.replace(
        "\u200D",
        "",
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="remove_zwj",
        count=zwj_count,
    )

    (
        normalized,
        whitespace_count,
    ) = _collapse_whitespace_runs(
        normalized
    )

    _record_change(
        changes,
        view=NormalizationView.RETRIEVAL,
        operation="collapse_whitespace",
        count=whitespace_count,
    )

    return (
        normalized,
        tuple(changes),
    )


def _normalize_alias_from_retrieval(
    retrieval_text: str,
    *,
    fold_alef_madda: bool,
) -> tuple[
    str,
    tuple[NormalizationChange, ...],
]:
    """Create a looser lexical alias from retrieval text."""

    changes: list[
        NormalizationChange
    ] = []

    normalized = retrieval_text

    alef_variant_count = sum(
        normalized.count(character)
        for character in (
            "\u0623",
            "\u0625",
            "\u0671",
        )
    )

    ta_marbuta_count = (
        normalized.count("\u0629")
    )

    normalized = normalized.translate(
        _ALIAS_CHARACTER_TRANSLATION
    )

    _record_change(
        changes,
        view=(
            NormalizationView.SEARCH_ALIAS
        ),
        operation="fold_alef_hamza_variants",
        count=alef_variant_count,
    )

    _record_change(
        changes,
        view=(
            NormalizationView.SEARCH_ALIAS
        ),
        operation="fold_ta_marbuta",
        count=ta_marbuta_count,
    )

    heh_hamza_count = normalized.count(
        _STANDARD_HEH_HAMZA
    )

    normalized = normalized.replace(
        _STANDARD_HEH_HAMZA,
        "\u0647",
    )

    _record_change(
        changes,
        view=(
            NormalizationView.SEARCH_ALIAS
        ),
        operation="fold_heh_hamza",
        count=heh_hamza_count,
    )

    if fold_alef_madda:
        alef_madda_count = (
            normalized.count("\u0622")
        )

        normalized = normalized.replace(
            "\u0622",
            "\u0627",
        )

        _record_change(
            changes,
            view=(
                NormalizationView.SEARCH_ALIAS
            ),
            operation="fold_alef_madda",
            count=alef_madda_count,
        )

    normalized = (
        _ALL_WHITESPACE_PATTERN.sub(
            " ",
            normalized,
        )
        .strip()
    )

    return (
        normalized,
        tuple(changes),
    )


def _collapse_horizontal_space_runs(
    line: str,
) -> tuple[str, int]:
    """Collapse only horizontal-space runs that really change.

    A run is counted when it contains a tab, contains more than one
    space, or occurs at the beginning or end of the line.
    """

    changed_run_count = sum(
        1
        for match
        in _HORIZONTAL_SPACE_PATTERN.finditer(
            line
        )
        if (
            match.group(0) != " "
            or match.start() == 0
            or match.end() == len(line)
        )
    )

    normalized = (
        _HORIZONTAL_SPACE_PATTERN.sub(
            " ",
            line,
        )
        .strip()
    )

    return (
        normalized,
        changed_run_count,
    )


def _collapse_whitespace_runs(
    text: str,
) -> tuple[str, int]:
    """Collapse only whitespace runs that really change."""

    changed_run_count = sum(
        1
        for match
        in _ALL_WHITESPACE_PATTERN.finditer(
            text
        )
        if (
            match.group(0) != " "
            or match.start() == 0
            or match.end() == len(text)
        )
    )

    normalized = (
        _ALL_WHITESPACE_PATTERN.sub(
            " ",
            text,
        )
        .strip()
    )

    return (
        normalized,
        changed_run_count,
    )


def _record_change(
    changes: list[NormalizationChange],
    *,
    view: NormalizationView,
    operation: str,
    count: int,
) -> None:
    """Append one change only when it actually occurred."""

    if count <= 0:
        return

    changes.append(
        NormalizationChange(
            view=view,
            operation=operation,
            count=count,
        )
    )


def _ensure_text(
    text: object,
) -> None:
    """Reject non-string inputs explicitly."""

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )
