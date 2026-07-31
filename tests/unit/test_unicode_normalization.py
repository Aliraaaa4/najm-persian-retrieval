"""Tests for language-neutral Unicode normalization."""

from __future__ import annotations

import pytest

from najm_retrieval.normalization.unicode import (
    normalize_unicode,
)


def test_collapses_horizontal_spaces() -> None:
    """Repeated spaces and tabs collapse within each line."""

    assert normalize_unicode(
        "  خدای   \t تعالی  "
    ) == "خدای تعالی"


def test_normalizes_line_endings() -> None:
    """CRLF and CR become LF without losing line boundaries."""

    assert normalize_unicode(
        "سلام\r\nدنیا\rپایان"
    ) == "سلام\nدنیا\nپایان"


def test_preserves_internal_blank_lines() -> None:
    """Internal blank lines remain available for later processing."""

    assert normalize_unicode(
        "خط اول\n\nخط سوم"
    ) == "خط اول\n\nخط سوم"


def test_removes_unsafe_control_characters() -> None:
    """Unsafe controls are removed without joining lines."""

    assert normalize_unicode(
        "الف\u0000ب\nج"
    ) == "الفب\nج"


def test_preserves_persian_zwnj() -> None:
    """Persian zero-width non-joiner must remain."""

    assert normalize_unicode(
        "می\u200cرود"
    ) == "می\u200cرود"


def test_removes_selected_invisible_formatting() -> None:
    """BOM and zero-width space do not enter retrieval text."""

    assert normalize_unicode(
        "\uFEFFالف\u200Bب"
    ) == "الفب"


def test_applies_nfkc() -> None:
    """Unicode compatibility forms normalize deterministically."""

    assert normalize_unicode(
        "ＡＢＣ"
    ) == "ABC"


def test_rejects_non_string_input() -> None:
    """Invalid input must fail explicitly."""

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        normalize_unicode(123)  # type: ignore[arg-type]