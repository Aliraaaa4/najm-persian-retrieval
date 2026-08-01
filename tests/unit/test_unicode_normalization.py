"""Tests for multi-view Arabic-script normalization."""

from __future__ import annotations

import pytest

from najm_retrieval.normalization import (
    NormalizationView,
    ScriptProfile,
    classify_script_profile,
    normalize_display,
    normalize_retrieval,
    normalize_search_alias,
    normalize_text,
)


def test_display_applies_nfc() -> None:
    assert normalize_display(
        "ا\u0654"
    ) == "أ"


def test_display_does_not_apply_nfkc() -> None:
    assert normalize_display(
        "ＡＢＣ ①"
    ) == "ＡＢＣ ①"


def test_display_preserves_zwnj_and_diacritics() -> None:
    text = "می\u200cرود قَهْوَة"

    assert normalize_display(text) == text


def test_display_standardizes_heh_hamza() -> None:
    assert normalize_display(
        "خانۀ من"
    ) == "خانهٔ من"


def test_display_cleans_controls_and_line_endings() -> None:
    assert normalize_display(
        "\uFEFF  سلام\t\tدنیا\r\n"
        "خط\u200Bدوم\u0000  "
    ) == "سلام دنیا\nخطدوم"


def test_retrieval_applies_conservative_mappings() -> None:
    text = (
        "ك ي ى می\u200cرود "
        "١٢٣ ۴۵۶ قَهْوَة ـ"
    )

    assert normalize_retrieval(text) == (
        "ک ی ی می رود 123 456 قهوة"
    )


def test_retrieval_preserves_orthographic_distinctions() -> None:
    text = "آ أ إ ٱ ة ؤ ئ ء هٔ"

    assert normalize_retrieval(text) == text


def test_alias_folds_selected_variants() -> None:
    text = "أسماء إله ٱبن آفاق هٔ ة"

    assert normalize_search_alias(text) == (
        "اسماء اله ابن آفاق ه ه"
    )


def test_alias_optionally_folds_alef_madda() -> None:
    assert normalize_search_alias(
        "آفاق",
        fold_alef_madda=True,
    ) == "افاق"


def test_alias_preserves_other_hamza_letters() -> None:
    assert normalize_search_alias(
        "ؤ ئ ء"
    ) == "ؤ ئ ء"


def test_normalize_text_returns_all_views() -> None:
    result = normalize_text(
        "كی می\u200cرود أمة"
    )

    assert result.display_text == (
        "كی می\u200cرود أمة"
    )

    assert result.retrieval_text == (
        "کی می رود أمة"
    )

    assert result.search_alias_text == (
        "کی می رود امه"
    )

    assert result.script_profile == (
        ScriptProfile.MIXED_SIGNALS
    )

    assert result.alias_fold_alef_madda is False
    assert result.changed is True

    assert result.changed_views == (
        NormalizationView.RETRIEVAL,
        NormalizationView.SEARCH_ALIAS,
    )


@pytest.mark.parametrize(
    (
        "text",
        "expected",
    ),
    [
        (
            "پژگ",
            ScriptProfile.PERSIAN_SIGNALS_ONLY,
        ),
        (
            "أمة",
            ScriptProfile.ARABIC_VARIANTS_ONLY,
        ),
        (
            "پأ",
            ScriptProfile.MIXED_SIGNALS,
        ),
        (
            "سلام",
            ScriptProfile.SHARED_OR_OTHER_ONLY,
        ),
    ],
)
def test_classifies_script_profile(
    text: str,
    expected: ScriptProfile,
) -> None:
    assert (
        classify_script_profile(text)
        == expected
    )


def test_display_normalization_is_idempotent() -> None:
    once = normalize_display(
        "خانۀ  من\r\nمی\u200cرود"
    )

    assert normalize_display(once) == once


def test_retrieval_normalization_is_idempotent() -> None:
    once = normalize_retrieval(
        "ك می\u200cرود قَهْوَة"
    )

    assert normalize_retrieval(once) == once


def test_alias_normalization_is_idempotent() -> None:
    once = normalize_search_alias(
        "أمة خانۀ"
    )

    assert normalize_search_alias(once) == once


def test_empty_text_is_supported() -> None:
    result = normalize_text("")

    assert result.display_text == ""
    assert result.retrieval_text == ""
    assert result.search_alias_text == ""
    assert result.changes == ()


@pytest.mark.parametrize(
    "function",
    [
        normalize_display,
        normalize_retrieval,
        normalize_search_alias,
        normalize_text,
        classify_script_profile,
    ],
)
def test_rejects_non_string_input(
    function: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        function(123)  # type: ignore[operator]


def test_records_changes_by_view() -> None:
    result = normalize_text(
        "ك می\u200cرود أمة"
    )

    retrieval_operations = {
        change.operation
        for change in result.changes_for(
            NormalizationView.RETRIEVAL
        )
    }

    alias_operations = {
        change.operation
        for change in result.changes_for(
            NormalizationView.SEARCH_ALIAS
        )
    }

    assert "map_arabic_kaf" in (
        retrieval_operations
    )

    assert "replace_zwnj_with_space" in (
        retrieval_operations
    )

    assert "fold_alef_hamza_variants" in (
        alias_operations
    )

    assert "fold_ta_marbuta" in (
        alias_operations
    )

def test_single_internal_spaces_do_not_create_change_records() -> None:
    """Correct single spaces must not be reported as changes."""

    result = normalize_text(
        "سلام دنیا"
    )

    operations = {
        (
            change.view,
            change.operation,
        )
        for change in result.changes
    }

    assert (
        NormalizationView.DISPLAY,
        "collapse_horizontal_spaces",
    ) not in operations

    assert (
        NormalizationView.RETRIEVAL,
        "collapse_whitespace",
    ) not in operations


def test_counts_only_whitespace_runs_that_actually_change() -> None:
    """Whitespace telemetry counts changed runs, not every space."""

    result = normalize_text(
        "  سلام\t\tدنیا  \nخط دوم"
    )

    display_counts = [
        change.count
        for change in result.changes
        if (
            change.view
            == NormalizationView.DISPLAY
            and change.operation
            == "collapse_horizontal_spaces"
        )
    ]

    retrieval_counts = [
        change.count
        for change in result.changes
        if (
            change.view
            == NormalizationView.RETRIEVAL
            and change.operation
            == "collapse_whitespace"
        )
    ]

    assert result.display_text == (
        "سلام دنیا\nخط دوم"
    )

    assert result.retrieval_text == (
        "سلام دنیا خط دوم"
    )

    # Leading spaces, tabs, and trailing spaces on the first line.
    assert display_counts == [3]

    # The preserved newline becomes one space in retrieval.
    assert retrieval_counts == [1]
