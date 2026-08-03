"""Tests for deterministic scope-aware query suggestions."""

from __future__ import annotations

from pathlib import Path

from najm_retrieval.api.query_suggestions import (
    QuerySuggestionEngine,
)
from najm_retrieval.retrieval import (
    AbstentionReason,
    CorpusScopeCatalog,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def _engine(
) -> QuerySuggestionEngine:
    catalog = (
        CorpusScopeCatalog.from_files(
            manifest_path=(
                PROJECT_ROOT
                / "config"
                / "corpus_manifest.yaml"
            ),
            aliases_path=(
                PROJECT_ROOT
                / "config"
                / "scope_aliases.yaml"
            ),
        )
    )

    return QuerySuggestionEngine(
        catalog
    )


def test_returned_results_have_no_suggestions(
) -> None:
    suggestions = _engine().suggest(
        query_text=(
            "اختیار در مثنوی"
        ),
        reason=(
            AbstentionReason
            .BASELINE_EVIDENCE_PASSED
        ),
        return_results=True,
    )

    assert suggestions == ()


def test_baseline_reason_has_no_suggestions(
) -> None:
    suggestions = _engine().suggest(
        query_text="پرسش آزمایشی",
        reason=(
            AbstentionReason
            .BASELINE_EVIDENCE_PASSED
        ),
        return_results=False,
    )

    assert suggestions == ()


def test_out_of_corpus_author_is_replaced(
) -> None:
    suggestions = _engine().suggest(
        query_text=(
            "حافظ درباره عشق چه می‌گوید؟"
        ),
        reason=(
            AbstentionReason
            .KNOWN_OUT_OF_CORPUS_SCOPE
        ),
        return_results=False,
    )

    assert len(suggestions) == 3

    assert all(
        suggestion.kind
        == "replace_out_of_scope"
        for suggestion in suggestions
    )

    assert all(
        "حافظ"
        not in suggestion.query_text
        for suggestion in suggestions
    )

    assert all(
        suggestion.entity_kind
        == "work"
        for suggestion in suggestions
    )


def test_mixed_scope_prefers_in_corpus_author_works(
) -> None:
    suggestions = _engine().suggest(
        query_text=(
            "مولانا و حافظ درباره "
            "عشق چه گفته‌اند؟"
        ),
        reason=(
            AbstentionReason
            .KNOWN_OUT_OF_CORPUS_SCOPE
        ),
        return_results=False,
    )

    assert len(suggestions) == 3

    assert all(
        suggestion.entity_id.startswith(
            "0672JalalDinRumi."
        )
        for suggestion in suggestions
    )


def test_explicit_work_produces_one_scoped_suggestion(
) -> None:
    suggestions = _engine().suggest(
        query_text=(
            "در مثنوی معنوی درباره "
            "اختیار چه آمده است؟"
        ),
        reason=(
            AbstentionReason
            .SOURCE_ATTRIBUTION_CONFLICT
        ),
        return_results=False,
    )

    assert len(suggestions) == 1

    suggestion = suggestions[0]

    assert suggestion.entity_id == (
        "0672JalalDinRumi.Mathnawi"
    )

    assert (
        "فقط بر اساس متن اصلی "
        "مثنوی معنوی"
        in suggestion.query_text
    )


def test_weak_evidence_offers_three_corpus_works(
) -> None:
    suggestions = _engine().suggest(
        query_text=(
            "معنای اختیار چیست؟"
        ),
        reason=(
            AbstentionReason
            .WEAK_CROSS_RETRIEVER_EVIDENCE
        ),
        return_results=False,
    )

    assert len(suggestions) == 3

    assert len({
        suggestion.entity_id
        for suggestion in suggestions
    }) == 3

def test_mixed_scope_query_text_is_natural(
) -> None:
    suggestions = _engine().suggest(
        query_text=(
            "مولانا و حافظ درباره "
            "عشق چه گفته‌اند؟"
        ),
        reason=(
            AbstentionReason
            .KNOWN_OUT_OF_CORPUS_SCOPE
        ),
        return_results=False,
    )

    assert suggestions[0].query_text == (
        "فقط بر اساس متن اصلی "
        "دیوان شمس: "
        "درباره عشق چه گفته‌اند؟"
    )

    assert all(
        "حافظ"
        not in suggestion.query_text
        for suggestion in suggestions
    )


def test_paratext_marker_is_removed_from_suggestion(
) -> None:
    suggestions = _engine().suggest(
        query_text=(
            "در مقدمه کتاب درباره "
            "اخلاق چه آمده است؟"
        ),
        reason=(
            AbstentionReason
            .TOP_HIT_PARATEXT
        ),
        return_results=False,
    )

    assert len(suggestions) == 3

    assert all(
        "مقدمه"
        not in suggestion.query_text
        for suggestion in suggestions
    )

    assert suggestions[0].query_text == (
        "فقط بر اساس متن اصلی "
        "دیوان بابا افضل: "
        "درباره اخلاق چه آمده است؟"
    )
