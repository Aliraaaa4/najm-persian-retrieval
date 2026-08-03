"""Deterministic scope-aware suggestions for abstained queries."""

from __future__ import annotations

from dataclasses import dataclass
import re

from najm_retrieval.retrieval import (
    AbstentionReason,
    CorpusScopeCatalog,
    ScopeCatalogEntity,
    ScopeEntityKind,
    ScopeMention,
)


QUERY_SUGGESTION_SCHEMA_VERSION = (
    "1.0.0"
)


@dataclass(frozen=True)
class QuerySuggestion:
    """One safe clickable query suggestion."""

    query_text: str
    label: str
    kind: str

    entity_id: str
    entity_kind: str
    version_ids: tuple[str, ...]

    schema_version: str = (
        QUERY_SUGGESTION_SCHEMA_VERSION
    )


class QuerySuggestionEngine:
    """Create deterministic suggestions from the frozen scope catalog."""

    def __init__(
        self,
        catalog: CorpusScopeCatalog,
        *,
        max_suggestions: int = 3,
    ) -> None:
        if max_suggestions < 1:
            raise ValueError(
                "max_suggestions must be "
                "at least 1."
            )

        self._catalog = catalog
        self._max_suggestions = (
            max_suggestions
        )

        self._in_corpus_works = tuple(
            entity
            for entity in catalog.entities
            if (
                entity.in_corpus
                and entity.kind
                is ScopeEntityKind.WORK
            )
        )

        if not self._in_corpus_works:
            raise ValueError(
                "The scope catalog contains "
                "no in-corpus works."
            )

    def suggest(
        self,
        *,
        query_text: str,
        reason: AbstentionReason,
        return_results: bool,
    ) -> tuple[QuerySuggestion, ...]:
        """Return safe suggestions without changing the retrieval decision."""

        normalized_query = (
            " ".join(
                query_text.split()
            )
        )

        if (
            not normalized_query
            or return_results
            or reason
            is AbstentionReason
            .BASELINE_EVIDENCE_PASSED
        ):
            return ()

        mentions = (
            self._catalog.match_query(
                normalized_query
            )
        )

        candidates = (
            self._candidate_works(
                mentions
            )
        )

        suggestion_kind = (
            _suggestion_kind(
                reason
            )
        )

        query_body = (
            _query_without_scope_mentions(
                normalized_query,
                mentions=mentions,
            )
        )

        if reason in {
            AbstentionReason.TOP_HIT_PARATEXT,
            AbstentionReason.TOP_HIT_MIXED,
        }:
            query_body = (
                _remove_paratext_markers(
                    query_body
                )
            )

        suggestions: list[
            QuerySuggestion
        ] = []

        seen_queries: set[str] = set()

        for entity in candidates:
            suggested_query = (
                _restrict_to_primary_text(
                    query_body,
                    work_label=(
                        entity.label_fa
                    ),
                )
            )

            if suggested_query in seen_queries:
                continue

            seen_queries.add(
                suggested_query
            )

            suggestions.append(
                QuerySuggestion(
                    query_text=(
                        suggested_query
                    ),
                    label=(
                        "جست‌وجوی همین پرسش در "
                        f"{entity.label_fa}"
                    ),
                    kind=suggestion_kind,
                    entity_id=(
                        entity.entity_id
                    ),
                    entity_kind=(
                        entity.kind.value
                    ),
                    version_ids=(
                        entity.version_ids
                    ),
                )
            )

            if (
                len(suggestions)
                >= self._max_suggestions
            ):
                break

        return tuple(
            suggestions
        )

    def _candidate_works(
        self,
        mentions: tuple[
            ScopeMention,
            ...,
        ],
    ) -> tuple[
        ScopeCatalogEntity,
        ...,
    ]:
        requested_work_ids = {
            mention.entity_id
            for mention in mentions
            if (
                mention.in_corpus
                and mention.kind
                is ScopeEntityKind.WORK
            )
        }

        if requested_work_ids:
            return tuple(
                work
                for work
                in self._in_corpus_works
                if (
                    work.entity_id
                    in requested_work_ids
                )
            )

        requested_author_versions = {
            version_id
            for mention in mentions
            if (
                mention.in_corpus
                and mention.kind
                is ScopeEntityKind.AUTHOR
            )
            for version_id
            in mention.version_ids
        }

        if requested_author_versions:
            matching_works = tuple(
                work
                for work
                in self._in_corpus_works
                if (
                    requested_author_versions
                    .intersection(
                        work.version_ids
                    )
                )
            )

            if matching_works:
                return matching_works

        return self._in_corpus_works


def _query_without_scope_mentions(
    query_text: str,
    *,
    mentions: tuple[
        ScopeMention,
        ...,
    ],
) -> str:
    cleaned = query_text

    aliases = sorted(
        {
            mention.matched_alias
            for mention in mentions
            if mention.matched_alias
        },
        key=len,
        reverse=True,
    )

    for alias in aliases:
        cleaned = cleaned.replace(
            alias,
            " ",
        )

    return _compact_query_text(
        cleaned
    )


def _remove_paratext_markers(
    query_text: str,
) -> str:
    cleaned = query_text

    patterns = (
        (
            r"\bدر\s+مقدمه"
            r"(?:\s+(?:کتاب|اثر))?"
        ),
        (
            r"\bمقدمه"
            r"(?:\s+(?:کتاب|اثر))?"
        ),
        (
            r"\bدر\s+فهرست"
            r"(?:\s+مطالب)?"
        ),
        (
            r"\bفهرست"
            r"(?:\s+مطالب)?"
        ),
        r"\bدر\s+نمایه",
        r"\bنمایه",
        (
            r"\bدر\s+واژه"
            r"[\u200c ]?نامه"
        ),
        (
            r"\bواژه"
            r"[\u200c ]?نامه"
        ),
    )

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
        )

    return _compact_query_text(
        cleaned
    )


def _compact_query_text(
    query_text: str,
) -> str:
    compact = " ".join(
        query_text.split()
    )

    compact = re.sub(
        (
            r"^در\s+"
            r"(?:دیوان|کتاب|اثر)ش\s+"
        ),
        "",
        compact,
    )

    compact = re.sub(
        r"^(?:و|یا)\s+",
        "",
        compact,
    )

    compact = re.sub(
        r"\bدر\s+(?=درباره\b)",
        "",
        compact,
    )

    compact = re.sub(
        r"\s+([،؛؟?!.,:])",
        r"\1",
        compact,
    )

    compact = compact.strip(
        " ،؛,:"
    )

    if not compact:
        return (
            "درباره موضوع پرسش "
            "چه آمده است؟"
        )

    return compact


def _restrict_to_primary_text(
    query_text: str,
    *,
    work_label: str,
) -> str:
    return (
        "فقط بر اساس متن اصلی "
        f"{work_label}: "
        f"{query_text}"
    )


def _suggestion_kind(
    reason: AbstentionReason,
) -> str:
    if reason is (
        AbstentionReason
        .KNOWN_OUT_OF_CORPUS_SCOPE
    ):
        return "replace_out_of_scope"

    if reason is (
        AbstentionReason
        .SOURCE_ATTRIBUTION_CONFLICT
    ):
        return "restrict_scope"

    if reason in {
        AbstentionReason.TOP_HIT_PARATEXT,
        AbstentionReason.TOP_HIT_MIXED,
    }:
        return "search_primary_text"

    return "scope_query"


__all__ = [
    "QUERY_SUGGESTION_SCHEMA_VERSION",
    "QuerySuggestion",
    "QuerySuggestionEngine",
]
