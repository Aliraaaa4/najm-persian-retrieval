"""Tests for enriching trusted retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from najm_retrieval.retrieval import (
    AbstentionReason,
    DecisionAction,
    HybridSearchHit,
    LexicalSearchMode,
    PassageStoreRecord,
    RetrievalDecision,
    RetrievalService,
    RetrievalServiceError,
)


VERSION_ID = (
    "0672JalalDinRumi."
    "Mathnawi.PDL00048-per1"
)

PASSAGE_ONE = (
    f"{VERSION_ID}:"
    "passage_000001"
)

PASSAGE_TWO = (
    f"{VERSION_ID}:"
    "passage_000002"
)


RETURN_ACTION = next(
    item
    for item in DecisionAction
    if item.value == "return_results"
)

ABSTAIN_ACTION = next(
    item
    for item in DecisionAction
    if item.value == "abstain"
)

RETURN_REASON = next(
    item
    for item in AbstentionReason
    if (
        item.value
        == "baseline_evidence_passed"
    )
)

ABSTAIN_REASON = next(
    item
    for item in AbstentionReason
    if (
        item.value
        == "known_out_of_corpus_scope"
    )
)


@dataclass
class _FakeTrustedResult:
    query_text: str
    decision: RetrievalDecision
    retrieval_run: object
    hits: tuple[
        HybridSearchHit,
        ...,
    ]
    diagnostic_hits: tuple[
        HybridSearchHit,
        ...,
    ]


class _FakeTrustedRetriever:
    def __init__(
        self,
        result: _FakeTrustedResult,
    ) -> None:
        self.result = result
        self.calls: list[
            tuple[
                str,
                LexicalSearchMode,
            ]
        ] = []

    def search(
        self,
        query_text: str,
        *,
        lexical_mode: LexicalSearchMode = (
            LexicalSearchMode.AUTO
        ),
    ) -> _FakeTrustedResult:
        self.calls.append(
            (
                query_text,
                lexical_mode,
            )
        )

        return self.result


class _FakePassageStore:
    def __init__(
        self,
        records: tuple[
            PassageStoreRecord,
            ...,
        ],
    ) -> None:
        self.records = {
            record.passage_id: record
            for record in records
        }

        self.requested_ids: tuple[
            str,
            ...,
        ] = ()

    def get_many(
        self,
        passage_ids: object,
    ) -> tuple[
        PassageStoreRecord,
        ...,
    ]:
        requested = tuple(
            passage_ids
        )

        self.requested_ids = (
            requested
        )

        return tuple(
            self.records[passage_id]
            for passage_id
            in reversed(
                requested
            )
            if passage_id in self.records
        )


def _hit(
    passage_id: str,
    *,
    rank: int,
) -> HybridSearchHit:
    return HybridSearchHit(
        passage_id=passage_id,
        version_id=VERSION_ID,
        kind="mathnawi",
        rank=rank,
        fusion_score=(
            0.03
            / rank
        ),
        lexical_rank=rank,
        dense_rank=rank + 1,
        lexical_bm25_score=(
            -2.5
            * rank
        ),
        dense_cosine_score=(
            0.91
            - rank / 100
        ),
    )


def _record(
    passage_id: str,
    *,
    ordinal: int,
    display_text: str,
    version_id: str = VERSION_ID,
) -> PassageStoreRecord:
    return PassageStoreRecord(
        passage_id=passage_id,
        version_id=version_id,
        author_id=(
            "0672JalalDinRumi"
        ),
        author_name=(
            "مولانا جلال‌الدین رومی"
        ),
        work_id=(
            "0672JalalDinRumi.Mathnawi"
        ),
        work_title="مثنوی معنوی",
        profile="structured_poetry",
        kind="mathnawi",
        ordinal=ordinal,
        display_text=display_text,
        retrieval_text=display_text,
        search_alias_text=display_text,
        previous_passage_id=None,
        next_passage_id=None,
        heading_path=(
            "دفتر اول",
        ),
        section_path=(
            "دفتر اول",
            "بخش آغازین",
        ),
        source_unit_ids=(
            f"{VERSION_ID}:"
            f"verse_{ordinal:04d}",
        ),
        word_count=12,
        unit_count=1,
        member_count=1,
    )


def _result(
    *,
    return_results: bool,
) -> _FakeTrustedResult:
    first = _hit(
        PASSAGE_ONE,
        rank=1,
    )

    second = _hit(
        PASSAGE_TWO,
        rank=2,
    )

    reason = (
        RETURN_REASON
        if return_results
        else ABSTAIN_REASON
    )

    action = (
        RETURN_ACTION
        if return_results
        else ABSTAIN_ACTION
    )

    decision = RetrievalDecision(
        query_text="پرسش آزمایشی",
        action=action,
        reason=reason,
        return_results=return_results,
        top_passage_id=(
            PASSAGE_ONE
        ),
        triggered_reasons=(
            reason,
        ),
    )

    return _FakeTrustedResult(
        query_text="پرسش آزمایشی",
        decision=decision,
        retrieval_run=(
            SimpleNamespace(
                hybrid_result=(
                    SimpleNamespace(
                        latency_ms=12.5
                    )
                )
            )
        ),
        hits=(
            (
                first,
                second,
            )
            if return_results
            else ()
        ),
        diagnostic_hits=(
            first,
            second,
        ),
    )


def _store() -> _FakePassageStore:
    return _FakePassageStore(
        (
            _record(
                PASSAGE_ONE,
                ordinal=1,
                display_text=(
                    "  متن Passage اول\n"
                    "با فاصله‌های متعدد  "
                ),
            ),
            _record(
                PASSAGE_TWO,
                ordinal=2,
                display_text=(
                    "متن Passage دوم"
                ),
            ),
        )
    )


def test_service_enriches_hits_and_preserves_hybrid_order() -> None:
    retriever = (
        _FakeTrustedRetriever(
            _result(
                return_results=True
            )
        )
    )

    store = _store()

    service = RetrievalService(
        retriever,
        store,
        snippet_chars=24,
    )

    response = service.search(
        "پرسش آزمایشی"
    )

    assert retriever.calls == [
        (
            "پرسش آزمایشی",
            LexicalSearchMode.AUTO,
        ),
    ]

    assert store.requested_ids == (
        PASSAGE_ONE,
        PASSAGE_TWO,
    )

    assert tuple(
        passage.passage_id
        for passage
        in response.passages
    ) == (
        PASSAGE_ONE,
        PASSAGE_TWO,
    )

    assert tuple(
        passage.passage_id
        for passage
        in response.diagnostic_passages
    ) == (
        PASSAGE_ONE,
        PASSAGE_TWO,
    )

    first = response.passages[0]

    assert first.author_name == (
        "مولانا جلال‌الدین رومی"
    )

    assert first.work_title == (
        "مثنوی معنوی"
    )

    assert first.rank == 1
    assert first.lexical_rank == 1
    assert first.dense_rank == 2

    assert len(
        first.snippet
    ) <= 24

    assert "\n" not in first.snippet

    assert response.retrieval_latency_ms == (
        12.5
    )


def test_abstention_hides_public_hits_but_keeps_diagnostics() -> None:
    service = RetrievalService(
        _FakeTrustedRetriever(
            _result(
                return_results=False
            )
        ),
        _store(),
    )

    response = service.search(
        "پرسش آزمایشی"
    )

    assert not response.return_results
    assert response.passages == ()

    assert tuple(
        passage.passage_id
        for passage
        in response.diagnostic_passages
    ) == (
        PASSAGE_ONE,
        PASSAGE_TWO,
    )


def test_missing_passage_store_record_is_rejected() -> None:
    store = _FakePassageStore(
        (
            _record(
                PASSAGE_ONE,
                ordinal=1,
                display_text="متن اول",
            ),
        )
    )

    service = RetrievalService(
        _FakeTrustedRetriever(
            _result(
                return_results=True
            )
        ),
        store,
    )

    with pytest.raises(
        RetrievalServiceError,
        match="missing retrieval hits",
    ):
        service.search(
            "پرسش آزمایشی"
        )


def test_version_mismatch_between_hit_and_store_is_rejected() -> None:
    store = _FakePassageStore(
        (
            _record(
                PASSAGE_ONE,
                ordinal=1,
                display_text="متن اول",
                version_id=(
                    "0672JalalDinRumi."
                    "Diwan.PDL00047-per1"
                ),
            ),
            _record(
                PASSAGE_TWO,
                ordinal=2,
                display_text="متن دوم",
            ),
        )
    )

    service = RetrievalService(
        _FakeTrustedRetriever(
            _result(
                return_results=True
            )
        ),
        store,
    )

    with pytest.raises(
        RetrievalServiceError,
        match="Version mismatch",
    ):
        service.search(
            "پرسش آزمایشی"
        )


def test_response_rejects_public_passages_for_abstention() -> None:
    service = RetrievalService(
        _FakeTrustedRetriever(
            _result(
                return_results=True
            )
        ),
        _store(),
    )

    response = service.search(
        "پرسش آزمایشی"
    )

    with pytest.raises(
        ValueError,
        match=(
            "abstention response"
        ),
    ):
        replace(
            response,
            return_results=False,
        )


def test_response_rejects_public_hit_missing_from_diagnostics() -> None:
    service = RetrievalService(
        _FakeTrustedRetriever(
            _result(
                return_results=True
            )
        ),
        _store(),
    )

    response = service.search(
        "پرسش آزمایشی"
    )

    with pytest.raises(
        ValueError,
        match=(
            "contained in diagnostic"
        ),
    ):
        replace(
            response,
            top_passage_id=(
                PASSAGE_TWO
            ),
            passages=(
                response.passages[1],
            ),
            diagnostic_passages=(
                response.diagnostic_passages[
                    0
                ],
            ),
        )
