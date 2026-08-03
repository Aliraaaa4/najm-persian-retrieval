"""Tests for the public FastAPI retrieval interface."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import (
    TestClient,
)

from najm_retrieval.api.app import (
    create_app,
)
from najm_retrieval.api.query_suggestions import (
    QuerySuggestion,
)
from najm_retrieval.retrieval import (
    AbstentionReason,
    DecisionAction,
    RetrievedPassage,
    TrustedRetrievalResponse,
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
class _FakeService:
    response: (
        TrustedRetrievalResponse
        | None
    ) = None

    error: Exception | None = None

    def __post_init__(
        self,
    ) -> None:
        self.calls: list[str] = []

    def search(
        self,
        query_text: str,
    ) -> TrustedRetrievalResponse:
        self.calls.append(
            query_text
        )

        if self.error is not None:
            raise self.error

        assert self.response is not None

        return self.response


@dataclass
class _FakeSuggestionEngine:
    suggestions: tuple[
        QuerySuggestion,
        ...,
    ] = ()

    error: Exception | None = None

    def __post_init__(
        self,
    ) -> None:
        self.calls: list[
            tuple[
                str,
                AbstentionReason,
                bool,
            ]
        ] = []

    def suggest(
        self,
        *,
        query_text: str,
        reason: AbstentionReason,
        return_results: bool,
    ) -> tuple[
        QuerySuggestion,
        ...,
    ]:
        self.calls.append(
            (
                query_text,
                reason,
                return_results,
            )
        )

        if self.error is not None:
            raise self.error

        return self.suggestions


def _passage(
    passage_id: str,
    *,
    rank: int,
    ordinal: int,
) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=passage_id,
        version_id=VERSION_ID,
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
        display_text=(
            f"متن کامل Passage {ordinal}"
        ),
        snippet=(
            f"خلاصه Passage {ordinal}"
        ),
        heading_path=(
            "دفتر اول",
        ),
        section_path=(
            "دفتر اول",
            "بخش آغازین",
        ),
        previous_passage_id=None,
        next_passage_id=None,
        rank=rank,
        fusion_score=(
            0.04 / rank
        ),
        lexical_rank=rank,
        dense_rank=rank + 1,
        lexical_bm25_score=(
            -3.5 * rank
        ),
        dense_cosine_score=(
            0.90 - rank / 100
        ),
    )


def _trusted_response(
    *,
    return_results: bool,
) -> TrustedRetrievalResponse:
    first = _passage(
        PASSAGE_ONE,
        rank=1,
        ordinal=1,
    )

    second = _passage(
        PASSAGE_TWO,
        rank=2,
        ordinal=2,
    )

    reason = (
        RETURN_REASON
        if return_results
        else ABSTAIN_REASON
    )

    return TrustedRetrievalResponse(
        query_text="پرسش آزمایشی",
        action=(
            RETURN_ACTION
            if return_results
            else ABSTAIN_ACTION
        ),
        reason=reason,
        return_results=(
            return_results
        ),
        top_passage_id=(
            PASSAGE_ONE
        ),
        triggered_reasons=(
            reason,
        ),
        passages=(
            (
                first,
                second,
            )
            if return_results
            else ()
        ),
        diagnostic_passages=(
            first,
            second,
        ),
        retrieval_latency_ms=12.5,
    )


def test_health_endpoint_is_live() -> None:
    app = create_app(
        runtime_loader=None
    )

    with TestClient(
        app
    ) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200

    assert response.json() == {
        "schema_version": "1.1.0",
        "status": "ok",
        "service": (
            "najm-persian-retrieval"
        ),
        "version": "0.1.0",
    }


def test_ready_endpoint_reports_injected_service() -> None:
    app = create_app(
        service=_FakeService(
            response=_trusted_response(
                return_results=True
            )
        )
    )

    with TestClient(
        app
    ) as client:
        response = client.get(
            "/ready"
        )

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["status"] == "ready"


def test_ready_endpoint_reports_loader_failure() -> None:
    def failing_loader():
        raise RuntimeError(
            "private startup detail"
        )

    app = create_app(
        runtime_loader=(
            failing_loader
        )
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.get(
            "/ready"
        )

    assert response.status_code == 503

    payload = response.json()

    assert payload["ready"] is False
    assert payload["status"] == (
        "not_ready"
    )

    assert (
        "private startup detail"
        not in response.text
    )


def test_retrieve_returns_referenced_public_results() -> None:
    service = _FakeService(
        response=_trusted_response(
            return_results=True
        )
    )

    app = create_app(
        service=service
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": (
                    "پرسش آزمایشی"
                ),
                "limit": 1,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert service.calls == [
        "پرسش آزمایشی",
    ]

    assert payload[
        "return_results"
    ] is True

    assert payload[
        "result_count"
    ] == 1

    assert len(
        payload["results"]
    ) == 1

    result = payload[
        "results"
    ][0]

    assert result["rank"] == 1

    assert result[
        "reference"
    ][
        "passage_id"
    ] == PASSAGE_ONE

    assert result[
        "reference"
    ][
        "author_name"
    ] == (
        "مولانا جلال‌الدین رومی"
    )

    assert result[
        "reference"
    ][
        "work_title"
    ] == "مثنوی معنوی"

    assert (
        "diagnostic_passages"
        not in payload
    )


def test_retrieve_abstention_hides_diagnostics() -> None:
    app = create_app(
        service=_FakeService(
            response=_trusted_response(
                return_results=False
            )
        )
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": (
                    "حافظ درباره عشق"
                ),
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "return_results"
    ] is False

    assert payload[
        "results"
    ] == []

    assert payload[
        "result_count"
    ] == 0

    assert payload[
        "top_passage_id"
    ] is None

    assert (
        "خارج از مجموعه"
        in payload["message"]
    )

    assert (
        PASSAGE_ONE
        not in response.text
    )


def test_retrieve_returns_safe_suggestions_on_abstention(
) -> None:
    engine = _FakeSuggestionEngine(
        suggestions=(
            QuerySuggestion(
                query_text=(
                    "مثنوی معنوی درباره عشق"
                ),
                label=(
                    "جست‌وجوی همین پرسش "
                    "در مثنوی معنوی"
                ),
                kind=(
                    "replace_out_of_scope"
                ),
                entity_id=(
                    "0672JalalDinRumi.Mathnawi"
                ),
                entity_kind="work",
                version_ids=(
                    VERSION_ID,
                ),
            ),
        )
    )

    app = create_app(
        service=_FakeService(
            response=_trusted_response(
                return_results=False
            )
        ),
        suggestion_engine=engine,
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": (
                    "حافظ درباره عشق"
                ),
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["results"] == []

    assert payload["suggestions"] == [
        {
            "query": (
                "مثنوی معنوی درباره عشق"
            ),
            "label": (
                "جست‌وجوی همین پرسش "
                "در مثنوی معنوی"
            ),
            "kind": (
                "replace_out_of_scope"
            ),
            "entity_id": (
                "0672JalalDinRumi.Mathnawi"
            ),
            "entity_kind": "work",
            "version_ids": [
                VERSION_ID,
            ],
        },
    ]

    assert engine.calls == [
        (
            "پرسش آزمایشی",
            ABSTAIN_REASON,
            False,
        ),
    ]

    assert (
        PASSAGE_ONE
        not in response.text
    )


def test_retrieve_skips_suggestions_when_returning_results(
) -> None:
    engine = _FakeSuggestionEngine()

    app = create_app(
        service=_FakeService(
            response=_trusted_response(
                return_results=True
            )
        ),
        suggestion_engine=engine,
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": (
                    "پرسش آزمایشی"
                ),
            },
        )

    assert response.status_code == 200

    assert response.json()[
        "suggestions"
    ] == []

    assert engine.calls == []


def test_retrieve_survives_suggestion_engine_failure(
) -> None:
    engine = _FakeSuggestionEngine(
        error=RuntimeError(
            "private suggestion failure"
        )
    )

    app = create_app(
        service=_FakeService(
            response=_trusted_response(
                return_results=False
            )
        ),
        suggestion_engine=engine,
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": (
                    "حافظ درباره عشق"
                ),
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["suggestions"] == []
    assert payload["results"] == []

    assert (
        "private suggestion failure"
        not in response.text
    )


def test_retrieve_rejects_blank_query() -> None:
    app = create_app(
        service=_FakeService(
            response=_trusted_response(
                return_results=True
            )
        )
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": "   ",
            },
        )

    assert response.status_code == 422


def test_retrieve_rejects_unknown_fields() -> None:
    app = create_app(
        service=_FakeService(
            response=_trusted_response(
                return_results=True
            )
        )
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": (
                    "پرسش آزمایشی"
                ),
                "unknown": True,
            },
        )

    assert response.status_code == 422


def test_retrieve_hides_internal_service_errors() -> None:
    app = create_app(
        service=_FakeService(
            error=RuntimeError(
                "private retrieval detail"
            )
        )
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "query": (
                    "پرسش آزمایشی"
                ),
            },
        )

    assert response.status_code == 500

    payload = response.json()

    assert payload[
        "detail"
    ][
        "code"
    ] == "retrieval_failed"

    assert (
        "private retrieval detail"
        not in response.text
    )
