"""FastAPI application for trusted Persian passage retrieval."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from threading import Lock
from typing import Any, Protocol

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)

from najm_retrieval.api.models import (
    HealthResponse,
    PassageReferenceResponse,
    ReadinessResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievalResultResponse,
    RetrievalScoresResponse,
)
from najm_retrieval.api.runtime import (
    build_retrieval_service,
)
from najm_retrieval.retrieval import (
    RetrievalService,
    TrustedRetrievalResponse,
)


logger = logging.getLogger(
    __name__
)


class _RetrievalServiceProtocol(
    Protocol
):
    def search(
        self,
        query_text: str,
    ) -> TrustedRetrievalResponse:
        """Return one trusted retrieval response."""


RuntimeLoader = Callable[
    [],
    _RetrievalServiceProtocol,
]


def create_app(
    *,
    service: (
        _RetrievalServiceProtocol
        | None
    ) = None,
    runtime_loader: (
        RuntimeLoader
        | None
    ) = build_retrieval_service,
) -> FastAPI:
    """Create an API without loading the dense model during import."""

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ):
        if (
            application.state.retrieval_service
            is None
            and runtime_loader
            is not None
        ):
            try:
                application.state.retrieval_service = (
                    runtime_loader()
                )

            except Exception as error:
                logger.exception(
                    "NAJM retrieval runtime "
                    "initialization failed."
                )

                application.state.startup_error = (
                    type(error).__name__
                )

        yield

    application = FastAPI(
        title=(
            "NAJM Persian Retrieval API"
        ),
        description=(
            "Local trusted retrieval over "
            "selected historical Persian texts."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.state.retrieval_service = (
        service
    )

    application.state.startup_error = (
        None
    )

    application.state.retrieval_lock = (
        Lock()
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=[
            "system",
        ],
    )
    def health(
    ) -> HealthResponse:
        """Return process liveness."""

        return HealthResponse(
            status="ok",
            service=(
                "najm-persian-retrieval"
            ),
            version="0.1.0",
        )

    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ReadinessResponse,
            },
        },
        tags=[
            "system",
        ],
    )
    def ready(
        request: Request,
        response: Response,
    ) -> ReadinessResponse:
        """Return whether retrieval artifacts are available."""

        is_ready = (
            request.app.state
            .retrieval_service
            is not None
        )

        if not is_ready:
            response.status_code = (
                status.HTTP_503_SERVICE_UNAVAILABLE
            )

            detail = (
                "Runtime initialization failed."
                if (
                    request.app.state
                    .startup_error
                    is not None
                )
                else (
                    "Retrieval service is "
                    "not initialized."
                )
            )

            return ReadinessResponse(
                status="not_ready",
                ready=False,
                detail=detail,
            )

        return ReadinessResponse(
            status="ready",
            ready=True,
            detail=(
                "Retrieval service is available."
            ),
        )

    @application.post(
        "/v1/retrieve",
        response_model=RetrieveResponse,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": (
                    "Retrieval runtime is unavailable."
                ),
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {
                "description": (
                    "Retrieval execution failed."
                ),
            },
        },
        tags=[
            "retrieval",
        ],
    )
    def retrieve(
        payload: RetrieveRequest,
        request: Request,
    ) -> RetrieveResponse:
        """Return trusted passages with stable source references."""

        retrieval_service = (
            request.app.state
            .retrieval_service
        )

        if retrieval_service is None:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail={
                    "code": "not_ready",
                    "message": (
                        "Retrieval service "
                        "is not ready."
                    ),
                },
            )

        try:
            with (
                request.app.state
                .retrieval_lock
            ):
                result = (
                    retrieval_service.search(
                        payload.query
                    )
                )

        except Exception:
            logger.exception(
                "Retrieval request failed."
            )

            raise HTTPException(
                status_code=(
                    status
                    .HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail={
                    "code": "retrieval_failed",
                    "message": (
                        "The retrieval request "
                        "could not be completed."
                    ),
                },
            ) from None

        return _public_response(
            result,
            limit=payload.limit,
        )

    return application


def _public_response(
    result: TrustedRetrievalResponse,
    *,
    limit: int,
) -> RetrieveResponse:
    public_passages = (
        result.passages[
            :limit
        ]
    )

    serialized = [
        _serialize_passage(
            passage
        )
        for passage in public_passages
    ]

    return RetrieveResponse(
        query=result.query_text,
        action=result.action.value,
        reason=result.reason.value,
        return_results=(
            result.return_results
        ),
        message=_response_message(
            result
        ),
        top_passage_id=(
            result.top_passage_id
            if result.return_results
            else None
        ),
        triggered_reasons=[
            reason.value
            for reason
            in result.triggered_reasons
        ],
        result_count=len(
            serialized
        ),
        results=serialized,
        retrieval_latency_ms=(
            result.retrieval_latency_ms
        ),
    )


def _serialize_passage(
    passage: Any,
) -> RetrievalResultResponse:
    return RetrievalResultResponse(
        rank=passage.rank,
        snippet=passage.snippet,
        display_text=(
            passage.display_text
        ),
        reference=(
            PassageReferenceResponse(
                passage_id=(
                    passage.passage_id
                ),
                version_id=(
                    passage.version_id
                ),
                author_id=(
                    passage.author_id
                ),
                author_name=(
                    passage.author_name
                ),
                work_id=(
                    passage.work_id
                ),
                work_title=(
                    passage.work_title
                ),
                profile=passage.profile,
                kind=passage.kind,
                ordinal=passage.ordinal,
                heading_path=list(
                    passage.heading_path
                ),
                section_path=list(
                    passage.section_path
                ),
                previous_passage_id=(
                    passage
                    .previous_passage_id
                ),
                next_passage_id=(
                    passage
                    .next_passage_id
                ),
            )
        ),
        scores=(
            RetrievalScoresResponse(
                fusion_score=(
                    passage.fusion_score
                ),
                lexical_rank=(
                    passage.lexical_rank
                ),
                dense_rank=(
                    passage.dense_rank
                ),
                lexical_bm25_score=(
                    passage
                    .lexical_bm25_score
                ),
                dense_cosine_score=(
                    passage
                    .dense_cosine_score
                ),
            )
        ),
    )


def _response_message(
    result: TrustedRetrievalResponse,
) -> str:
    if result.return_results:
        return (
            "نتایج مرتبط از مجموعه "
            "بازیابی شد."
        )

    reason = result.reason.value

    if reason == (
        "known_out_of_corpus_scope"
    ):
        return (
            "پرسش به نویسنده یا اثری "
            "خارج از مجموعه فعلی اشاره دارد."
        )

    if reason == (
        "weak_cross_retriever_evidence"
    ):
        return (
            "شواهد بازیابی برای ارائه "
            "نتیجه قابل‌اعتماد کافی نبود."
        )

    if "paratext" in reason:
        return (
            "نتیجه قابل‌اعتمادی از متن "
            "اصلی اثر پیدا نشد."
        )

    return (
        "نتیجه قابل‌اعتمادی برای این "
        "پرسش پیدا نشد."
    )


app = create_app()


__all__ = [
    "app",
    "create_app",
]
