"""Validate runtime retrieval settings against a frozen policy profile."""

from __future__ import annotations

import math

from najm_retrieval.retrieval.policy_models import (
    RetrievalProfileConfig,
)


class RetrievalProfileMismatchError(
    ValueError
):
    """Raised when runtime retrieval settings differ from calibration."""


def validate_retrieval_profile(
    expected: RetrievalProfileConfig,
    *,
    corpus_artifact_id: str,
    dense_model_name: str,
    lexical_weight: float,
    dense_weight: float,
    rrf_constant: float,
    candidate_limit: int,
    return_limit: int,
) -> None:
    """Require an exact calibrated retrieval profile."""

    actual = RetrievalProfileConfig(
        corpus_artifact_id=(
            corpus_artifact_id
        ),
        dense_model_name=(
            dense_model_name
        ),
        lexical_weight=(
            lexical_weight
        ),
        dense_weight=(
            dense_weight
        ),
        rrf_constant=(
            rrf_constant
        ),
        candidate_limit=(
            candidate_limit
        ),
        return_limit=(
            return_limit
        ),
    )

    mismatches: list[
        str
    ] = []

    for field_name in (
        "corpus_artifact_id",
        "dense_model_name",
        "candidate_limit",
        "return_limit",
    ):
        expected_value = getattr(
            expected,
            field_name,
        )
        actual_value = getattr(
            actual,
            field_name,
        )

        if actual_value != expected_value:
            mismatches.append(
                f"{field_name}: expected "
                f"{expected_value!r}, found "
                f"{actual_value!r}"
            )

    for field_name in (
        "lexical_weight",
        "dense_weight",
        "rrf_constant",
    ):
        expected_value = float(
            getattr(
                expected,
                field_name,
            )
        )
        actual_value = float(
            getattr(
                actual,
                field_name,
            )
        )

        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            mismatches.append(
                f"{field_name}: expected "
                f"{expected_value!r}, found "
                f"{actual_value!r}"
            )

    if mismatches:
        raise RetrievalProfileMismatchError(
            "Runtime retrieval profile does not match "
            "the calibrated abstention policy: "
            + "; ".join(
                mismatches
            )
        )


__all__ = [
    "RetrievalProfileMismatchError",
    "validate_retrieval_profile",
]
