"""Tests for the calibrated retrieval-profile lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from najm_retrieval.retrieval import (
    RetrievalProfileConfig,
    RetrievalProfileMismatchError,
    load_abstention_policy_config,
    validate_retrieval_profile,
)


CONFIG_PATH = Path(
    "config/abstention_policy.yaml"
)


def loaded_profile() -> RetrievalProfileConfig:
    return (
        load_abstention_policy_config(
            CONFIG_PATH
        ).retrieval_profile
    )


def validate(
    **overrides,
) -> None:
    values = {
        "corpus_artifact_id": (
            "corpus-ad111acd912e"
        ),
        "dense_model_name": (
            "intfloat/multilingual-e5-small"
        ),
        "lexical_weight": 2.0,
        "dense_weight": 1.0,
        "rrf_constant": 60.0,
        "candidate_limit": 100,
        "return_limit": 10,
    }

    values.update(
        overrides
    )

    validate_retrieval_profile(
        loaded_profile(),
        **values,
    )


def test_yaml_loads_exact_retrieval_profile() -> None:
    profile = loaded_profile()

    assert profile == RetrievalProfileConfig(
        corpus_artifact_id=(
            "corpus-ad111acd912e"
        ),
        dense_model_name=(
            "intfloat/multilingual-e5-small"
        ),
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=100,
        return_limit=10,
    )


def test_matching_runtime_profile_is_accepted() -> None:
    validate()


@pytest.mark.parametrize(
    (
        "field_name",
        "value",
    ),
    (
        (
            "corpus_artifact_id",
            "corpus-other",
        ),
        (
            "dense_model_name",
            "other/model",
        ),
        (
            "lexical_weight",
            1.0,
        ),
        (
            "dense_weight",
            2.0,
        ),
        (
            "rrf_constant",
            30.0,
        ),
        (
            "candidate_limit",
            50,
        ),
        (
            "return_limit",
            5,
        ),
    ),
)
def test_runtime_profile_mismatch_is_rejected(
    field_name: str,
    value,
) -> None:
    with pytest.raises(
        RetrievalProfileMismatchError,
        match=field_name,
    ):
        validate(
            **{
                field_name: value,
            }
        )


def test_return_limit_cannot_exceed_candidate_limit() -> None:
    with pytest.raises(
        ValueError,
        match="must not exceed",
    ):
        RetrievalProfileConfig(
            candidate_limit=5,
            return_limit=10,
        )


def test_both_weights_cannot_be_zero() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        RetrievalProfileConfig(
            lexical_weight=0.0,
            dense_weight=0.0,
        )
