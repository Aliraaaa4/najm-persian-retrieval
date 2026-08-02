"""Load a frozen abstention-policy configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from najm_retrieval.retrieval.policy_models import (
    ABSTENTION_POLICY_SCHEMA_VERSION,
    AbstentionPolicyConfig,
    RetrievalProfileConfig,
)


class AbstentionPolicyConfigError(
    ValueError
):
    """Raised when a policy configuration is invalid."""


def load_abstention_policy_config(
    path: str | Path,
) -> AbstentionPolicyConfig:
    """Load and validate one frozen policy YAML file."""

    config_path = Path(
        path
    )

    if not config_path.is_file():
        raise AbstentionPolicyConfigError(
            "Abstention policy config not found: "
            f"{config_path}"
        )

    try:
        raw = yaml.safe_load(
            config_path.read_text(
                encoding="utf-8-sig"
            )
        )
    except UnicodeDecodeError as error:
        raise AbstentionPolicyConfigError(
            "Abstention policy config is not valid UTF-8."
        ) from error
    except yaml.YAMLError as error:
        raise AbstentionPolicyConfigError(
            "Abstention policy config contains invalid YAML."
        ) from error

    root = _require_mapping(
        raw,
        "policy root",
    )

    _reject_unknown_keys(
        root,
        allowed={
            "schema_version",
            "policy_id",
            "calibration_split_id",
            "retrieval_profile",
            "rules",
        },
        label="policy root",
    )

    schema_version = _require_string(
        root,
        "schema_version",
    )

    if (
        schema_version
        != ABSTENTION_POLICY_SCHEMA_VERSION
    ):
        raise AbstentionPolicyConfigError(
            "Unsupported abstention-policy schema version: "
            f"{schema_version}"
        )

    retrieval_profile_raw = _require_mapping(
        root.get(
            "retrieval_profile"
        ),
        "retrieval_profile",
    )

    _reject_unknown_keys(
        retrieval_profile_raw,
        allowed={
            "corpus_artifact_id",
            "dense_model_name",
            "lexical_weight",
            "dense_weight",
            "rrf_constant",
            "candidate_limit",
            "return_limit",
        },
        label="retrieval_profile",
    )

    rules = _require_mapping(
        root.get(
            "rules"
        ),
        "rules",
    )

    _reject_unknown_keys(
        rules,
        allowed={
            "reject_known_out_of_corpus_scope",
            "reject_source_attribution_conflict",
            "reject_paratext_top_hit",
            "reject_mixed_top_hit",
            "weak_cross_retriever_evidence",
        },
        label="rules",
    )

    weak_rule = _require_mapping(
        rules.get(
            "weak_cross_retriever_evidence"
        ),
        "rules.weak_cross_retriever_evidence",
    )

    _reject_unknown_keys(
        weak_rule,
        allowed={
            "enabled",
            "dense_top_1_score_less_than",
            "overlap_at_10_less_than_or_equal",
        },
        label="rules.weak_cross_retriever_evidence",
    )

    try:
        retrieval_profile = RetrievalProfileConfig(
            corpus_artifact_id=_require_string(
                retrieval_profile_raw,
                "corpus_artifact_id",
            ),
            dense_model_name=_require_string(
                retrieval_profile_raw,
                "dense_model_name",
            ),
            lexical_weight=_require_number(
                retrieval_profile_raw,
                "lexical_weight",
            ),
            dense_weight=_require_number(
                retrieval_profile_raw,
                "dense_weight",
            ),
            rrf_constant=_require_number(
                retrieval_profile_raw,
                "rrf_constant",
            ),
            candidate_limit=_require_int(
                retrieval_profile_raw,
                "candidate_limit",
            ),
            return_limit=_require_int(
                retrieval_profile_raw,
                "return_limit",
            ),
        )

        return AbstentionPolicyConfig(
            policy_id=_require_string(
                root,
                "policy_id",
            ),
            calibration_split_id=_require_string(
                root,
                "calibration_split_id",
            ),
            retrieval_profile=(
                retrieval_profile
            ),
            reject_known_out_of_corpus_scope=(
                _require_bool(
                    rules,
                    "reject_known_out_of_corpus_scope",
                )
            ),
            reject_source_attribution_conflict=(
                _require_bool(
                    rules,
                    "reject_source_attribution_conflict",
                )
            ),
            reject_paratext_top_hit=(
                _require_bool(
                    rules,
                    "reject_paratext_top_hit",
                )
            ),
            reject_mixed_top_hit=(
                _require_bool(
                    rules,
                    "reject_mixed_top_hit",
                )
            ),
            reject_weak_cross_retriever_evidence=(
                _require_bool(
                    weak_rule,
                    "enabled",
                )
            ),
            weak_evidence_dense_top_1_threshold=(
                _require_number(
                    weak_rule,
                    "dense_top_1_score_less_than",
                )
            ),
            weak_evidence_max_overlap_at_10=(
                _require_int(
                    weak_rule,
                    "overlap_at_10_less_than_or_equal",
                )
            ),
            schema_version=(
                schema_version
            ),
        )
    except ValueError as error:
        raise AbstentionPolicyConfigError(
            str(
                error
            )
        ) from error


def _require_mapping(
    value: Any,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise AbstentionPolicyConfigError(
            f"{label} must be a YAML mapping."
        )

    return value


def _require_string(
    mapping: Mapping[str, Any],
    field_name: str,
) -> str:
    value = mapping.get(
        field_name
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise AbstentionPolicyConfigError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def _require_bool(
    mapping: Mapping[str, Any],
    field_name: str,
) -> bool:
    value = mapping.get(
        field_name
    )

    if not isinstance(
        value,
        bool,
    ):
        raise AbstentionPolicyConfigError(
            f"{field_name} must be Boolean."
        )

    return value


def _require_number(
    mapping: Mapping[str, Any],
    field_name: str,
) -> float:
    value = mapping.get(
        field_name
    )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise AbstentionPolicyConfigError(
            f"{field_name} must be numeric."
        )

    return float(
        value
    )


def _require_int(
    mapping: Mapping[str, Any],
    field_name: str,
) -> int:
    value = mapping.get(
        field_name
    )

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise AbstentionPolicyConfigError(
            f"{field_name} must be an integer."
        )

    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    *,
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(
        set(
            mapping
        )
        - allowed
    )

    if unknown:
        raise AbstentionPolicyConfigError(
            f"{label} contains unknown fields: "
            + ", ".join(
                unknown
            )
        )


__all__ = [
    "AbstentionPolicyConfigError",
    "load_abstention_policy_config",
]
