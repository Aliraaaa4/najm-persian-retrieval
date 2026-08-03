from __future__ import annotations

from pathlib import Path

import pytest

from najm_retrieval.api.runtime import (
    ApiRuntimeError,
    ApiSettings,
)


_RUNTIME_ENVIRONMENT_VARIABLES = (
    "NAJM_PROJECT_ROOT",
    "NAJM_CORPUS_ARTIFACT_ID",
    "NAJM_DENSE_MODEL_NAME",
    "NAJM_DENSE_LOCAL_FILES_ONLY",
    "NAJM_RETURN_LIMIT",
    "NAJM_LEXICAL_INDEX_PATH",
    "NAJM_PASSAGE_STORE_PATH",
    "NAJM_DENSE_ARTIFACT_ROOT",
    "NAJM_POLICY_PATH",
    "NAJM_CORPUS_MANIFEST_PATH",
    "NAJM_SCOPE_ALIASES_PATH",
    "NAJM_PARATEXT_PATH",
)


def _clear_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable_name in _RUNTIME_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(
            variable_name,
            raising=False,
        )


def test_runtime_defaults_use_extracted_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(
        "NAJM_PROJECT_ROOT",
        str(tmp_path),
    )

    settings = ApiSettings.from_environment()

    runtime_root = (
        tmp_path.resolve()
        / "artifacts"
        / "runtime"
        / "corpus-ad111acd912e"
    )

    assert settings.lexical_index_path == (
        runtime_root / "lexical.sqlite3"
    )
    assert settings.passage_store_path == (
        runtime_root / "passage_store.sqlite3"
    )
    assert settings.dense_artifact_root == (
        runtime_root
        / "dense"
        / "intfloat__multilingual-e5-small"
    )
    assert settings.dense_local_files_only is False


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("YES", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("No", False),
        ("off", False),
    ],
)
def test_runtime_parses_dense_download_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_value: str,
    expected: bool,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(
        "NAJM_PROJECT_ROOT",
        str(tmp_path),
    )
    monkeypatch.setenv(
        "NAJM_DENSE_LOCAL_FILES_ONLY",
        raw_value,
    )

    settings = ApiSettings.from_environment()

    assert settings.dense_local_files_only is expected


def test_runtime_rejects_invalid_dense_download_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(
        "NAJM_PROJECT_ROOT",
        str(tmp_path),
    )
    monkeypatch.setenv(
        "NAJM_DENSE_LOCAL_FILES_ONLY",
        "sometimes",
    )

    with pytest.raises(
        ApiRuntimeError,
        match="NAJM_DENSE_LOCAL_FILES_ONLY",
    ):
        ApiSettings.from_environment()


def test_runtime_keeps_explicit_artifact_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(
        "NAJM_PROJECT_ROOT",
        str(tmp_path),
    )

    lexical_path = tmp_path / "custom-lexical.sqlite3"
    passage_path = tmp_path / "custom-passage.sqlite3"
    dense_root = tmp_path / "custom-dense"

    monkeypatch.setenv(
        "NAJM_LEXICAL_INDEX_PATH",
        str(lexical_path),
    )
    monkeypatch.setenv(
        "NAJM_PASSAGE_STORE_PATH",
        str(passage_path),
    )
    monkeypatch.setenv(
        "NAJM_DENSE_ARTIFACT_ROOT",
        str(dense_root),
    )

    settings = ApiSettings.from_environment()

    assert settings.lexical_index_path == (
        lexical_path.resolve()
    )
    assert settings.passage_store_path == (
        passage_path.resolve()
    )
    assert settings.dense_artifact_root == (
        dense_root.resolve()
    )
