"""Construct the real retrieval service from local artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import gettempdir

from najm_retrieval.api.query_suggestions import (
    QuerySuggestionEngine,
)
from najm_retrieval.retrieval import (
    CorpusScopeCatalog,
    DenseIndex,
    HybridRetriever,
    LexicalIndex,
    ParatextCatalog,
    ParatextEvidenceExtractor,
    PassageStore,
    RetrievalService,
    ScopeEvidenceExtractor,
    TrustedRetriever,
    load_abstention_policy_config,
)


class ApiRuntimeError(
    RuntimeError
):
    """Raised when runtime artifacts are missing or invalid."""


@dataclass(frozen=True)
class ApiSettings:
    """Paths and frozen retrieval settings used by the API."""

    project_root: Path

    lexical_index_path: Path
    passage_store_path: Path
    dense_artifact_root: Path

    policy_path: Path
    corpus_manifest_path: Path
    scope_aliases_path: Path
    paratext_path: Path

    corpus_artifact_id: str = (
        "corpus-ad111acd912e"
    )

    dense_model_name: str = (
        "intfloat/multilingual-e5-small"
    )

    return_limit: int = 10

    @classmethod
    def from_environment(
        cls,
    ) -> "ApiSettings":
        """Resolve settings from environment variables and local defaults."""

        project_root = _resolve_project_root()

        temporary_root = Path(
            gettempdir()
        )

        corpus_artifact_id = (
            os.environ.get(
                "NAJM_CORPUS_ARTIFACT_ID",
                "corpus-ad111acd912e",
            ).strip()
        )

        dense_model_name = (
            os.environ.get(
                "NAJM_DENSE_MODEL_NAME",
                (
                    "intfloat/"
                    "multilingual-e5-small"
                ),
            ).strip()
        )

        return_limit = _positive_integer(
            os.environ.get(
                "NAJM_RETURN_LIMIT",
                "10",
            ),
            field_name=(
                "NAJM_RETURN_LIMIT"
            ),
        )

        return cls(
            project_root=project_root,
            lexical_index_path=_environment_path(
                "NAJM_LEXICAL_INDEX_PATH",
                (
                    temporary_root
                    / "najm_real_lexical_index.sqlite3"
                ),
            ),
            passage_store_path=_environment_path(
                "NAJM_PASSAGE_STORE_PATH",
                (
                    temporary_root
                    / "najm_real_passage_store.sqlite3"
                ),
            ),
            dense_artifact_root=_environment_path(
                "NAJM_DENSE_ARTIFACT_ROOT",
                (
                    project_root
                    / "artifacts"
                    / "indexes"
                    / corpus_artifact_id
                    / "dense"
                    / "intfloat__multilingual-e5-small"
                ),
            ),
            policy_path=_environment_path(
                "NAJM_POLICY_PATH",
                (
                    project_root
                    / "config"
                    / "abstention_policy.yaml"
                ),
            ),
            corpus_manifest_path=_environment_path(
                "NAJM_CORPUS_MANIFEST_PATH",
                (
                    project_root
                    / "config"
                    / "corpus_manifest.yaml"
                ),
            ),
            scope_aliases_path=_environment_path(
                "NAJM_SCOPE_ALIASES_PATH",
                (
                    project_root
                    / "config"
                    / "scope_aliases.yaml"
                ),
            ),
            paratext_path=_environment_path(
                "NAJM_PARATEXT_PATH",
                (
                    project_root
                    / "config"
                    / "paratext_zones.yaml"
                ),
            ),
            corpus_artifact_id=(
                corpus_artifact_id
            ),
            dense_model_name=(
                dense_model_name
            ),
            return_limit=return_limit,
        )

    def validate_paths(
        self,
    ) -> None:
        """Require every runtime artifact before construction."""

        file_paths = (
            self.lexical_index_path,
            self.passage_store_path,
            self.policy_path,
            self.corpus_manifest_path,
            self.scope_aliases_path,
            self.paratext_path,
        )

        for path in file_paths:
            if not path.is_file():
                raise ApiRuntimeError(
                    "Required runtime file "
                    f"not found: {path}"
                )

        if not self.dense_artifact_root.is_dir():
            raise ApiRuntimeError(
                "Dense artifact directory "
                "not found: "
                f"{self.dense_artifact_root}"
            )

        if not self.corpus_artifact_id:
            raise ApiRuntimeError(
                "corpus_artifact_id must "
                "not be empty."
            )

        if not self.dense_model_name:
            raise ApiRuntimeError(
                "dense_model_name must "
                "not be empty."
            )

        if self.return_limit < 1:
            raise ApiRuntimeError(
                "return_limit must be "
                "at least 1."
            )


def build_retrieval_service(
    settings: ApiSettings | None = None,
) -> RetrievalService:
    """Build the production retrieval-service object graph."""

    resolved = (
        settings
        if settings is not None
        else ApiSettings.from_environment()
    )

    resolved.validate_paths()

    lexical_index = LexicalIndex(
        resolved.lexical_index_path
    )

    dense_index = DenseIndex(
        resolved.dense_artifact_root,
        device="cpu",
        local_files_only=True,
        verify_hashes=True,
    )

    hybrid_retriever = HybridRetriever(
        lexical_index,
        dense_index,
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=100,
    )

    policy_config = (
        load_abstention_policy_config(
            resolved.policy_path
        )
    )

    scope_catalog = (
        CorpusScopeCatalog.from_files(
            manifest_path=(
                resolved.corpus_manifest_path
            ),
            aliases_path=(
                resolved.scope_aliases_path
            ),
        )
    )

    paratext_catalog = (
        ParatextCatalog.from_yaml(
            resolved.paratext_path
        )
    )

    trusted_retriever = TrustedRetriever(
        hybrid_retriever,
        policy_config=policy_config,
        scope_extractor=(
            ScopeEvidenceExtractor(
                scope_catalog
            )
        ),
        paratext_extractor=(
            ParatextEvidenceExtractor(
                paratext_catalog
            )
        ),
        corpus_artifact_id=(
            resolved.corpus_artifact_id
        ),
        dense_model_name=(
            resolved.dense_model_name
        ),
        return_limit=(
            resolved.return_limit
        ),
    )

    passage_store = PassageStore(
        resolved.passage_store_path
    )

    return RetrievalService(
        trusted_retriever,
        passage_store,
        snippet_chars=280,
    )


def build_query_suggestion_engine(
    settings: ApiSettings | None = None,
) -> QuerySuggestionEngine:
    """Build the lightweight deterministic suggestion engine."""

    resolved = (
        settings
        if settings is not None
        else ApiSettings.from_environment()
    )

    required_paths = (
        resolved.corpus_manifest_path,
        resolved.scope_aliases_path,
    )

    for path in required_paths:
        if not path.is_file():
            raise ApiRuntimeError(
                "Required suggestion file "
                f"not found: {path}"
            )

    catalog = (
        CorpusScopeCatalog.from_files(
            manifest_path=(
                resolved.corpus_manifest_path
            ),
            aliases_path=(
                resolved.scope_aliases_path
            ),
        )
    )

    return QuerySuggestionEngine(
        catalog
    )


def _resolve_project_root(
) -> Path:
    configured = os.environ.get(
        "NAJM_PROJECT_ROOT"
    )

    if configured:
        return Path(
            configured
        ).expanduser().resolve()

    current = Path.cwd().resolve()

    if (
        current
        / "config"
        / "abstention_policy.yaml"
    ).is_file():
        return current

    return (
        Path(__file__)
        .resolve()
        .parents[3]
    )


def _environment_path(
    variable_name: str,
    default: Path,
) -> Path:
    value = os.environ.get(
        variable_name
    )

    if value:
        return Path(
            value
        ).expanduser().resolve()

    return default.resolve()


def _positive_integer(
    value: str,
    *,
    field_name: str,
) -> int:
    try:
        parsed = int(
            value
        )
    except ValueError as error:
        raise ApiRuntimeError(
            f"{field_name} must be an integer."
        ) from error

    if parsed < 1:
        raise ApiRuntimeError(
            f"{field_name} must be at least 1."
        )

    return parsed


__all__ = [
    "ApiRuntimeError",
    "ApiSettings",
    "build_query_suggestion_engine",
    "build_retrieval_service",
]
