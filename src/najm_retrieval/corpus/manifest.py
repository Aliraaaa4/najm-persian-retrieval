"""Load and validate the project corpus manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from najm_retrieval.parsing.profiles import SUPPORTED_PROFILES


class ManifestError(ValueError):
    """Raised when the corpus manifest is missing or invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing the source dataset."""

    name: str
    repository: str
    commit: str


@dataclass(frozen=True)
class ReferenceVersionConfig:
    """A non-canonical version retained for audit or comparison."""

    version_id: str
    profile: str
    include_in_index: bool = False


@dataclass(frozen=True)
class WorkConfig:
    """Configuration for one work and its text versions."""

    work_id: str
    title_fa: str
    profile: str
    canonical_version: str
    include_in_index: bool
    reference_versions: tuple[ReferenceVersionConfig, ...] = ()

    @property
    def version_ids(self) -> tuple[str, ...]:
        """Return canonical and reference version identifiers."""

        return (
            self.canonical_version,
            *(version.version_id for version in self.reference_versions),
        )


@dataclass(frozen=True)
class CorpusManifest:
    """Validated project-level corpus configuration."""

    dataset: DatasetConfig
    works: dict[str, WorkConfig]

    def get_work(self, work_id: str) -> WorkConfig:
        """Return one work or raise a clear error."""

        try:
            return self.works[work_id]
        except KeyError as exc:
            raise ManifestError(f"Unknown work ID: {work_id}") from exc

    @property
    def canonical_versions(self) -> tuple[str, ...]:
        """Return all canonical version IDs."""

        return tuple(work.canonical_version for work in self.works.values())

    @property
    def reference_versions(self) -> tuple[str, ...]:
        """Return all reference version IDs."""

        return tuple(
            version.version_id
            for work in self.works.values()
            for version in work.reference_versions
        )

    @property
    def all_versions(self) -> tuple[str, ...]:
        """Return every configured version ID."""

        return self.canonical_versions + self.reference_versions

    @property
    def indexable_versions(self) -> tuple[str, ...]:
        """Return versions that should be included in the search index."""

        versions: list[str] = []

        for work in self.works.values():
            if work.include_in_index:
                versions.append(work.canonical_version)

            versions.extend(
                version.version_id
                for version in work.reference_versions
                if version.include_in_index
            )

        return tuple(versions)


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"'{field_name}' must be a YAML mapping.")

    return value


def _require_text(mapping: dict[str, Any], field_name: str, context: str) -> str:
    value = mapping.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ManifestError(
            f"'{field_name}' must be a non-empty string in {context}."
        )

    return value.strip()


def _read_boolean(
    mapping: dict[str, Any],
    field_name: str,
    *,
    default: bool,
    context: str,
) -> bool:
    value = mapping.get(field_name, default)

    if not isinstance(value, bool):
        raise ManifestError(f"'{field_name}' must be true or false in {context}.")

    return value


def _validate_profile(profile: str, context: str) -> None:
    if profile not in SUPPORTED_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_PROFILES))
        raise ManifestError(
            f"Unsupported profile '{profile}' in {context}. "
            f"Supported profiles: {supported}"
        )


def load_manifest(path: str | Path) -> CorpusManifest:
    """Load and validate a corpus manifest YAML file."""

    manifest_path = Path(path)

    if not manifest_path.is_file():
        raise ManifestError(f"Manifest file not found: {manifest_path}")

    try:
        # utf-8-sig also accepts files written by Windows PowerShell with a BOM.
        text = manifest_path.read_text(encoding="utf-8-sig")
        raw_data = yaml.safe_load(text)
    except UnicodeDecodeError as exc:
        raise ManifestError(
            f"Manifest is not valid UTF-8: {manifest_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ManifestError(
            f"Manifest contains invalid YAML: {manifest_path}"
        ) from exc

    root = _require_mapping(raw_data, "manifest root")

    dataset_raw = _require_mapping(root.get("dataset"), "dataset")
    dataset = DatasetConfig(
        name=_require_text(dataset_raw, "name", "dataset"),
        repository=_require_text(dataset_raw, "repository", "dataset"),
        commit=_require_text(dataset_raw, "commit", "dataset"),
    )

    works_raw = _require_mapping(root.get("works"), "works")

    if not works_raw:
        raise ManifestError("'works' must contain at least one configured work.")

    works: dict[str, WorkConfig] = {}
    seen_version_ids: set[str] = set()

    for work_id, work_value in works_raw.items():
        if not isinstance(work_id, str) or not work_id.strip():
            raise ManifestError("Every work ID must be a non-empty string.")

        work_id = work_id.strip()
        context = f"work '{work_id}'"
        work_raw = _require_mapping(work_value, context)

        title_fa = _require_text(work_raw, "title_fa", context)
        profile = _require_text(work_raw, "profile", context)
        canonical_version = _require_text(
            work_raw,
            "canonical_version",
            context,
        )
        include_in_index = _read_boolean(
            work_raw,
            "include_in_index",
            default=True,
            context=context,
        )

        _validate_profile(profile, context)

        if not canonical_version.startswith(f"{work_id}."):
            raise ManifestError(
                f"Canonical version '{canonical_version}' does not belong "
                f"to {context}."
            )

        if canonical_version in seen_version_ids:
            raise ManifestError(
                f"Duplicate version ID in manifest: {canonical_version}"
            )

        seen_version_ids.add(canonical_version)

        references_raw = work_raw.get("reference_versions", [])

        if references_raw is None:
            references_raw = []

        if not isinstance(references_raw, list):
            raise ManifestError(
                f"'reference_versions' must be a list in {context}."
            )

        reference_versions: list[ReferenceVersionConfig] = []

        for index, reference_value in enumerate(references_raw):
            reference_context = (
                f"reference version {index + 1} of work '{work_id}'"
            )
            reference_raw = _require_mapping(
                reference_value,
                reference_context,
            )

            version_id = _require_text(
                reference_raw,
                "version_id",
                reference_context,
            )
            reference_profile = _require_text(
                reference_raw,
                "profile",
                reference_context,
            )
            reference_include = _read_boolean(
                reference_raw,
                "include_in_index",
                default=False,
                context=reference_context,
            )

            _validate_profile(reference_profile, reference_context)

            if not version_id.startswith(f"{work_id}."):
                raise ManifestError(
                    f"Reference version '{version_id}' does not belong "
                    f"to work '{work_id}'."
                )

            if version_id in seen_version_ids:
                raise ManifestError(
                    f"Duplicate version ID in manifest: {version_id}"
                )

            seen_version_ids.add(version_id)

            reference_versions.append(
                ReferenceVersionConfig(
                    version_id=version_id,
                    profile=reference_profile,
                    include_in_index=reference_include,
                )
            )

        works[work_id] = WorkConfig(
            work_id=work_id,
            title_fa=title_fa,
            profile=profile,
            canonical_version=canonical_version,
            include_in_index=include_in_index,
            reference_versions=tuple(reference_versions),
        )

    return CorpusManifest(dataset=dataset, works=works)