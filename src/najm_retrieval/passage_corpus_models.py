"""Corpus-level models for passage generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from najm_retrieval.corpus.metadata import (
    DocumentMetadata,
)
from najm_retrieval.passages import (
    PassageBuildResult,
    PassageConfig,
)


@dataclass(frozen=True)
class BuiltPassageVersion:
    """Passage-building result for one parsed corpus version."""

    author_id: str
    work_id: str
    version_id: str

    profile: str
    include_in_index: bool
    is_canonical: bool

    source_path: Path
    build_result: PassageBuildResult

    document_metadata: DocumentMetadata | None = None

    def __post_init__(self) -> None:
        """Validate version identity and build contents."""

        for label, value in (
            ("author_id", self.author_id),
            ("work_id", self.work_id),
            ("version_id", self.version_id),
            ("profile", self.profile),
        ):
            if not value.strip():
                raise ValueError(
                    f"{label} must not be empty."
                )

        for passage in self.build_result.passages:
            if passage.version_id != self.version_id:
                raise ValueError(
                    "Every passage must belong to "
                    "the built version."
                )

            if passage.profile != self.profile:
                raise ValueError(
                    "Every passage must use the "
                    "built version profile."
                )

            if (
                passage.include_in_index
                != self.include_in_index
            ):
                raise ValueError(
                    "Passage index eligibility must "
                    "match the version."
                )

        if not self.include_in_index:
            if self.build_result.passages:
                raise ValueError(
                    "A reference-only version cannot "
                    "contain retrieval passages."
                )

        if self.document_metadata is not None:
            metadata = self.document_metadata

            if metadata.version_id != self.version_id:
                raise ValueError(
                    "Document metadata version ID "
                    "does not match."
                )

            if metadata.work_id != self.work_id:
                raise ValueError(
                    "Document metadata work ID "
                    "does not match."
                )

            if metadata.author_id != self.author_id:
                raise ValueError(
                    "Document metadata author ID "
                    "does not match."
                )

            if metadata.profile != self.profile:
                raise ValueError(
                    "Document metadata profile "
                    "does not match."
                )

            if (
                metadata.include_in_index
                != self.include_in_index
            ):
                raise ValueError(
                    "Document metadata index flag "
                    "does not match."
                )

    @property
    def passage_count(self) -> int:
        """Return generated passage count."""

        return self.build_result.passage_count

    @property
    def skipped_unit_count(self) -> int:
        """Return skipped source-unit count."""

        return len(
            self.build_result.skipped_unit_ids
        )


@dataclass(frozen=True)
class PassageCorpusBuildResult:
    """Complete passage-build result for one parsed corpus."""

    input_dir: Path
    config: PassageConfig

    versions: tuple[
        BuiltPassageVersion,
        ...,
    ]

    runtime_seconds: float

    def __post_init__(self) -> None:
        """Validate versions and shared configuration."""

        if self.runtime_seconds < 0:
            raise ValueError(
                "runtime_seconds must not be negative."
            )

        version_ids = [
            version.version_id
            for version in self.versions
        ]

        if len(version_ids) != len(
            set(version_ids)
        ):
            raise ValueError(
                "Version IDs must be unique."
            )

        for version in self.versions:
            if (
                version.build_result.config
                != self.config
            ):
                raise ValueError(
                    "Every version must use the "
                    "corpus passage configuration."
                )

    @property
    def passage_count(self) -> int:
        """Return total passage count."""

        return sum(
            version.passage_count
            for version in self.versions
        )

    @property
    def indexable_versions(
        self,
    ) -> tuple[BuiltPassageVersion, ...]:
        """Return versions eligible for indexing."""

        return tuple(
            version
            for version in self.versions
            if version.include_in_index
        )

    @property
    def reference_versions(
        self,
    ) -> tuple[BuiltPassageVersion, ...]:
        """Return reference-only versions."""

        return tuple(
            version
            for version in self.versions
            if not version.include_in_index
        )

    @property
    def skipped_unit_count(self) -> int:
        """Return total skipped source-unit count."""

        return sum(
            version.skipped_unit_count
            for version in self.versions
        )


__all__ = [
    "BuiltPassageVersion",
    "PassageCorpusBuildResult",
]
