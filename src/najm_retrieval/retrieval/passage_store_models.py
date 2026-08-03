"""Immutable records returned by the passage store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PASSAGE_STORE_SCHEMA_VERSION = "1.0.0"


def _require_non_empty_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


def _validate_string_tuple(
    value: tuple[str, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(
            f"{field_name} must be a tuple."
        )

    for item in value:
        _require_non_empty_text(
            item,
            field_name=field_name,
        )


@dataclass(frozen=True)
class PassageStoreRecord:
    """One display-ready passage and its source metadata."""

    passage_id: str
    version_id: str
    author_id: str
    author_name: str
    work_id: str
    work_title: str
    profile: str
    kind: str
    ordinal: int

    display_text: str
    retrieval_text: str
    search_alias_text: str

    previous_passage_id: str | None
    next_passage_id: str | None

    heading_path: tuple[str, ...]
    section_path: tuple[str, ...]
    source_unit_ids: tuple[str, ...]

    word_count: int
    unit_count: int
    member_count: int

    schema_version: str = (
        PASSAGE_STORE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        """Validate the stable reader contract."""

        for field_name in (
            "passage_id",
            "version_id",
            "author_id",
            "author_name",
            "work_id",
            "work_title",
            "profile",
            "kind",
            "display_text",
            "retrieval_text",
            "search_alias_text",
        ):
            _require_non_empty_text(
                getattr(
                    self,
                    field_name,
                ),
                field_name=field_name,
            )

        if (
            self.schema_version
            != PASSAGE_STORE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported passage store record "
                f"schema version: {self.schema_version}"
            )

        if self.ordinal < 1:
            raise ValueError(
                "ordinal must be at least 1."
            )

        for field_name in (
            "word_count",
            "unit_count",
            "member_count",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        for field_name in (
            "previous_passage_id",
            "next_passage_id",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value is not None:
                _require_non_empty_text(
                    value,
                    field_name=field_name,
                )

        _validate_string_tuple(
            self.heading_path,
            field_name="heading_path",
        )

        _validate_string_tuple(
            self.section_path,
            field_name="section_path",
        )

        _validate_string_tuple(
            self.source_unit_ids,
            field_name="source_unit_ids",
        )

    def snippet(
        self,
        *,
        max_chars: int = 280,
    ) -> str:
        """Return compact display text for API responses."""

        if max_chars < 1:
            raise ValueError(
                "max_chars must be at least 1."
            )

        compact = " ".join(
            self.display_text.split()
        )

        if len(compact) <= max_chars:
            return compact

        if max_chars == 1:
            return "…"

        return (
            compact[
                : max_chars - 1
            ].rstrip()
            + "…"
        )


@dataclass(frozen=True)
class PassageStoreBuildReport:
    """Summary of one completed SQLite passage-store build."""

    output_path: Path
    passage_count: int
    version_count: int
    source_file_count: int
    database_byte_count: int
    source_manifest_sha256: str
    schema_version: str = (
        PASSAGE_STORE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != PASSAGE_STORE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported passage store "
                f"schema version: {self.schema_version}"
            )

        for field_name in (
            "passage_count",
            "version_count",
            "source_file_count",
            "database_byte_count",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        digest = (
            self.source_manifest_sha256
            .lower()
        )

        if (
            len(digest) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(
                "source_manifest_sha256 must "
                "contain 64 hexadecimal characters."
            )


__all__ = [
    "PASSAGE_STORE_SCHEMA_VERSION",
    "PassageStoreBuildReport",
    "PassageStoreRecord",
]
