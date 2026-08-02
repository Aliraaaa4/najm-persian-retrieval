"""Load a validated catalog of in-scope and known out-of-scope entities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import unicodedata
from typing import Any

import yaml

from najm_retrieval.corpus.manifest import (
    CorpusManifest,
    load_manifest,
)
from najm_retrieval.retrieval.scope_models import (
    SCOPE_CATALOG_SCHEMA_VERSION,
    ScopeCatalogEntity,
    ScopeEntityKind,
    ScopeMention,
)


class ScopeCatalogError(ValueError):
    """Raised when the scope alias catalog is invalid."""


_PERSIAN_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ؤ": "و",
        "ة": "ه",
        "ۀ": "ه",
        "‌": " ",
        "‍": " ",
        "﻿": " ",
    }
)

_DIACRITIC_PATTERN = re.compile(
    r"[ً-ٰٟۖ-ۭ]"
)

_NON_WORD_PATTERN = re.compile(
    r"[^0-9A-Za-z؀-ۿ]+"
)


def normalize_scope_text(value: str) -> str:
    """Normalize Persian/Arabic query text for conservative alias matching."""

    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).translate(_PERSIAN_TRANSLATION)

    normalized = _DIACRITIC_PATTERN.sub(
        "",
        normalized,
    )

    normalized = _NON_WORD_PATTERN.sub(
        " ",
        normalized,
    )

    return " ".join(
        normalized.casefold().split()
    )


class CorpusScopeCatalog:
    """Validated author/work scope catalog backed by the corpus manifest."""

    def __init__(
        self,
        *,
        manifest: CorpusManifest,
        entities: tuple[ScopeCatalogEntity, ...],
    ) -> None:
        if not entities:
            raise ScopeCatalogError(
                "Scope catalog must contain at least one entity."
            )

        entity_ids = tuple(
            entity.entity_id
            for entity in entities
        )

        if len(entity_ids) != len(set(entity_ids)):
            raise ScopeCatalogError(
                "Scope entity IDs must be unique."
            )

        alias_index: dict[
            str,
            tuple[ScopeCatalogEntity, str],
        ] = {}

        for entity in entities:
            for alias in entity.aliases:
                normalized_alias = normalize_scope_text(
                    alias
                )

                if not normalized_alias:
                    raise ScopeCatalogError(
                        f"Alias normalizes to empty text: {alias!r}"
                    )

                existing = alias_index.get(
                    normalized_alias
                )

                if (
                    existing is not None
                    and existing[0].entity_id
                    != entity.entity_id
                ):
                    raise ScopeCatalogError(
                        "Normalized alias collision between "
                        f"'{existing[0].entity_id}' and "
                        f"'{entity.entity_id}': {alias!r}"
                    )

                alias_index[
                    normalized_alias
                ] = (
                    entity,
                    alias,
                )

        self.manifest = manifest
        self.entities = entities
        self._entities_by_id = {
            entity.entity_id: entity
            for entity in entities
        }
        self._alias_entries = tuple(
            sorted(
                (
                    (
                        normalized_alias,
                        entity,
                        original_alias,
                    )
                    for normalized_alias, (
                        entity,
                        original_alias,
                    ) in alias_index.items()
                ),
                key=lambda item: (
                    -len(item[0].split()),
                    -len(item[0]),
                    item[0],
                ),
            )
        )

    @classmethod
    def from_files(
        cls,
        *,
        manifest_path: str | Path,
        aliases_path: str | Path,
    ) -> "CorpusScopeCatalog":
        """Load the corpus manifest and the curated scope alias catalog."""

        manifest = load_manifest(
            manifest_path
        )

        aliases_file = Path(
            aliases_path
        )

        if not aliases_file.is_file():
            raise ScopeCatalogError(
                f"Scope alias file not found: {aliases_file}"
            )

        try:
            raw = yaml.safe_load(
                aliases_file.read_text(
                    encoding="utf-8-sig"
                )
            )
        except UnicodeDecodeError as error:
            raise ScopeCatalogError(
                "Scope alias file is not valid UTF-8."
            ) from error
        except yaml.YAMLError as error:
            raise ScopeCatalogError(
                "Scope alias file contains invalid YAML."
            ) from error

        entities = _build_entities(
            raw=raw,
            manifest=manifest,
        )

        return cls(
            manifest=manifest,
            entities=entities,
        )

    @property
    def in_corpus_authors(
        self,
    ) -> tuple[ScopeCatalogEntity, ...]:
        return tuple(
            entity
            for entity in self.entities
            if entity.in_corpus
            and entity.kind is ScopeEntityKind.AUTHOR
        )

    @property
    def in_corpus_works(
        self,
    ) -> tuple[ScopeCatalogEntity, ...]:
        return tuple(
            entity
            for entity in self.entities
            if entity.in_corpus
            and entity.kind is ScopeEntityKind.WORK
        )

    def get_entity(
        self,
        entity_id: str,
    ) -> ScopeCatalogEntity:
        try:
            return self._entities_by_id[
                entity_id
            ]
        except KeyError as error:
            raise ScopeCatalogError(
                f"Unknown scope entity ID: {entity_id}"
            ) from error

    def match_query(
        self,
        query_text: str,
    ) -> tuple[ScopeMention, ...]:
        """Return all distinct catalog entities explicitly named in a query."""

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(
                "query_text must not be empty."
            )

        normalized_query = normalize_scope_text(
            query_text
        )

        padded_query = (
            f" {normalized_query} "
        )

        mentions_by_entity: dict[
            str,
            ScopeMention,
        ] = {}

        for (
            normalized_alias,
            entity,
            original_alias,
        ) in self._alias_entries:
            if entity.entity_id in mentions_by_entity:
                continue

            if (
                f" {normalized_alias} "
                not in padded_query
            ):
                continue

            mentions_by_entity[
                entity.entity_id
            ] = ScopeMention(
                entity_id=entity.entity_id,
                kind=entity.kind,
                label_fa=entity.label_fa,
                matched_alias=original_alias,
                in_corpus=entity.in_corpus,
                version_ids=entity.version_ids,
            )

        return tuple(
            sorted(
                mentions_by_entity.values(),
                key=lambda mention: (
                    mention.kind.value,
                    mention.entity_id,
                ),
            )
        )


def _require_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScopeCatalogError(
            f"{label} must be a YAML mapping."
        )

    return value


def _read_entity_specs(
    *,
    mapping: dict[str, Any],
    kind: ScopeEntityKind,
    in_corpus: bool,
    version_ids_by_entity: dict[
        str,
        tuple[str, ...],
    ],
) -> list[ScopeCatalogEntity]:
    entities: list[
        ScopeCatalogEntity
    ] = []

    for entity_id, raw_spec in mapping.items():
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ScopeCatalogError(
                "Every scope entity ID must be a non-empty string."
            )

        spec = _require_mapping(
            raw_spec,
            f"entity '{entity_id}'",
        )

        label_fa = spec.get(
            "label_fa"
        )

        aliases = spec.get(
            "aliases"
        )

        if not isinstance(label_fa, str) or not label_fa.strip():
            raise ScopeCatalogError(
                f"entity '{entity_id}' requires label_fa."
            )

        if (
            not isinstance(aliases, list)
            or not aliases
            or any(
                not isinstance(alias, str)
                or not alias.strip()
                for alias in aliases
            )
        ):
            raise ScopeCatalogError(
                f"entity '{entity_id}' requires non-empty aliases."
            )

        entities.append(
            ScopeCatalogEntity(
                entity_id=entity_id.strip(),
                kind=kind,
                label_fa=label_fa.strip(),
                aliases=tuple(
                    alias.strip()
                    for alias in aliases
                ),
                in_corpus=in_corpus,
                version_ids=(
                    version_ids_by_entity[
                        entity_id
                    ]
                    if in_corpus
                    else ()
                ),
            )
        )

    return entities


def _build_entities(
    *,
    raw: Any,
    manifest: CorpusManifest,
) -> tuple[ScopeCatalogEntity, ...]:
    root = _require_mapping(
        raw,
        "scope alias root",
    )

    schema_version = root.get(
        "schema_version"
    )

    if schema_version != SCOPE_CATALOG_SCHEMA_VERSION:
        raise ScopeCatalogError(
            "Unsupported scope alias schema version: "
            f"{schema_version!r}"
        )

    in_corpus = _require_mapping(
        root.get("in_corpus"),
        "in_corpus",
    )

    out_of_corpus = _require_mapping(
        root.get("known_out_of_corpus"),
        "known_out_of_corpus",
    )

    in_author_specs = _require_mapping(
        in_corpus.get("authors"),
        "in_corpus.authors",
    )

    in_work_specs = _require_mapping(
        in_corpus.get("works"),
        "in_corpus.works",
    )

    out_author_specs = _require_mapping(
        out_of_corpus.get("authors"),
        "known_out_of_corpus.authors",
    )

    out_work_specs = _require_mapping(
        out_of_corpus.get("works"),
        "known_out_of_corpus.works",
    )

    work_version_ids: dict[
        str,
        tuple[str, ...],
    ] = {}

    author_versions: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for work in manifest.works.values():
        indexable_versions: list[str] = []

        if work.include_in_index:
            indexable_versions.append(
                work.canonical_version
            )

        indexable_versions.extend(
            reference.version_id
            for reference in work.reference_versions
            if reference.include_in_index
        )

        work_version_ids[
            work.work_id
        ] = tuple(
            indexable_versions
        )

        author_id = work.work_id.split(
            ".",
            maxsplit=1,
        )[0]

        author_versions[
            author_id
        ].extend(
            indexable_versions
        )

    author_version_ids = {
        author_id: tuple(versions)
        for author_id, versions in author_versions.items()
    }

    if set(in_work_specs) != set(work_version_ids):
        raise ScopeCatalogError(
            "In-corpus work aliases must exactly match manifest works."
        )

    if set(in_author_specs) != set(author_version_ids):
        raise ScopeCatalogError(
            "In-corpus author aliases must exactly match manifest authors."
        )

    entities: list[
        ScopeCatalogEntity
    ] = []

    entities.extend(
        _read_entity_specs(
            mapping=in_author_specs,
            kind=ScopeEntityKind.AUTHOR,
            in_corpus=True,
            version_ids_by_entity=author_version_ids,
        )
    )

    entities.extend(
        _read_entity_specs(
            mapping=in_work_specs,
            kind=ScopeEntityKind.WORK,
            in_corpus=True,
            version_ids_by_entity=work_version_ids,
        )
    )

    entities.extend(
        _read_entity_specs(
            mapping=out_author_specs,
            kind=ScopeEntityKind.AUTHOR,
            in_corpus=False,
            version_ids_by_entity={},
        )
    )

    entities.extend(
        _read_entity_specs(
            mapping=out_work_specs,
            kind=ScopeEntityKind.WORK,
            in_corpus=False,
            version_ids_by_entity={},
        )
    )

    return tuple(
        sorted(
            entities,
            key=lambda entity: (
                not entity.in_corpus,
                entity.kind.value,
                entity.entity_id,
            ),
        )
    )


__all__ = [
    "CorpusScopeCatalog",
    "ScopeCatalogError",
    "normalize_scope_text",
]
