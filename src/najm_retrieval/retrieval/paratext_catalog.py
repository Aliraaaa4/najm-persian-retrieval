"""Load version-specific structural paratext zones."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import yaml

from najm_retrieval.retrieval.paratext_models import (
    PARATEXT_CATALOG_SCHEMA_VERSION,
    ContentRole,
    ParatextZone,
    PassageRoleEvidence,
)


_PASSAGE_ID_PATTERN = re.compile(
    r"^(?P<version_id>.+):passage_(?P<ordinal>[0-9]+)$"
)


class ParatextCatalogError(
    ValueError
):
    """Raised when structural paratext configuration is invalid."""


class ParatextCatalog:
    """Validated structural role zones keyed by version ID."""

    def __init__(
        self,
        *,
        zones_by_version: Mapping[
            str,
            tuple[ParatextZone, ...],
        ],
        expected_passage_counts: Mapping[
            str,
            int,
        ],
    ) -> None:
        if not zones_by_version:
            raise ParatextCatalogError(
                "At least one configured version is required."
            )

        if (
            set(zones_by_version)
            != set(expected_passage_counts)
        ):
            raise ParatextCatalogError(
                "Configured versions and expected passage counts must match."
            )

        self._zones_by_version = dict(
            zones_by_version
        )
        self._expected_passage_counts = dict(
            expected_passage_counts
        )

        for version_id, zones in self._zones_by_version.items():
            self._validate_version_zones(
                version_id=version_id,
                zones=zones,
                expected_passage_count=(
                    self._expected_passage_counts[
                        version_id
                    ]
                ),
            )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "ParatextCatalog":
        """Load a structural paratext catalog from YAML."""

        config_path = Path(
            path
        )

        if not config_path.is_file():
            raise ParatextCatalogError(
                f"Paratext catalog not found: {config_path}"
            )

        try:
            raw = yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8-sig"
                )
            )
        except UnicodeDecodeError as error:
            raise ParatextCatalogError(
                "Paratext catalog is not valid UTF-8."
            ) from error
        except yaml.YAMLError as error:
            raise ParatextCatalogError(
                "Paratext catalog contains invalid YAML."
            ) from error

        root = _require_mapping(
            raw,
            "paratext catalog root",
        )

        schema_version = root.get(
            "schema_version"
        )

        if (
            schema_version
            != PARATEXT_CATALOG_SCHEMA_VERSION
        ):
            raise ParatextCatalogError(
                "Unsupported paratext catalog schema version: "
                f"{schema_version!r}"
            )

        raw_versions = _require_mapping(
            root.get(
                "versions"
            ),
            "versions",
        )

        zones_by_version: dict[
            str,
            tuple[ParatextZone, ...],
        ] = {}

        expected_counts: dict[
            str,
            int,
        ] = {}

        for (
            version_id,
            raw_version,
        ) in raw_versions.items():
            if (
                not isinstance(
                    version_id,
                    str,
                )
                or not version_id.strip()
            ):
                raise ParatextCatalogError(
                    "Every configured version ID must be non-empty."
                )

            version_spec = _require_mapping(
                raw_version,
                f"version '{version_id}'",
            )

            expected_passage_count = (
                version_spec.get(
                    "expected_passage_count"
                )
            )

            if (
                not isinstance(
                    expected_passage_count,
                    int,
                )
                or isinstance(
                    expected_passage_count,
                    bool,
                )
                or expected_passage_count < 1
            ):
                raise ParatextCatalogError(
                    "expected_passage_count must be a positive integer "
                    f"for version '{version_id}'."
                )

            raw_zones = version_spec.get(
                "zones"
            )

            if (
                not isinstance(
                    raw_zones,
                    list,
                )
                or not raw_zones
            ):
                raise ParatextCatalogError(
                    f"Version '{version_id}' requires non-empty zones."
                )

            zones: list[
                ParatextZone
            ] = []

            for zone_index, raw_zone in enumerate(
                raw_zones,
                start=1,
            ):
                zone_spec = _require_mapping(
                    raw_zone,
                    (
                        f"version '{version_id}' "
                        f"zone {zone_index}"
                    ),
                )

                try:
                    role = ContentRole(
                        zone_spec.get(
                            "role"
                        )
                    )
                except ValueError as error:
                    raise ParatextCatalogError(
                        "Unsupported content role in "
                        f"version '{version_id}' zone {zone_index}."
                    ) from error

                if role is ContentRole.UNKNOWN:
                    raise ParatextCatalogError(
                        "Configured zones cannot use the unknown role."
                    )

                reason = zone_spec.get(
                    "reason"
                )

                if not isinstance(
                    reason,
                    str,
                ):
                    raise ParatextCatalogError(
                        "Every configured zone requires a string reason."
                    )

                try:
                    zone = ParatextZone(
                        version_id=version_id,
                        start_ordinal=zone_spec.get(
                            "start_ordinal"
                        ),
                        end_ordinal=zone_spec.get(
                            "end_ordinal"
                        ),
                        role=role,
                        reason=reason,
                    )
                except ValueError as error:
                    raise ParatextCatalogError(
                        "Invalid zone in "
                        f"version '{version_id}' zone {zone_index}: "
                        f"{error}"
                    ) from error

                zones.append(
                    zone
                )

            zones_by_version[
                version_id
            ] = tuple(
                zones
            )

            expected_counts[
                version_id
            ] = expected_passage_count

        return cls(
            zones_by_version=zones_by_version,
            expected_passage_counts=expected_counts,
        )

    @property
    def configured_version_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._zones_by_version
            )
        )

    def expected_passage_count(
        self,
        version_id: str,
    ) -> int | None:
        return self._expected_passage_counts.get(
            version_id
        )

    def classify(
        self,
        *,
        passage_id: str,
        version_id: str,
    ) -> PassageRoleEvidence:
        """Classify a passage using its validated structural ordinal."""

        ordinal = parse_passage_ordinal(
            passage_id=passage_id,
            version_id=version_id,
        )

        zones = self._zones_by_version.get(
            version_id
        )

        if zones is None:
            return PassageRoleEvidence(
                passage_id=passage_id,
                version_id=version_id,
                ordinal=ordinal,
                role=ContentRole.UNKNOWN,
                configured=False,
                reason=None,
            )

        for zone in zones:
            if zone.contains(
                ordinal
            ):
                return PassageRoleEvidence(
                    passage_id=passage_id,
                    version_id=version_id,
                    ordinal=ordinal,
                    role=zone.role,
                    configured=True,
                    reason=zone.reason,
                )

        raise ParatextCatalogError(
            "Configured version does not cover passage ordinal "
            f"{ordinal}: {version_id}"
        )

    @staticmethod
    def _validate_version_zones(
        *,
        version_id: str,
        zones: tuple[
            ParatextZone,
            ...,
        ],
        expected_passage_count: int,
    ) -> None:
        if not zones:
            raise ParatextCatalogError(
                f"Version '{version_id}' has no zones."
            )

        ordered = tuple(
            sorted(
                zones,
                key=lambda zone: (
                    zone.start_ordinal,
                    zone.end_ordinal,
                ),
            )
        )

        expected_start = 1

        for zone in ordered:
            if zone.version_id != version_id:
                raise ParatextCatalogError(
                    "Zone version_id does not match its catalog key."
                )

            if (
                zone.start_ordinal
                != expected_start
            ):
                raise ParatextCatalogError(
                    "Zones must be contiguous and begin at ordinal 1 "
                    f"for version '{version_id}'."
                )

            expected_start = (
                zone.end_ordinal
                + 1
            )

        if (
            ordered[-1].end_ordinal
            != expected_passage_count
        ):
            raise ParatextCatalogError(
                "Zones must end at expected_passage_count "
                f"for version '{version_id}'."
            )


def parse_passage_ordinal(
    *,
    passage_id: str,
    version_id: str,
) -> int:
    """Parse and validate the ordinal encoded in a passage ID."""

    if not passage_id.strip():
        raise ValueError(
            "passage_id must not be empty."
        )

    if not version_id.strip():
        raise ValueError(
            "version_id must not be empty."
        )

    match = _PASSAGE_ID_PATTERN.fullmatch(
        passage_id
    )

    if match is None:
        raise ValueError(
            "passage_id must end with ':passage_<ordinal>'."
        )

    encoded_version_id = match.group(
        "version_id"
    )

    if encoded_version_id != version_id:
        raise ValueError(
            "passage_id version prefix does not match version_id."
        )

    ordinal = int(
        match.group(
            "ordinal"
        )
    )

    if ordinal < 1:
        raise ValueError(
            "Passage ordinal must be at least 1."
        )

    return ordinal


def _require_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        raise ParatextCatalogError(
            f"{label} must be a YAML mapping."
        )

    return value


__all__ = [
    "ParatextCatalog",
    "ParatextCatalogError",
    "parse_passage_ordinal",
]
