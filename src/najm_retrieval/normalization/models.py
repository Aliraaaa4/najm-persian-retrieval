"""Data models for multi-view text normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


NORMALIZATION_VERSION = "1.0.0"


class NormalizationView(str, Enum):
    """One normalized representation of a text."""

    DISPLAY = "display"
    RETRIEVAL = "retrieval"
    SEARCH_ALIAS = "search_alias"


class ScriptProfile(str, Enum):
    """Character-level Arabic-script profile.

    This is diagnostic metadata, not language identification.
    A Persian text may contain Arabic variants, quotations, names,
    or OCR artifacts and therefore receive a mixed profile.
    """

    PERSIAN_SIGNALS_ONLY = (
        "persian_signals_only"
    )

    ARABIC_VARIANTS_ONLY = (
        "arabic_variants_only"
    )

    MIXED_SIGNALS = (
        "mixed_signals"
    )

    SHARED_OR_OTHER_ONLY = (
        "shared_or_other_only"
    )


@dataclass(frozen=True)
class NormalizationChange:
    """One normalization operation applied to a text view.

    count records the number of input characters, sequences, or
    whitespace runs that were actually changed by the operation.
    """

    view: NormalizationView
    operation: str
    count: int

    def __post_init__(self) -> None:
        """Validate a recorded normalization change."""

        if not isinstance(
            self.view,
            NormalizationView,
        ):
            raise TypeError(
                "view must be a NormalizationView."
            )

        if (
            not isinstance(self.operation, str)
            or not self.operation.strip()
        ):
            raise ValueError(
                "operation must be a non-empty string."
            )

        if (
            not isinstance(self.count, int)
            or isinstance(self.count, bool)
            or self.count <= 0
        ):
            raise ValueError(
                "count must be a positive integer."
            )


@dataclass(frozen=True)
class NormalizationResult:
    """All normalized representations of one input text."""

    original_text: str
    display_text: str
    retrieval_text: str
    search_alias_text: str
    script_profile: ScriptProfile

    alias_fold_alef_madda: bool = False

    changes: tuple[
        NormalizationChange,
        ...,
    ] = field(
        default_factory=tuple
    )

    normalization_version: str = (
        NORMALIZATION_VERSION
    )

    def __post_init__(self) -> None:
        """Validate normalization-result fields."""

        for field_name in (
            "original_text",
            "display_text",
            "retrieval_text",
            "search_alias_text",
        ):
            value = getattr(
                self,
                field_name,
            )

            if not isinstance(value, str):
                raise TypeError(
                    f"{field_name} must be a string."
                )

        if not isinstance(
            self.script_profile,
            ScriptProfile,
        ):
            raise TypeError(
                "script_profile must be a "
                "ScriptProfile."
            )

        if not isinstance(
            self.alias_fold_alef_madda,
            bool,
        ):
            raise TypeError(
                "alias_fold_alef_madda must be "
                "a boolean."
            )

        if (
            not isinstance(
                self.normalization_version,
                str,
            )
            or not self.normalization_version
        ):
            raise ValueError(
                "normalization_version must be "
                "a non-empty string."
            )

        if not isinstance(
            self.changes,
            tuple,
        ):
            raise TypeError(
                "changes must be a tuple."
            )

        for change in self.changes:
            if not isinstance(
                change,
                NormalizationChange,
            ):
                raise TypeError(
                    "Every change must be a "
                    "NormalizationChange."
                )

    @property
    def changed(self) -> bool:
        """Return whether any output differs from the input."""

        return bool(
            self.changed_views
        )

    @property
    def changed_views(
        self,
    ) -> tuple[NormalizationView, ...]:
        """Return views whose text differs from the input."""

        changed: list[
            NormalizationView
        ] = []

        if (
            self.display_text
            != self.original_text
        ):
            changed.append(
                NormalizationView.DISPLAY
            )

        if (
            self.retrieval_text
            != self.original_text
        ):
            changed.append(
                NormalizationView.RETRIEVAL
            )

        if (
            self.search_alias_text
            != self.original_text
        ):
            changed.append(
                NormalizationView.SEARCH_ALIAS
            )

        return tuple(changed)

    def changes_for(
        self,
        view: NormalizationView,
    ) -> tuple[NormalizationChange, ...]:
        """Return recorded changes for one output view."""

        if not isinstance(
            view,
            NormalizationView,
        ):
            raise TypeError(
                "view must be a NormalizationView."
            )

        return tuple(
            change
            for change in self.changes
            if change.view == view
        )
