"""Data models for text normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TextLanguage(str, Enum):
    """Detected language/script category."""

    PERSIAN = "persian"
    ARABIC = "arabic"
    MIXED = "mixed"
    ENGLISH = "english"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormalizationChange:
    """One applied normalization operation."""

    name: str
    description: str


@dataclass(frozen=True)
class NormalizedText:
    """Result of normalizing one text unit.

    original_text:
        Exact input text before normalization.

    normalized_text:
        Text used for retrieval/search.

    language:
        Dominant language/script detected.

    changes:
        List of transformations applied.
    """

    original_text: str

    normalized_text: str

    language: TextLanguage

    changes: tuple[
        NormalizationChange,
        ...
    ] = field(
        default_factory=tuple
    )

    @property
    def changed(self) -> bool:
        """Return whether any transformation happened."""

        return (
            self.original_text
            != self.normalized_text
        )