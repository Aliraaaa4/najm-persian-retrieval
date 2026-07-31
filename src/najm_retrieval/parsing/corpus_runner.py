"""Manifest-driven execution of all configured corpus parsers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from najm_retrieval.corpus.manifest import (
    load_manifest,
)
from najm_retrieval.corpus.scanner import (
    CorpusScanResult,
    CorpusVersionFiles,
    scan_corpus,
)
from najm_retrieval.parsing.core import (
    OpenITISource,
    load_openiti_source,
)
from najm_retrieval.parsing.dispatcher import (
    parse_source,
)
from najm_retrieval.parsing.models import (
    ParseMetrics,
    ParsedDocument,
)


class CorpusRunnerError(RuntimeError):
    """Raised when a complete corpus parse cannot continue."""


@dataclass(frozen=True)
class ParsedCorpusVersion:
    """Parser result and configuration for one corpus version."""

    author_id: str
    work_id: str
    version_id: str

    profile: str
    include_in_index: bool
    is_canonical: bool

    source_path: Path
    source: OpenITISource
    document: ParsedDocument
    metrics: ParseMetrics

    @property
    def passes_lossless_gate(self) -> bool:
        """Return whether this version passed strict validation."""

        return self.metrics.passes_lossless_gate


@dataclass(frozen=True)
class CorpusParseResult:
    """Complete result of parsing every configured version."""

    corpus_root: Path
    versions: tuple[ParsedCorpusVersion, ...]
    runtime_seconds: float

    @property
    def version_count(self) -> int:
        """Return the number of parsed versions."""

        return len(self.versions)

    @property
    def all_lossless(self) -> bool:
        """Return whether every parsed version passed its gate."""

        return all(
            version.passes_lossless_gate
            for version in self.versions
        )

    @property
    def indexable_versions(
        self,
    ) -> tuple[ParsedCorpusVersion, ...]:
        """Return versions configured for semantic indexing."""

        return tuple(
            version
            for version in self.versions
            if version.include_in_index
        )

    @property
    def canonical_versions(
        self,
    ) -> tuple[ParsedCorpusVersion, ...]:
        """Return canonical versions only."""

        return tuple(
            version
            for version in self.versions
            if version.is_canonical
        )

    @property
    def reference_versions(
        self,
    ) -> tuple[ParsedCorpusVersion, ...]:
        """Return non-canonical reference versions."""

        return tuple(
            version
            for version in self.versions
            if not version.is_canonical
        )

    @property
    def total_body_chars(self) -> int:
        """Return total parsed body character count."""

        return sum(
            version.metrics.total_body_chars
            for version in self.versions
        )

    @property
    def total_blocks(self) -> int:
        """Return total block count across all versions."""

        return sum(
            len(version.document.blocks)
            for version in self.versions
        )


def run_corpus(
    *,
    corpus_root: str | Path,
    manifest_path: str | Path,
) -> CorpusParseResult:
    """Load configuration, scan files, and parse the corpus."""

    manifest = load_manifest(
        manifest_path
    )

    scan = scan_corpus(
        corpus_root,
        manifest,
    )

    return parse_scanned_corpus(
        scan
    )


def parse_scanned_corpus(
    scan: CorpusScanResult,
) -> CorpusParseResult:
    """Parse every validated version in one scan result."""

    _require_clean_scan(
        scan
    )

    started_at = perf_counter()

    parsed_versions: list[
        ParsedCorpusVersion
    ] = []

    seen_version_ids: set[str] = set()

    for version in scan.versions:
        if version.version_id in seen_version_ids:
            raise CorpusRunnerError(
                "Duplicate version ID in corpus scan: "
                f"{version.version_id}"
            )

        seen_version_ids.add(
            version.version_id
        )

        parsed_versions.append(
            _parse_version(
                version
            )
        )

    result = CorpusParseResult(
        corpus_root=scan.corpus_root,
        versions=tuple(parsed_versions),
        runtime_seconds=(
            perf_counter() - started_at
        ),
    )

    if not result.versions:
        raise CorpusRunnerError(
            "Corpus scan contains no configured versions."
        )

    if not result.all_lossless:
        failed = [
            version.version_id
            for version in result.versions
            if not version.passes_lossless_gate
        ]

        raise CorpusRunnerError(
            "One or more corpus versions failed "
            "the lossless gate: "
            + ", ".join(failed)
        )

    return result


def _parse_version(
    version: CorpusVersionFiles,
) -> ParsedCorpusVersion:
    """Load and parse one configured text version."""

    try:
        source = load_openiti_source(
            version.text_path
        )

        if source.version_id != version.version_id:
            raise ValueError(
                "Loaded source version ID "
                f"{source.version_id!r} does not match "
                f"configured ID {version.version_id!r}."
            )

        document, metrics = parse_source(
            source=source,
            profile=version.profile,
        )

    except Exception as error:
        raise CorpusRunnerError(
            "Failed to parse version "
            f"{version.version_id!r} "
            f"with profile {version.profile!r}: "
            f"{error}"
        ) from error

    if (
        document.reconstruct_body()
        != source.body_text
    ):
        raise CorpusRunnerError(
            "Parsed document does not reconstruct "
            "the source body for version "
            f"{version.version_id!r}."
        )

    return ParsedCorpusVersion(
        author_id=version.author_id,
        work_id=version.work_id,
        version_id=version.version_id,
        profile=version.profile,
        include_in_index=(
            version.include_in_index
        ),
        is_canonical=version.is_canonical,
        source_path=version.text_path,
        source=source,
        document=document,
        metrics=metrics,
    )


def _require_clean_scan(
    scan: CorpusScanResult,
) -> None:
    """Reject missing, invalid, or unexpected corpus files."""

    if scan.ok:
        return

    problems: list[str] = []

    for issue in scan.issues:
        location = (
            f" [{issue.path}]"
            if issue.path is not None
            else ""
        )

        problems.append(
            f"{issue.code}: "
            f"{issue.message}"
            f"{location}"
        )

    for path in scan.unexpected_versions:
        problems.append(
            f"Unexpected version: {path}"
        )

    details = "\n".join(
        f"- {problem}"
        for problem in problems
    )

    raise CorpusRunnerError(
        "Corpus scan failed. Parsing was not started."
        + (
            "\n" + details
            if details
            else ""
        )
    )
