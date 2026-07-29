"""Discover and validate OpenITI corpus files using the project manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from najm_retrieval.corpus.manifest import CorpusManifest


OPENITI_MAGIC_LINE = "######OpenITI#"

# Examples:
# 0672JalalDinRumi.Mathnawi.PDL00048-per1
# 0670IbnAscadHanati.Masalik.Kraken220107010708-per1
VERSION_TEXT_PATTERN = re.compile(r"-[a-z]{3}\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ScanIssue:
    """A problem found while validating corpus files."""

    code: str
    message: str
    path: Path | None = None


@dataclass(frozen=True)
class CorpusVersionFiles:
    """Paths and configuration associated with one text version."""

    author_id: str
    work_id: str
    version_id: str

    text_path: Path
    author_yml_path: Path
    work_yml_path: Path
    version_yml_path: Path

    profile: str
    include_in_index: bool
    is_canonical: bool

    @property
    def required_paths(self) -> tuple[Path, ...]:
        """Return all files required for this version."""

        return (
            self.text_path,
            self.author_yml_path,
            self.work_yml_path,
            self.version_yml_path,
        )


@dataclass(frozen=True)
class CorpusScanResult:
    """Result of comparing the manifest with files on disk."""

    corpus_root: Path
    versions: tuple[CorpusVersionFiles, ...]
    issues: tuple[ScanIssue, ...]
    unexpected_versions: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        """Return True when no missing, invalid, or unexpected files exist."""

        return not self.issues and not self.unexpected_versions

    @property
    def indexable_versions(self) -> tuple[CorpusVersionFiles, ...]:
        """Return versions that should be included in the search index."""

        return tuple(
            version
            for version in self.versions
            if version.include_in_index
        )

    @property
    def canonical_versions(self) -> tuple[CorpusVersionFiles, ...]:
        """Return canonical versions only."""

        return tuple(
            version
            for version in self.versions
            if version.is_canonical
        )


def _build_version_record(
    corpus_root: Path,
    *,
    work_id: str,
    version_id: str,
    profile: str,
    include_in_index: bool,
    is_canonical: bool,
) -> CorpusVersionFiles:
    """Build all expected paths for one configured version."""

    author_id = work_id.split(".", maxsplit=1)[0]

    author_dir = corpus_root / author_id
    work_dir = author_dir / work_id

    return CorpusVersionFiles(
        author_id=author_id,
        work_id=work_id,
        version_id=version_id,
        text_path=work_dir / version_id,
        author_yml_path=author_dir / f"{author_id}.yml",
        work_yml_path=work_dir / f"{work_id}.yml",
        version_yml_path=work_dir / f"{version_id}.yml",
        profile=profile,
        include_in_index=include_in_index,
        is_canonical=is_canonical,
    )


def _read_first_line(path: Path) -> str:
    """Read the first line while accepting UTF-8 files with or without BOM."""

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:
        return file.readline().strip()


def _discover_version_texts(corpus_root: Path) -> tuple[Path, ...]:
    """Discover OpenITI-style version text filenames under the corpus root."""

    discovered = [
        path
        for path in corpus_root.rglob("*")
        if path.is_file() and VERSION_TEXT_PATTERN.search(path.name)
    ]

    return tuple(sorted(discovered, key=lambda path: str(path)))


def scan_corpus(
    corpus_root: str | Path,
    manifest: CorpusManifest,
) -> CorpusScanResult:
    """Compare configured corpus versions with actual files on disk."""

    root = Path(corpus_root)

    if not root.is_dir():
        raise FileNotFoundError(f"Corpus root not found: {root}")

    versions: list[CorpusVersionFiles] = []

    for work in manifest.works.values():
        versions.append(
            _build_version_record(
                root,
                work_id=work.work_id,
                version_id=work.canonical_version,
                profile=work.profile,
                include_in_index=work.include_in_index,
                is_canonical=True,
            )
        )

        for reference in work.reference_versions:
            versions.append(
                _build_version_record(
                    root,
                    work_id=work.work_id,
                    version_id=reference.version_id,
                    profile=reference.profile,
                    include_in_index=reference.include_in_index,
                    is_canonical=False,
                )
            )

    issues: list[ScanIssue] = []
    issue_keys: set[tuple[str, str]] = set()

    def add_issue(
        code: str,
        message: str,
        path: Path | None = None,
    ) -> None:
        """Add an issue once, even when shared files are checked repeatedly."""

        key = (code, str(path) if path is not None else "")

        if key in issue_keys:
            return

        issue_keys.add(key)
        issues.append(
            ScanIssue(
                code=code,
                message=message,
                path=path,
            )
        )

    for version in versions:
        required_files = (
            (
                "missing_text",
                version.text_path,
                "Version text file",
            ),
            (
                "missing_author_yml",
                version.author_yml_path,
                "Author YAML file",
            ),
            (
                "missing_work_yml",
                version.work_yml_path,
                "Work YAML file",
            ),
            (
                "missing_version_yml",
                version.version_yml_path,
                "Version YAML file",
            ),
        )

        for code, path, label in required_files:
            if not path.is_file():
                add_issue(
                    code,
                    f"{label} is missing for version "
                    f"'{version.version_id}': {path}",
                    path,
                )

        if version.text_path.is_file():
            try:
                first_line = _read_first_line(version.text_path)
            except OSError as exc:
                add_issue(
                    "text_read_error",
                    f"Could not read version "
                    f"'{version.version_id}': {exc}",
                    version.text_path,
                )
            else:
                if first_line != OPENITI_MAGIC_LINE:
                    add_issue(
                        "invalid_magic_line",
                        f"Version '{version.version_id}' does not start "
                        f"with '{OPENITI_MAGIC_LINE}'.",
                        version.text_path,
                    )

    expected_text_paths = {
        version.text_path.resolve()
        for version in versions
    }

    discovered_text_paths = _discover_version_texts(root)

    unexpected_versions = tuple(
        path
        for path in discovered_text_paths
        if path.resolve() not in expected_text_paths
    )

    sorted_issues = tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.code,
                str(issue.path) if issue.path else "",
            ),
        )
    )

    return CorpusScanResult(
        corpus_root=root,
        versions=tuple(versions),
        issues=sorted_issues,
        unexpected_versions=unexpected_versions,
    )