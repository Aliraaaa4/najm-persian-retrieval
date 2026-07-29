"""Audit configured OpenITI corpus files and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from najm_retrieval.corpus.manifest import (
    CorpusManifest,
    ManifestError,
    load_manifest,
)
from najm_retrieval.corpus.scanner import (
    CorpusScanResult,
    scan_corpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "config" / "corpus_manifest.yaml"
)

DEFAULT_CORPUS_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "PER0675AH" / "data"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "corpus_audit.json"
)


def _path_for_report(path: Path) -> str:
    """Return a portable project-relative path when possible."""

    resolved_path = path.resolve()
    resolved_root = PROJECT_ROOT.resolve()

    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def build_audit_report(
    manifest: CorpusManifest,
    scan_result: CorpusScanResult,
) -> dict[str, object]:
    """Create a JSON-serializable corpus audit report."""

    authors = sorted(
        {
            version.author_id
            for version in scan_result.versions
        }
    )

    canonical_versions = [
        version
        for version in scan_result.versions
        if version.is_canonical
    ]

    reference_versions = [
        version
        for version in scan_result.versions
        if not version.is_canonical
    ]

    version_records = []

    for version in sorted(
        scan_result.versions,
        key=lambda item: item.version_id,
    ):
        version_records.append(
            {
                "author_id": version.author_id,
                "work_id": version.work_id,
                "version_id": version.version_id,
                "profile": version.profile,
                "is_canonical": version.is_canonical,
                "include_in_index": version.include_in_index,
                "paths": {
                    "text": _path_for_report(version.text_path),
                    "author_yml": _path_for_report(
                        version.author_yml_path
                    ),
                    "work_yml": _path_for_report(
                        version.work_yml_path
                    ),
                    "version_yml": _path_for_report(
                        version.version_yml_path
                    ),
                },
            }
        )

    issues = [
        {
            "code": issue.code,
            "message": issue.message,
            "path": (
                _path_for_report(issue.path)
                if issue.path is not None
                else None
            ),
        }
        for issue in scan_result.issues
    ]

    unexpected_versions = [
        _path_for_report(path)
        for path in scan_result.unexpected_versions
    ]

    return {
        "dataset": {
            "name": manifest.dataset.name,
            "repository": manifest.dataset.repository,
            "commit": manifest.dataset.commit,
            "corpus_root": _path_for_report(
                scan_result.corpus_root
            ),
        },
        "summary": {
            "authors": len(authors),
            "works": len(manifest.works),
            "configured_versions": len(scan_result.versions),
            "canonical_versions": len(canonical_versions),
            "reference_versions": len(reference_versions),
            "indexable_versions": len(
                scan_result.indexable_versions
            ),
            "issues": len(scan_result.issues),
            "unexpected_versions": len(
                scan_result.unexpected_versions
            ),
            "scan_ok": scan_result.ok,
        },
        "authors": authors,
        "versions": version_records,
        "issues": issues,
        "unexpected_versions": unexpected_versions,
    }


def print_summary(
    manifest: CorpusManifest,
    scan_result: CorpusScanResult,
) -> None:
    """Print a human-readable audit summary."""

    authors = {
        version.author_id
        for version in scan_result.versions
    }

    reference_count = sum(
        not version.is_canonical
        for version in scan_result.versions
    )

    print(f"Dataset: {manifest.dataset.name}")
    print(f"Dataset commit: {manifest.dataset.commit}")
    print()
    print(f"Authors: {len(authors)}")
    print(f"Works: {len(manifest.works)}")
    print(f"Configured versions: {len(scan_result.versions)}")
    print(
        f"Canonical versions: "
        f"{len(scan_result.canonical_versions)}"
    )
    print(f"Reference versions: {reference_count}")
    print(
        f"Indexable versions: "
        f"{len(scan_result.indexable_versions)}"
    )
    print(f"Issues: {len(scan_result.issues)}")
    print(
        f"Unexpected versions: "
        f"{len(scan_result.unexpected_versions)}"
    )
    print()

    status = "OK" if scan_result.ok else "FAILED"
    print(f"Status: {status}")

    if scan_result.issues:
        print()
        print("Issues:")

        for issue in scan_result.issues:
            print(f"- [{issue.code}] {issue.message}")

    if scan_result.unexpected_versions:
        print()
        print("Unexpected versions:")

        for path in scan_result.unexpected_versions:
            print(f"- {path}")


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate configured OpenITI corpus files "
            "and generate an audit report."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the corpus manifest YAML file.",
    )

    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="Path to the OpenITI corpus data directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the generated JSON audit report.",
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the audit result without writing JSON.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus audit command."""

    arguments = parse_arguments(argv)

    try:
        manifest = load_manifest(arguments.manifest)
        scan_result = scan_corpus(
            arguments.corpus_root,
            manifest,
        )
    except (
        ManifestError,
        FileNotFoundError,
        OSError,
    ) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2

    print_summary(manifest, scan_result)

    if not arguments.no_write:
        report = build_audit_report(
            manifest,
            scan_result,
        )

        arguments.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        arguments.output.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            "Report written to: "
            f"{arguments.output}"
        )

    return 0 if scan_result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())