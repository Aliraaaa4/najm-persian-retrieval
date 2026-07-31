"""Extract exact source ranges for parser golden evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from najm_retrieval.corpus.manifest import load_manifest
from najm_retrieval.corpus.scanner import scan_corpus
from najm_retrieval.parsing.samples import (
    extract_parser_sample,
    write_parser_sample,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an exact OpenITI source-line range "
            "for parser golden evaluation."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/corpus_manifest.yaml"),
    )

    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("data/raw/PER0675AH/data"),
    )

    parser.add_argument(
        "--version-id",
        required=True,
    )

    parser.add_argument(
        "--sample-id",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
        required=True,
    )

    parser.add_argument(
        "--line-start",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--line-end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )

    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    manifest = load_manifest(args.manifest)

    scan = scan_corpus(
        args.corpus_root,
        manifest,
    )

    matching_versions = [
        version
        for version in scan.versions
        if version.version_id == args.version_id
    ]

    if not matching_versions:
        available = "\n".join(
            f"  - {version.version_id}"
            for version in scan.versions
        )

        raise SystemExit(
            f"Unknown version ID: {args.version_id}\n"
            f"Available versions:\n{available}"
        )

    version = matching_versions[0]

    try:
        source_label = version.text_path.relative_to(
            args.corpus_root
        ).as_posix()
    except ValueError:
        source_label = version.text_path.name

    sample = extract_parser_sample(
        sample_id=args.sample_id,
        split=args.split,
        version_id=version.version_id,
        profile=version.profile,
        source_path=version.text_path,
        source_label=source_label,
        line_start=args.line_start,
        line_end=args.line_end,
    )

    write_parser_sample(
        sample,
        args.output,
        overwrite=args.force,
    )

    print(f"Sample ID: {sample.sample_id}")
    print(f"Version: {sample.version_id}")
    print(f"Profile: {sample.profile}")
    print(
        f"Lines: {sample.line_start}-{sample.line_end}"
    )
    print(
        f"Characters: {sample.char_start}-{sample.char_end}"
    )
    print(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())