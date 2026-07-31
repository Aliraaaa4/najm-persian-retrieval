"""Parse the configured OpenITI corpus and write JSON outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from najm_retrieval.parsing.corpus_runner import (
    CorpusRunnerError,
    run_corpus,
)
from najm_retrieval.parsing.serialization import (
    write_corpus_outputs,
)


DEFAULT_MANIFEST_PATH = Path(
    "config/corpus_manifest.yaml"
)

DEFAULT_CORPUS_ROOT = Path(
    "data/raw/PER0675AH/data"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/processed/parser"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Parse all configured OpenITI corpus "
            "versions and write lossless JSON outputs."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "Path to corpus_manifest.yaml "
            f"(default: {DEFAULT_MANIFEST_PATH})"
        ),
    )

    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help=(
            "Root directory containing the "
            "OpenITI corpus checkout "
            f"(default: {DEFAULT_CORPUS_ROOT})"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for parse_report.json "
            "and version files "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the complete corpus parser."""

    arguments = (
        build_argument_parser().parse_args(
            argv
        )
    )

    print(
        "Manifest:",
        arguments.manifest,
    )

    print(
        "Corpus root:",
        arguments.corpus_root,
    )

    print(
        "Output directory:",
        arguments.output_dir,
    )

    try:
        result = run_corpus(
            corpus_root=(
                arguments.corpus_root
            ),
            manifest_path=(
                arguments.manifest
            ),
        )

        outputs = write_corpus_outputs(
            result,
            output_dir=(
                arguments.output_dir
            ),
        )

    except (
        CorpusRunnerError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        return 1

    print()
    print("Corpus parse completed.")
    print(
        "Parsed versions:",
        result.version_count,
    )
    print(
        "Indexable versions:",
        len(
            result.indexable_versions
        ),
    )
    print(
        "Reference versions:",
        len(
            result.reference_versions
        ),
    )
    print(
        "Total blocks:",
        result.total_blocks,
    )
    print(
        "Total body characters:",
        result.total_body_chars,
    )
    print(
        "All lossless:",
        result.all_lossless,
    )
    print(
        "Runtime seconds:",
        f"{result.runtime_seconds:.3f}",
    )
    print(
        "Report:",
        outputs.report_path,
    )
    print(
        "Version JSON files:",
        len(outputs.version_paths),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
