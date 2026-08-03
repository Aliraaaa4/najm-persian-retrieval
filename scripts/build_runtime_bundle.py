from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MODEL_DIRECTORY = "intfloat__multilingual-e5-small"
DENSE_REQUIRED_FILES = (
    "artifact_manifest.json",
    "embeddings.npy",
    "metadata.json",
    "passages.jsonl",
    "pilot_evaluation.json",
)
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


class BundleBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BundleBuildError(f"{label} not found: {resolved}")
    return resolved


def discover_dense_root(project_root: Path) -> Path:
    matches = sorted(
        p.resolve()
        for p in (project_root / "artifacts" / "indexes").glob(
            f"corpus-*/dense/{MODEL_DIRECTORY}"
        )
        if p.is_dir()
    )
    if len(matches) != 1:
        details = "\n".join(f"  - {p}" for p in matches) or "  (none)"
        raise BundleBuildError(
            "Expected exactly one dense artifact directory. Found:\n"
            + details
            + "\nPass --dense-root explicitly when needed."
        )
    return matches[0]


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def write_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: PurePosixPath,
) -> None:
    with source.open("rb") as input_handle:
        with archive.open(
            zip_info(archive_name.as_posix()),
            mode="w",
            force_zip64=True,
        ) as output_handle:
            shutil.copyfileobj(
                input_handle,
                output_handle,
                length=1024 * 1024,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cross-platform NAJM runtime bundle with POSIX ZIP paths."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--lexical-index",
        type=Path,
        default=Path(tempfile.gettempdir())
        / "najm_real_lexical_index.sqlite3",
    )
    parser.add_argument(
        "--passage-store",
        type=Path,
        default=Path(tempfile.gettempdir())
        / "najm_real_passage_store.sqlite3",
    )
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    if not (project_root / "pyproject.toml").is_file():
        raise BundleBuildError(
            f"Not a repository root: {project_root}"
        )

    lexical = require_file(args.lexical_index, "Lexical index")
    passage_store = require_file(args.passage_store, "Passage store")
    dense_root = (
        args.dense_root.expanduser().resolve()
        if args.dense_root is not None
        else discover_dense_root(project_root)
    )

    if not dense_root.is_dir():
        raise BundleBuildError(
            f"Dense artifact directory not found: {dense_root}"
        )

    missing = [
        name
        for name in DENSE_REQUIRED_FILES
        if not (dense_root / name).is_file()
    ]
    if missing:
        raise BundleBuildError(
            "Dense artifact directory is incomplete: "
            + ", ".join(missing)
        )

    metadata = json.loads(
        (dense_root / "metadata.json").read_text(encoding="utf-8")
    )
    expected_model = "intfloat/multilingual-e5-small"
    if metadata.get("model_name") != expected_model:
        raise BundleBuildError(
            f"Unexpected model_name: {metadata.get('model_name')!r}"
        )

    corpus_id = dense_root.parent.parent.name
    if not corpus_id.startswith("corpus-"):
        raise BundleBuildError(
            f"Invalid corpus artifact id: {corpus_id}"
        )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else project_root / "najm-runtime-artifacts-v1.zip"
    )
    checksum_path = output.with_suffix(output.suffix + ".sha256")

    if not args.force and (output.exists() or checksum_path.exists()):
        raise BundleBuildError(
            "Output already exists. Use --force to replace it."
        )

    runtime_root = PurePosixPath("artifacts") / "runtime" / corpus_id
    dense_archive_root = (
        runtime_root / "dense" / MODEL_DIRECTORY
    )
    entries = [
        (lexical, runtime_root / "lexical.sqlite3"),
        (passage_store, runtime_root / "passage_store.sqlite3"),
    ]
    entries.extend(
        (dense_root / name, dense_archive_root / name)
        for name in DENSE_REQUIRED_FILES
    )

    manifest_files = []
    total_bytes = 0
    for source, archive_name in entries:
        size = source.stat().st_size
        total_bytes += size
        manifest_files.append(
            {
                "path": archive_name.as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(source),
            }
        )

    manifest = {
        "schema_version": "1.0.0",
        "bundle_name": "najm-runtime-artifacts-v1",
        "corpus_artifact_id": corpus_id,
        "dense_model_name": metadata.get("model_name"),
        "passage_count": metadata.get("passage_count"),
        "embedding_dimension": metadata.get("embedding_dimension"),
        "file_count": len(entries),
        "total_uncompressed_bytes": total_bytes,
        "files": manifest_files,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    temp_output.unlink(missing_ok=True)

    try:
        with zipfile.ZipFile(
            temp_output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            manifest_name = (
                PurePosixPath("artifacts")
                / "runtime"
                / "bundle_manifest.json"
            ).as_posix()
            archive.writestr(
                zip_info(manifest_name),
                manifest_bytes,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
            for source, archive_name in entries:
                write_file(archive, source, archive_name)

        with zipfile.ZipFile(temp_output, mode="r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise BundleBuildError(
                    f"ZIP integrity failure: {bad_member}"
                )
            names = archive.namelist()
            if any("\\" in name for name in names):
                raise BundleBuildError(
                    "Archive contains non-portable backslash paths."
                )
            if len(names) != len(entries) + 1:
                raise BundleBuildError(
                    f"Unexpected entry count: {len(names)}"
                )

        temp_output.replace(output)
    finally:
        temp_output.unlink(missing_ok=True)

    archive_hash = sha256_file(output)
    checksum_path.write_text(
        f"{archive_hash}  {output.name}\n",
        encoding="ascii",
        newline="\n",
    )

    print("RUNTIME BUNDLE BUILT")
    print(f"archive: {output}")
    print(f"size_mb: {output.stat().st_size / (1024 * 1024):.2f}")
    print(f"sha256: {archive_hash.upper()}")
    print(f"entry_count: {len(names)}")
    print("entries:")
    for name in names:
        print(f"  {name}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleBuildError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
