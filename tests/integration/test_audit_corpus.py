"""Integration tests for the corpus audit command."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_corpus.py"

AUTHOR_ID = "0001Author"
WORK_ID = "0001Author.Work"
VERSION_ID = "0001Author.Work.Version-per1"


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_test_corpus(
    base_path: Path,
    *,
    create_version_yml: bool = True,
) -> tuple[Path, Path]:
    corpus_root = base_path / "corpus"
    manifest_path = base_path / "manifest.yaml"

    author_dir = corpus_root / AUTHOR_ID
    work_dir = author_dir / WORK_ID

    _write_file(
        author_dir / f"{AUTHOR_ID}.yml",
        "00#AUTH#URI######: 0001Author\n",
    )

    _write_file(
        work_dir / f"{WORK_ID}.yml",
        "00#BOOK#URI######: 0001Author.Work\n",
    )

    _write_file(
        work_dir / VERSION_ID,
        (
            "######OpenITI#\n"
            "#META#Header#End#\n"
            "Test text\n"
        ),
    )

    if create_version_yml:
        _write_file(
            work_dir / f"{VERSION_ID}.yml",
            f"00#VERS#URI######: {VERSION_ID}\n",
        )

    manifest_path.write_text(
        f"""
dataset:
  name: "test-corpus"
  repository: "https://example.com/test-corpus"
  commit: "abc123"

works:
  "{WORK_ID}":
    title_fa: "Test work"
    profile: "structured_poetry"
    canonical_version: "{VERSION_ID}"
    include_in_index: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return manifest_path, corpus_root


def _run_audit(
    manifest_path: Path,
    corpus_root: Path,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--manifest",
            str(manifest_path),
            "--corpus-root",
            str(corpus_root),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_audit_command_writes_valid_report(
    tmp_path: Path,
) -> None:
    manifest_path, corpus_root = _create_test_corpus(
        tmp_path
    )

    output_path = tmp_path / "audit.json"

    result = _run_audit(
        manifest_path,
        corpus_root,
        output_path,
    )

    assert result.returncode == 0
    assert "Status: OK" in result.stdout
    assert output_path.is_file()

    report = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert report["summary"]["authors"] == 1
    assert report["summary"]["works"] == 1
    assert report["summary"]["configured_versions"] == 1
    assert report["summary"]["issues"] == 0
    assert report["summary"]["scan_ok"] is True


def test_audit_command_returns_failure_for_missing_file(
    tmp_path: Path,
) -> None:
    manifest_path, corpus_root = _create_test_corpus(
        tmp_path,
        create_version_yml=False,
    )

    output_path = tmp_path / "audit.json"

    result = _run_audit(
        manifest_path,
        corpus_root,
        output_path,
    )

    assert result.returncode == 1
    assert "Status: FAILED" in result.stdout
    assert output_path.is_file()

    report = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    issue_codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "missing_version_yml" in issue_codes
    assert report["summary"]["scan_ok"] is False