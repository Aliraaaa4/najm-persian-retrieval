"""Load OpenITI YAML metadata and text headers with diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any
import yaml

from najm_retrieval.corpus.manifest import WorkConfig
from najm_retrieval.corpus.scanner import CorpusVersionFiles

OPENITI_MAGIC_LINE = "######OpenITI#"
HEADER_END_MARKER = "#META#Header#End#"

PLACEHOLDER_PATTERN = re.compile(
    r"""
    (
        \bunknown\b
        |
        \bunknown\s+value\b
        |
        \btodo\b
        |
        \btbd\b
        |
        \bxxx+\b
        |
        \byyyy(?:-mm-dd)?\b
        |
        ^\s*n/?a\s*$
        |
        \bnone\b
        |
        \bnot\s+available\b
        |
        \bnot\s+specified\b
        |
        \bpermalink\b
        |
        \bthe\s+name\s+of\s+the\s+annotator\b
        |
        \bfulān\b
        |
        \buris?\s+from\s+althurayya\b
        |
        \bauth_uri\s+from\s+openiti\b
        |
        \bbook_uri\s+from\s+openiti\b
        |
        \byear-mon-da\b
        |
        \ba\s+free\s+running\s+comment\s+here\b
        |
        \bviaf@id,\s*wikidata@id,\s*src@id\b
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


class MetadataError(ValueError):
    """Raised when a metadata source cannot be loaded."""


class MetadataStatus(str, Enum):
    """Quality status assigned to one metadata field."""

    VALID = "valid"
    MISSING = "missing"
    PLACEHOLDER = "placeholder"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class MetadataIssue:
    """A diagnostic discovered while loading metadata."""

    code: str
    message: str
    path: Path
    field_name: str | None = None
    line_number: int | None = None


@dataclass(frozen=True)
class MetadataField:
    """One metadata field with its original and normalized values."""

    name: str
    raw_value: Any
    value: str | None
    status: MetadataStatus
    source: str
    line_number: int | None = None


@dataclass(frozen=True)
class OpenITIYamlRecord:
    """Metadata fields loaded from one OpenITI YAML file."""

    path: Path
    fields: tuple[MetadataField, ...]
    issues: tuple[MetadataIssue, ...]

    def get_all(self, field_name: str) -> tuple[MetadataField, ...]:
        """Return every field whose name exactly matches field_name."""

        return tuple(
            field
            for field in self.fields
            if field.name == field_name
        )

    def get_first(self, field_name: str) -> MetadataField | None:
        """Return the first exact field match."""

        matches = self.get_all(field_name)
        return matches[0] if matches else None

    @property
    def valid_fields(self) -> tuple[MetadataField, ...]:
        """Return fields classified as valid."""

        return tuple(
            field
            for field in self.fields
            if field.status is MetadataStatus.VALID
        )

    @property
    def placeholder_fields(self) -> tuple[MetadataField, ...]:
        """Return fields containing placeholder values."""

        return tuple(
            field
            for field in self.fields
            if field.status is MetadataStatus.PLACEHOLDER
        )


@dataclass(frozen=True)
class OpenITITextHeader:
    """Metadata header extracted from the beginning of a text file."""

    path: Path
    magic_line_valid: bool
    header_end_found: bool
    fields: tuple[MetadataField, ...]
    raw_lines: tuple[str, ...]
    issues: tuple[MetadataIssue, ...]

    def get_all(self, field_name: str) -> tuple[MetadataField, ...]:
        """Return every field whose name exactly matches field_name."""

        return tuple(
            field
            for field in self.fields
            if field.name == field_name
        )

    def get_first(self, field_name: str) -> MetadataField | None:
        """Return the first exact field match."""

        matches = self.get_all(field_name)
        return matches[0] if matches else None


def classify_metadata_value(
    value: Any,
) -> tuple[str | None, MetadataStatus]:
    """Normalize one value and determine its metadata status."""

    if value is None:
        return None, MetadataStatus.MISSING

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            return None, MetadataStatus.MISSING

        if PLACEHOLDER_PATTERN.search(normalized):
            return normalized, MetadataStatus.PLACEHOLDER

        return normalized, MetadataStatus.VALID

    if isinstance(value, bool):
        return str(value).lower(), MetadataStatus.VALID

    if isinstance(value, (int, float)):
        return str(value), MetadataStatus.VALID

    # Nested lists and mappings are preserved as raw_value, but this generic
    # scalar loader cannot safely interpret them yet.
    return None, MetadataStatus.MALFORMED

OPENITI_YAML_FIELD_PATTERN = re.compile(
    r"^(?P<name>[^\s][^:]*?):[ \t]*(?P<value>.*)$"
)


def _parse_openiti_yml_lines(
    text: str,
    *,
    path: Path,
) -> tuple[
    list[tuple[Any, Any, int | None]],
    list[MetadataIssue],
]:
    """Parse YAML-like OpenITI metadata line by line.

    Some OpenITI .yml files contain indented continuation lines with
    unquoted colons. Those files are understandable as metadata but are
    not accepted by a strict YAML parser.
    """

    entries: list[tuple[Any, Any, int | None]] = []
    issues: list[MetadataIssue] = []

    current_name: str | None = None
    current_value_lines: list[str] = []
    current_line_number: int | None = None

    def flush_current() -> None:
        nonlocal current_name
        nonlocal current_value_lines
        nonlocal current_line_number

        if current_name is None:
            return

        joined_value = "\n".join(current_value_lines).strip()

        entries.append(
            (
                current_name,
                joined_value if joined_value else None,
                current_line_number,
            )
        )

        current_name = None
        current_value_lines = []
        current_line_number = None

    for line_number, raw_line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            if current_name is not None and current_value_lines:
                current_value_lines.append("")
            continue

        # Ignore ordinary YAML comments and document markers.
        if stripped in {"---", "..."}:
            continue

        if stripped.startswith("#") and ":" not in stripped:
            continue

        # An indented line belongs to the previous field.
        if line.startswith((" ", "\t")):
            if current_name is None:
                issues.append(
                    MetadataIssue(
                        code="orphan_continuation_line",
                        message=(
                            "An indented metadata line appeared before "
                            "any field declaration."
                        ),
                        path=path,
                        line_number=line_number,
                    )
                )
            else:
                current_value_lines.append(line.lstrip())

            continue

        match = OPENITI_YAML_FIELD_PATTERN.match(line)

        if match is not None:
            flush_current()

            current_name = match.group("name").strip()
            initial_value = match.group("value").strip()
            current_line_number = line_number

            current_value_lines = (
                [initial_value]
                if initial_value
                else []
            )

            continue

        # Preserve malformed standalone lines instead of silently dropping them.
        if current_name is not None:
            current_value_lines.append(stripped)

            issues.append(
                MetadataIssue(
                    code="unparsed_yaml_line_attached",
                    message=(
                        "A non-standard metadata line was attached "
                        "to the preceding field."
                    ),
                    path=path,
                    field_name=current_name,
                    line_number=line_number,
                )
            )
        else:
            issues.append(
                MetadataIssue(
                    code="unparsed_yaml_line",
                    message=(
                        "A metadata line could not be associated "
                        "with any field."
                    ),
                    path=path,
                    line_number=line_number,
                )
            )

    flush_current()

    return entries, issues


def _load_yaml_entries(
    text: str,
    *,
    path: Path,
) -> tuple[
    list[tuple[Any, Any, int | None]],
    list[MetadataIssue],
]:
    """Use strict YAML first and a loss-preserving fallback when needed."""

    try:
        raw_data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        entries, issues = _parse_openiti_yml_lines(
            text,
            path=path,
        )

        problem_mark = getattr(exc, "problem_mark", None)

        error_line = (
            problem_mark.line + 1
            if problem_mark is not None
            else None
        )

        issues.insert(
            0,
            MetadataIssue(
                code="yaml_fallback_used",
                message=(
                    "Standard YAML parsing failed; the OpenITI "
                    "line-based fallback parser was used."
                ),
                path=path,
                line_number=error_line,
            ),
        )

        if not entries:
            raise MetadataError(
                f"Invalid YAML and no metadata fields could be "
                f"recovered from: {path}"
            ) from exc

        return entries, issues

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise MetadataError(
            f"Metadata YAML root must be a mapping: {path}"
        )

    entries = [
        (raw_name, raw_value, None)
        for raw_name, raw_value in raw_data.items()
    ]

    return entries, []

def load_openiti_yml(path: str | Path) -> OpenITIYamlRecord:
    """Load one OpenITI YAML or YAML-like metadata file."""

    metadata_path = Path(path)

    if not metadata_path.is_file():
        raise MetadataError(
            f"Metadata YAML file not found: {metadata_path}"
        )

    try:
        text = metadata_path.read_text(
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError as exc:
        raise MetadataError(
            f"Metadata YAML is not valid UTF-8: {metadata_path}"
        ) from exc
    except OSError as exc:
        raise MetadataError(
            f"Could not read metadata file: {metadata_path}"
        ) from exc

    entries, parser_issues = _load_yaml_entries(
        text,
        path=metadata_path,
    )

    fields: list[MetadataField] = []
    issues: list[MetadataIssue] = list(parser_issues)

    for raw_name, raw_value, line_number in entries:
        if not isinstance(raw_name, str) or not raw_name.strip():
            issues.append(
                MetadataIssue(
                    code="invalid_field_name",
                    message=(
                        "Metadata field name must be "
                        "a non-empty string."
                    ),
                    path=metadata_path,
                    line_number=line_number,
                )
            )
            continue

        field_name = raw_name.strip()

        normalized_value, status = classify_metadata_value(
            raw_value
        )

        fields.append(
            MetadataField(
                name=field_name,
                raw_value=raw_value,
                value=normalized_value,
                status=status,
                source="yaml",
                line_number=line_number,
            )
        )

        if status is MetadataStatus.PLACEHOLDER:
            issues.append(
                MetadataIssue(
                    code="placeholder_value",
                    message=(
                        f"Field '{field_name}' contains "
                        "a placeholder value."
                    ),
                    path=metadata_path,
                    field_name=field_name,
                    line_number=line_number,
                )
            )

        if status is MetadataStatus.MALFORMED:
            issues.append(
                MetadataIssue(
                    code="non_scalar_value",
                    message=(
                        f"Field '{field_name}' contains a "
                        "non-scalar value that was preserved "
                        "but not interpreted."
                    ),
                    path=metadata_path,
                    field_name=field_name,
                    line_number=line_number,
                )
            )

    return OpenITIYamlRecord(
        path=metadata_path,
        fields=tuple(fields),
        issues=tuple(issues),
    )

def _parse_header_metadata_line(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[MetadataField | None, MetadataIssue | None]:
    """Parse one #META# header line."""

    content = line[len("#META#") :].strip()

    if not content:
        return None, MetadataIssue(
            code="empty_header_line",
            message="An empty metadata header line was found.",
            path=path,
            line_number=line_number,
        )

    if ":" not in content:
        return None, MetadataIssue(
            code="unparsed_header_line",
            message=(
                "Metadata header line does not contain a ':' separator."
            ),
            path=path,
            line_number=line_number,
        )

    field_name, raw_value = content.split(":", maxsplit=1)
    field_name = field_name.strip()
    raw_value = raw_value.strip()

    if not field_name:
        return None, MetadataIssue(
            code="invalid_header_field_name",
            message="Metadata header field name is empty.",
            path=path,
            line_number=line_number,
        )

    normalized_value, status = classify_metadata_value(raw_value)

    field = MetadataField(
        name=field_name,
        raw_value=raw_value,
        value=normalized_value,
        status=status,
        source="text_header",
        line_number=line_number,
    )

    if status is MetadataStatus.PLACEHOLDER:
        return field, MetadataIssue(
            code="placeholder_header_value",
            message=(
                f"Header field '{field_name}' contains a placeholder value."
            ),
            path=path,
            field_name=field_name,
            line_number=line_number,
        )

    return field, None


def extract_text_header(
    path: str | Path,
    *,
    max_lines: int = 1000,
) -> OpenITITextHeader:
    """Extract the OpenITI header without reading the entire text file."""

    text_path = Path(path)

    if not text_path.is_file():
        raise MetadataError(f"Text file not found: {text_path}")

    if max_lines < 1:
        raise ValueError("max_lines must be at least 1.")

    raw_lines: list[str] = []
    fields: list[MetadataField] = []
    issues: list[MetadataIssue] = []

    magic_line_valid = False
    header_end_found = False

    try:
        with text_path.open(
            mode="r",
            encoding="utf-8-sig",
            errors="replace",
        ) as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.rstrip("\r\n")
                raw_lines.append(line)

                if line_number == 1:
                    magic_line_valid = (
                        line.strip() == OPENITI_MAGIC_LINE
                    )

                    if not magic_line_valid:
                        issues.append(
                            MetadataIssue(
                                code="invalid_magic_line",
                                message=(
                                    f"Text does not start with "
                                    f"'{OPENITI_MAGIC_LINE}'."
                                ),
                                path=text_path,
                                line_number=1,
                            )
                        )

                if line.strip() == HEADER_END_MARKER:
                    header_end_found = True
                    break

                if line.startswith("#META#"):
                    field, issue = _parse_header_metadata_line(
                        line,
                        path=text_path,
                        line_number=line_number,
                    )

                    if field is not None:
                        fields.append(field)

                    if issue is not None:
                        issues.append(issue)

                if line_number >= max_lines:
                    issues.append(
                        MetadataIssue(
                            code="header_scan_limit_reached",
                            message=(
                                f"Header end marker was not found within "
                                f"the first {max_lines} lines."
                            ),
                            path=text_path,
                            line_number=line_number,
                        )
                    )
                    break
    except OSError as exc:
        raise MetadataError(
            f"Could not read text header: {text_path}"
        ) from exc

    if not header_end_found:
        issues.append(
            MetadataIssue(
                code="missing_header_end",
                message=(
                    f"Header end marker '{HEADER_END_MARKER}' was not found."
                ),
                path=text_path,
            )
        )

    return OpenITITextHeader(
        path=text_path,
        magic_line_valid=magic_line_valid,
        header_end_found=header_end_found,
        fields=tuple(fields),
        raw_lines=tuple(raw_lines),
        issues=tuple(issues),
    )
    
    
@dataclass(frozen=True)
class MetadataBundle:
    """All raw metadata sources belonging to one corpus version."""

    author_yml: OpenITIYamlRecord
    work_yml: OpenITIYamlRecord
    version_yml: OpenITIYamlRecord
    text_header: OpenITITextHeader

    @property
    def issues(self) -> tuple[MetadataIssue, ...]:
        """Return diagnostics from all raw metadata sources."""

        return (
            self.author_yml.issues
            + self.work_yml.issues
            + self.version_yml.issues
            + self.text_header.issues
        )


@dataclass(frozen=True)
class DocumentMetadata:
    """Normalized metadata used by passages, search, and the API."""

    author_id: str
    author_name: str
    author_name_ar: str | None
    author_name_en: str | None

    work_id: str
    title_fa: str
    title_transliterated: str | None

    version_id: str
    language_code: str | None
    language: str | None

    source_url: str | None
    text_quality: tuple[str, ...]
    origin: str | None
    transcription_layer: str | None
    ocr_confidence: float | None

    profile: str
    is_canonical: bool
    include_in_index: bool

    issues: tuple[MetadataIssue, ...]


def _first_valid_value(
    record: OpenITIYamlRecord | OpenITITextHeader,
    *field_names: str,
) -> str | None:
    """Return the first valid value matching one of the field names."""

    for field_name in field_names:
        for field in record.get_all(field_name):
            if (
                field.status is MetadataStatus.VALID
                and field.value is not None
            ):
                return field.value

    return None


def _first_valid_url(
    record: OpenITIYamlRecord | OpenITITextHeader,
    *field_names: str,
) -> str | None:
    """Return the first valid HTTP or HTTPS URL."""

    for field_name in field_names:
        value = _first_valid_value(record, field_name)

        if value is not None and value.startswith(
            ("http://", "https://")
        ):
            return value

    return None


def _language_from_version_id(
    version_id: str,
) -> tuple[str | None, str | None]:
    """Derive language from the OpenITI version suffix."""

    match = re.search(
        r"-(?P<code>[a-z]{3})\d+$",
        version_id,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None, None

    code = match.group("code").lower()

    language_names = {
        "per": "Persian",
        "ara": "Arabic",
    }

    return code, language_names.get(code, code)


def _split_comma_separated(
    value: str | None,
) -> tuple[str, ...]:
    """Split a comma-separated metadata value."""

    if value is None:
        return ()

    return tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )


def _check_identifier(
    *,
    expected: str,
    actual: str | None,
    label: str,
    path: Path,
    field_name: str,
    issues: list[MetadataIssue],
) -> None:
    """Compare a YAML identifier with the Scanner identifier."""

    code_prefix = label.lower().replace(" ", "_")

    if actual is None:
        issues.append(
            MetadataIssue(
                code=f"missing_{code_prefix}",
                message=(
                    f"{label} is missing; the Scanner value "
                    f"'{expected}' will be used."
                ),
                path=path,
                field_name=field_name,
            )
        )
        return

    if actual != expected:
        issues.append(
            MetadataIssue(
                code=f"{code_prefix}_mismatch",
                message=(
                    f"{label} '{actual}' does not match "
                    f"the Scanner value '{expected}'."
                ),
                path=path,
                field_name=field_name,
            )
        )


def load_version_metadata(
    version: CorpusVersionFiles,
) -> MetadataBundle:
    """Load all raw metadata sources for one corpus version."""

    return MetadataBundle(
        author_yml=load_openiti_yml(
            version.author_yml_path
        ),
        work_yml=load_openiti_yml(
            version.work_yml_path
        ),
        version_yml=load_openiti_yml(
            version.version_yml_path
        ),
        text_header=extract_text_header(
            version.text_path
        ),
    )


def build_document_metadata(
    version: CorpusVersionFiles,
    work: WorkConfig,
    bundle: MetadataBundle,
) -> DocumentMetadata:
    """Combine Manifest, Scanner, YAML, and header metadata."""

    if work.work_id != version.work_id:
        raise MetadataError(
            f"Work configuration '{work.work_id}' does not match "
            f"version work ID '{version.work_id}'."
        )

    issues = list(bundle.issues)

    author_uri = _first_valid_value(
        bundle.author_yml,
        "00#AUTH#URI######",
    )

    work_uri = _first_valid_value(
        bundle.work_yml,
        "00#BOOK#URI######",
    )

    version_uri = _first_valid_value(
        bundle.version_yml,
        "00#VERS#URI######",
    )

    _check_identifier(
        expected=version.author_id,
        actual=author_uri,
        label="author URI",
        path=version.author_yml_path,
        field_name="00#AUTH#URI######",
        issues=issues,
    )

    _check_identifier(
        expected=version.work_id,
        actual=work_uri,
        label="work URI",
        path=version.work_yml_path,
        field_name="00#BOOK#URI######",
        issues=issues,
    )

    _check_identifier(
        expected=version.version_id,
        actual=version_uri,
        label="version URI",
        path=version.version_yml_path,
        field_name="00#VERS#URI######",
        issues=issues,
    )

    author_name_ar = _first_valid_value(
        bundle.author_yml,
        "10#AUTH#SHUHRA#AR",
    )

    author_name_en = _first_valid_value(
        bundle.author_yml,
        "10#AUTH#SHUHRA#EN",
    )

    # Arabic-script/transliterated OpenITI name is available
    # consistently for all selected authors.
    author_name = (
        author_name_ar
        or author_name_en
        or version.author_id
    )

    title_transliterated = _first_valid_value(
        bundle.work_yml,
        "10#BOOK#TITLEA#AR",
    )

    source_url = _first_valid_url(
        bundle.version_yml,
        "80#VERS#BASED####",
    )

    if source_url is None:
        source_url = _first_valid_url(
            bundle.text_header,
            "url",
        )

    if source_url is None:
        source_url = _first_valid_url(
            bundle.version_yml,
            "80#VERS#LINKS####",
        )

    quality_value = _first_valid_value(
        bundle.version_yml,
        "90#VERS#ISSUES###",
    )

    text_quality = _split_comma_separated(
        quality_value
    )

    origin = _first_valid_value(
        bundle.text_header,
        "Origin",
    )

    transcription_layer = _first_valid_value(
        bundle.text_header,
        "transcription layer name",
    )

    confidence_value = _first_valid_value(
        bundle.text_header,
        "avg transcription confidence",
    )

    ocr_confidence: float | None = None

    if confidence_value is not None:
        try:
            ocr_confidence = float(confidence_value)
        except ValueError:
            issues.append(
                MetadataIssue(
                    code="invalid_ocr_confidence",
                    message=(
                        "OCR confidence could not be converted "
                        f"to a number: {confidence_value!r}"
                    ),
                    path=version.text_path,
                    field_name="avg transcription confidence",
                )
            )

    language_code, language = _language_from_version_id(
        version.version_id
    )

    if language_code is None:
        issues.append(
            MetadataIssue(
                code="missing_language_code",
                message=(
                    "Language could not be derived from "
                    f"version ID '{version.version_id}'."
                ),
                path=version.text_path,
            )
        )

    return DocumentMetadata(
        author_id=version.author_id,
        author_name=author_name,
        author_name_ar=author_name_ar,
        author_name_en=author_name_en,
        work_id=version.work_id,
        title_fa=work.title_fa,
        title_transliterated=title_transliterated,
        version_id=version.version_id,
        language_code=language_code,
        language=language,
        source_url=source_url,
        text_quality=text_quality,
        origin=origin,
        transcription_layer=transcription_layer,
        ocr_confidence=ocr_confidence,
        profile=version.profile,
        is_canonical=version.is_canonical,
        include_in_index=version.include_in_index,
        issues=tuple(issues),
    )


def load_document_metadata(
    version: CorpusVersionFiles,
    work: WorkConfig,
) -> DocumentMetadata:
    """Load and normalize metadata for one corpus version."""

    bundle = load_version_metadata(version)

    return build_document_metadata(
        version,
        work,
        bundle,
    )