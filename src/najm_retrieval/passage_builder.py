"""Structure-aware passage construction for retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable
import re

from najm_retrieval.normalization import (
    normalize_text,
)
from najm_retrieval.passages import (
    Passage,
    PassageBoundary,
    PassageBuildResult,
    PassageConfig,
    PassageIssue,
    PassageKind,
    PassageMember,
)
from najm_retrieval.text_preparation.context_models import (
    ContextualLogicalUnit,
)
from najm_retrieval.text_preparation.structural_cleanup import (
    StructurallyCleanedText,
)


_CONTENT_TYPES = {
    "verse",
    "paragraph",
    "raw",
}

_BOUNDARY_TYPES = {
    "heading",
    "section",
}

_MIXED_CONTENT_TYPES = {
    "verse",
    "paragraph",
}

_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!؟!?؛])\s+"
)


@dataclass(frozen=True)
class _PreparedUnit:
    """One aligned contextual, cleaned, and normalized unit."""

    unit_id: str
    unit_type: str

    heading_path: tuple[str, ...]
    section_path: tuple[str, ...]

    member: PassageMember | None = None
    boundary: PassageBoundary | None = None


@dataclass(frozen=True)
class _Run:
    """One source-order run that cannot cross a boundary."""

    heading_path: tuple[str, ...]
    section_path: tuple[str, ...]
    members: tuple[PassageMember, ...]


@dataclass(frozen=True)
class _PassageSpec:
    """Internal passage data before IDs and neighbors are assigned."""

    kind: PassageKind
    link_group: int

    heading_path: tuple[str, ...]
    section_path: tuple[str, ...]

    members: tuple[PassageMember, ...]

    issues: tuple[PassageIssue, ...] = field(
        default_factory=tuple
    )


def build_passages(
    *,
    version_id: str,
    profile: str,
    include_in_index: bool,
    contextual_units: Iterable[
        ContextualLogicalUnit
    ],
    cleaned_units: Iterable[
        StructurallyCleanedText
    ],
    config: PassageConfig | None = None,
) -> PassageBuildResult:
    """Build deterministic structure-aware passages for one version."""

    if not version_id.strip():
        raise ValueError(
            "version_id must not be empty."
        )

    if not profile.strip():
        raise ValueError(
            "profile must not be empty."
        )

    resolved_config = (
        config
        if config is not None
        else PassageConfig()
    )

    contextual_tuple = tuple(
        contextual_units
    )

    cleaned_tuple = tuple(
        cleaned_units
    )

    if (
        len(contextual_tuple)
        != len(cleaned_tuple)
    ):
        raise ValueError(
            "Contextual and cleaned unit counts "
            "must match."
        )

    prepared_units: list[
        _PreparedUnit
    ] = []

    skipped_unit_ids: list[str] = []
    build_issues: list[PassageIssue] = []

    for contextual, cleaned in zip(
        contextual_tuple,
        cleaned_tuple,
    ):
        prepared = _prepare_unit(
            version_id=version_id,
            contextual=contextual,
            cleaned=cleaned,
        )

        prepared_units.append(
            prepared
        )

        if (
            prepared.unit_type
            in _CONTENT_TYPES
            and prepared.member is None
        ):
            skipped_unit_ids.append(
                prepared.unit_id
            )

            build_issues.append(
                PassageIssue(
                    code="empty_content_unit",
                    message=(
                        "The source unit contains no "
                        "indexable text after cleanup "
                        "and normalization."
                    ),
                    source_unit_ids=(
                        prepared.unit_id,
                    ),
                )
            )

    if not include_in_index:
        reference_content_ids = [
            prepared.unit_id
            for prepared in prepared_units
            if (
                prepared.unit_type
                in _CONTENT_TYPES
                and prepared.member is not None
            )
        ]

        skipped_unit_ids.extend(
            reference_content_ids
        )

        if reference_content_ids:
            build_issues.append(
                PassageIssue(
                    code="version_excluded_from_index",
                    message=(
                        "The version is reference-only "
                        "and no retrieval passages were built."
                    ),
                )
            )

        return PassageBuildResult(
            config=resolved_config,
            passages=(),
            skipped_unit_ids=tuple(
                dict.fromkeys(
                    skipped_unit_ids
                )
            ),
            issues=tuple(
                build_issues
            ),
        )

    boundary_map = {
        prepared.boundary.unit_id:
            prepared.boundary
        for prepared in prepared_units
        if prepared.boundary is not None
    }

    if profile == "structured_poetry":
        if ".Mathnawi." in version_id:
            specs = _build_poetry_specs(
                prepared_units,
                kind=PassageKind.MATHNAWI,
                target_verses=(
                    resolved_config
                    .mathnawi_target_verses
                ),
                overlap_verses=(
                    resolved_config
                    .mathnawi_overlap_verses
                ),
                minimum_tail_verses=(
                    resolved_config
                    .mathnawi_minimum_tail_verses
                ),
            )
        else:
            specs = _build_poetry_specs(
                prepared_units,
                kind=PassageKind.DIWAN,
                target_verses=(
                    resolved_config
                    .diwan_target_verses
                ),
                overlap_verses=(
                    resolved_config
                    .diwan_overlap_verses
                ),
                minimum_tail_verses=(
                    resolved_config
                    .diwan_minimum_tail_verses
                ),
            )

    elif profile == "mixed_prose_ocr":
        specs = _build_mixed_specs(
            prepared_units,
            config=resolved_config,
        )

    else:
        raise ValueError(
            "Unsupported indexable passage profile: "
            f"{profile!r}."
        )

    passages = _materialize_passages(
        version_id=version_id,
        profile=profile,
        include_in_index=include_in_index,
        specs=specs,
        boundary_map=boundary_map,
    )

    return PassageBuildResult(
        config=resolved_config,
        passages=passages,
        skipped_unit_ids=tuple(
            dict.fromkeys(
                skipped_unit_ids
            )
        ),
        issues=tuple(
            build_issues
        ),
    )


def _prepare_unit(
    *,
    version_id: str,
    contextual: ContextualLogicalUnit,
    cleaned: StructurallyCleanedText,
) -> _PreparedUnit:
    """Align and prepare one logical unit."""

    unit = contextual.unit
    context = contextual.context

    if unit.unit_id != cleaned.unit_id:
        raise ValueError(
            "Contextual and cleaned unit IDs "
            "must match."
        )

    if unit.version_id != version_id:
        raise ValueError(
            "Every logical unit must belong to "
            "the requested version."
        )

    unit_type = _block_type_value(
        unit.unit_type
    )

    normalized = normalize_text(
        cleaned.display_text
    )

    if unit_type in _BOUNDARY_TYPES:
        return _PreparedUnit(
            unit_id=unit.unit_id,
            unit_type=unit_type,
            heading_path=(
                context.heading_path
            ),
            section_path=(
                context.section_path
            ),
            boundary=PassageBoundary(
                unit_id=unit.unit_id,
                unit_type=unit.unit_type,
                display_text=(
                    normalized.display_text
                ),
                metadata=tuple(
                    cleaned.metadata
                ),
            ),
        )

    if unit_type not in _CONTENT_TYPES:
        return _PreparedUnit(
            unit_id=unit.unit_id,
            unit_type=unit_type,
            heading_path=(
                context.heading_path
            ),
            section_path=(
                context.section_path
            ),
        )

    if not normalized.retrieval_text:
        return _PreparedUnit(
            unit_id=unit.unit_id,
            unit_type=unit_type,
            heading_path=(
                context.heading_path
            ),
            section_path=(
                context.section_path
            ),
        )

    source_issue_codes = tuple(
        dict.fromkeys(
            [
                issue.code
                for issue in unit.issues
            ]
            + [
                issue.code
                for issue in context.issues
            ]
            + [
                issue.code
                for issue in cleaned.issues
            ]
        )
    )

    member = PassageMember(
        unit_id=unit.unit_id,
        unit_type=unit.unit_type,
        display_text=(
            normalized.display_text
        ),
        retrieval_text=(
            normalized.retrieval_text
        ),
        search_alias_text=(
            normalized.search_alias_text
        ),
        source_spans=tuple(
            unit.source_spans
        ),
        metadata=tuple(
            cleaned.metadata
        ),
        source_issue_codes=(
            source_issue_codes
        ),
    )

    return _PreparedUnit(
        unit_id=unit.unit_id,
        unit_type=unit_type,
        heading_path=(
            context.heading_path
        ),
        section_path=(
            context.section_path
        ),
        member=member,
    )


def _build_poetry_specs(
    prepared_units: list[_PreparedUnit],
    *,
    kind: PassageKind,
    target_verses: int,
    overlap_verses: int,
    minimum_tail_verses: int,
) -> tuple[_PassageSpec, ...]:
    """Build structure-aware verse windows."""

    runs = _collect_runs(
        prepared_units,
        allowed_types={"verse"},
    )

    specs: list[_PassageSpec] = []

    for link_group, run in enumerate(
        runs
    ):
        windows = _poetry_windows(
            run.members,
            target_verses=target_verses,
            overlap_verses=overlap_verses,
            minimum_tail_verses=(
                minimum_tail_verses
            ),
        )

        for members in windows:
            specs.append(
                _PassageSpec(
                    kind=kind,
                    link_group=link_group,
                    heading_path=(
                        run.heading_path
                    ),
                    section_path=(
                        run.section_path
                    ),
                    members=members,
                )
            )

    return tuple(specs)


def _build_mixed_specs(
    prepared_units: list[_PreparedUnit],
    *,
    config: PassageConfig,
) -> tuple[_PassageSpec, ...]:
    """Build passages from paragraphs and embedded verses."""

    runs = _collect_runs(
        prepared_units,
        allowed_types=(
            _MIXED_CONTENT_TYPES
        ),
    )

    specs: list[_PassageSpec] = []

    for link_group, run in enumerate(
        runs
    ):
        expanded_members: list[
            PassageMember
        ] = []

        for member in run.members:
            if (
                _block_type_value(
                    member.unit_type
                )
                == "paragraph"
                and member.word_count
                > config.prose_hard_max_words
            ):
                expanded_members.extend(
                    _split_oversized_member(
                        member,
                        target_words=(
                            config
                            .prose_target_words
                        ),
                        hard_max_words=(
                            config
                            .prose_hard_max_words
                        ),
                    )
                )
            else:
                expanded_members.append(
                    member
                )

        packed_passages = (
            _pack_mixed_members(
                tuple(expanded_members),
                target_words=(
                    config.prose_target_words
                ),
                soft_min_words=(
                    config
                    .prose_soft_min_words
                ),
                hard_max_words=(
                    config
                    .prose_hard_max_words
                ),
            )
        )

        for members in packed_passages:
            specs.append(
                _PassageSpec(
                    kind=(
                        PassageKind.MIXED_PROSE
                    ),
                    link_group=link_group,
                    heading_path=(
                        run.heading_path
                    ),
                    section_path=(
                        run.section_path
                    ),
                    members=members,
                    issues=(
                        _split_issues_for_members(
                            members
                        )
                    ),
                )
            )

    return tuple(specs)


def _collect_runs(
    prepared_units: Iterable[
        _PreparedUnit
    ],
    *,
    allowed_types: set[str],
) -> tuple[_Run, ...]:
    """Collect source-order runs sharing structural context."""

    runs: list[_Run] = []

    current_members: list[
        PassageMember
    ] = []

    current_heading: tuple[
        str,
        ...,
    ] = ()

    current_section: tuple[
        str,
        ...,
    ] = ()

    current_key: tuple[
        tuple[str, ...],
        tuple[str, ...],
    ] | None = None

    def flush() -> None:
        nonlocal current_members
        nonlocal current_heading
        nonlocal current_section
        nonlocal current_key

        if current_members:
            runs.append(
                _Run(
                    heading_path=(
                        current_heading
                    ),
                    section_path=(
                        current_section
                    ),
                    members=tuple(
                        current_members
                    ),
                )
            )

        current_members = []
        current_heading = ()
        current_section = ()
        current_key = None

    for prepared in prepared_units:
        if (
            prepared.unit_type
            in _BOUNDARY_TYPES
        ):
            flush()
            continue

        if (
            prepared.unit_type
            not in allowed_types
            or prepared.member is None
        ):
            flush()
            continue

        key = (
            prepared.heading_path,
            prepared.section_path,
        )

        if (
            current_key is not None
            and key != current_key
        ):
            flush()

        if current_key is None:
            current_key = key
            current_heading = (
                prepared.heading_path
            )
            current_section = (
                prepared.section_path
            )

        current_members.append(
            prepared.member
        )

    flush()

    return tuple(runs)


def _poetry_windows(
    members: tuple[
        PassageMember,
        ...,
    ],
    *,
    target_verses: int,
    overlap_verses: int,
    minimum_tail_verses: int,
) -> tuple[
    tuple[PassageMember, ...],
    ...,
]:
    """Create verse windows without crossing one structural run."""

    if not members:
        return ()

    if len(members) <= target_verses:
        return (
            members,
        )

    step = (
        target_verses
        - overlap_verses
    )

    windows: list[
        tuple[PassageMember, ...]
    ] = []

    start = 0

    while start < len(members):
        end = min(
            start + target_verses,
            len(members),
        )

        window = members[
            start:end
        ]

        if (
            end == len(members)
            and len(window)
            < minimum_tail_verses
            and windows
        ):
            shifted_start = max(
                0,
                len(members)
                - target_verses,
            )

            shifted_window = members[
                shifted_start:
            ]

            if (
                shifted_window
                != windows[-1]
            ):
                windows.append(
                    shifted_window
                )

            break

        windows.append(
            window
        )

        if end >= len(members):
            break

        start += step

    return tuple(windows)


def _split_oversized_member(
    member: PassageMember,
    *,
    target_words: int,
    hard_max_words: int,
) -> tuple[PassageMember, ...]:
    """Split one oversized paragraph into normalized fragments."""

    segments, used_fallback = (
        _split_display_text(
            member.display_text,
            target_words=target_words,
            hard_max_words=(
                hard_max_words
            ),
        )
    )

    issue_code = (
        "oversized_unit_fallback_split"
        if used_fallback
        else "oversized_unit_sentence_split"
    )

    segment_count = len(
        segments
    )

    fragments: list[
        PassageMember
    ] = []

    for segment_index, segment in enumerate(
        segments
    ):
        normalized = normalize_text(
            segment
        )

        fragments.append(
            PassageMember(
                unit_id=member.unit_id,
                unit_type=member.unit_type,
                display_text=(
                    normalized.display_text
                ),
                retrieval_text=(
                    normalized.retrieval_text
                ),
                search_alias_text=(
                    normalized.search_alias_text
                ),
                segment_index=(
                    segment_index
                ),
                segment_count=(
                    segment_count
                ),
                source_spans=(
                    member.source_spans
                ),
                metadata=(
                    member.metadata
                ),
                source_issue_codes=tuple(
                    dict.fromkeys(
                        member.source_issue_codes
                        + (
                            issue_code,
                        )
                    )
                ),
            )
        )

    if any(
        fragment.word_count
        > hard_max_words
        for fragment in fragments
    ):
        raise RuntimeError(
            "Oversized paragraph splitting "
            "exceeded the hard maximum."
        )

    return tuple(fragments)


def _split_display_text(
    text: str,
    *,
    target_words: int,
    hard_max_words: int,
) -> tuple[
    tuple[str, ...],
    bool,
]:
    """Split display text at sentences with a word fallback."""

    sentence_parts = tuple(
        part.strip()
        for part in (
            _SENTENCE_SPLIT_PATTERN
            .split(text)
        )
        if part.strip()
    )

    if not sentence_parts:
        return (
            tuple(
                _split_words(
                    text,
                    target_words=(
                        target_words
                    ),
                )
            ),
            True,
        )

    segments: list[str] = []

    current_parts: list[str] = []
    current_words = 0
    used_fallback = False

    def flush() -> None:
        nonlocal current_parts
        nonlocal current_words

        if current_parts:
            segments.append(
                " ".join(
                    current_parts
                )
            )

        current_parts = []
        current_words = 0

    for sentence in sentence_parts:
        sentence_words = len(
            sentence.split()
        )

        if sentence_words > hard_max_words:
            flush()

            fallback_parts = _split_words(
                sentence,
                target_words=(
                    target_words
                ),
            )

            segments.extend(
                fallback_parts
            )

            used_fallback = True
            continue

        proposed_words = (
            current_words
            + sentence_words
        )

        if (
            current_parts
            and proposed_words
            > hard_max_words
        ):
            flush()

        current_parts.append(
            sentence
        )

        current_words += (
            sentence_words
        )

        if current_words >= target_words:
            flush()

    flush()

    if len(sentence_parts) == 1:
        used_fallback = True

    return (
        tuple(segments),
        used_fallback,
    )


def _split_words(
    text: str,
    *,
    target_words: int,
) -> tuple[str, ...]:
    """Split text into nonempty fixed-word fragments."""

    tokens = text.split()

    if not tokens:
        return ()

    return tuple(
        " ".join(
            tokens[
                start:
                start + target_words
            ]
        )
        for start in range(
            0,
            len(tokens),
            target_words,
        )
    )


def _pack_mixed_members(
    members: tuple[
        PassageMember,
        ...,
    ],
    *,
    target_words: int,
    soft_min_words: int,
    hard_max_words: int,
) -> tuple[
    tuple[PassageMember, ...],
    ...,
]:
    """Greedily pack mixed prose members within one run."""

    passages: list[
        tuple[PassageMember, ...]
    ] = []

    current: list[
        PassageMember
    ] = []

    current_words = 0

    def flush() -> None:
        nonlocal current
        nonlocal current_words

        if current:
            passages.append(
                tuple(current)
            )

        current = []
        current_words = 0

    for member in members:
        if member.word_count > hard_max_words:
            raise RuntimeError(
                "An unsplit member exceeds the "
                "prose hard maximum."
            )

        proposed_words = (
            current_words
            + member.word_count
        )

        if (
            current
            and proposed_words
            > hard_max_words
        ):
            flush()

        current.append(
            member
        )

        current_words += (
            member.word_count
        )

        if current_words >= target_words:
            flush()

    flush()

    passages = _rebalance_mixed_tail(
        passages,
        soft_min_words=soft_min_words,
        hard_max_words=hard_max_words,
    )

    if any(
        _members_word_count(passage)
        > hard_max_words
        for passage in passages
    ):
        raise RuntimeError(
            "Packed prose passage exceeds "
            "the hard maximum."
        )

    return tuple(passages)


def _rebalance_mixed_tail(
    passages: list[
        tuple[PassageMember, ...]
    ],
    *,
    soft_min_words: int,
    hard_max_words: int,
) -> list[
    tuple[PassageMember, ...]
]:
    """Rebalance a short final passage inside one run."""

    if len(passages) < 2:
        return passages

    previous = list(
        passages[-2]
    )

    tail = list(
        passages[-1]
    )

    previous_words = (
        _members_word_count(
            tuple(previous)
        )
    )

    tail_words = (
        _members_word_count(
            tuple(tail)
        )
    )

    while (
        tail_words < soft_min_words
        and len(previous) > 1
    ):
        candidate = previous[-1]

        proposed_previous = (
            previous_words
            - candidate.word_count
        )

        proposed_tail = (
            tail_words
            + candidate.word_count
        )

        if (
            proposed_previous
            < soft_min_words
            or proposed_tail
            > hard_max_words
        ):
            break

        previous.pop()

        tail.insert(
            0,
            candidate,
        )

        previous_words = (
            proposed_previous
        )

        tail_words = (
            proposed_tail
        )

    if (
        tail_words < soft_min_words
        and previous_words + tail_words
        <= hard_max_words
    ):
        passages[-2:] = [
            tuple(
                previous + tail
            )
        ]

        return passages

    passages[-2] = tuple(
        previous
    )

    passages[-1] = tuple(
        tail
    )

    return passages


def _split_issues_for_members(
    members: tuple[
        PassageMember,
        ...,
    ],
) -> tuple[PassageIssue, ...]:
    """Create passage diagnostics for split source units."""

    sentence_split_ids = tuple(
        dict.fromkeys(
            member.unit_id
            for member in members
            if (
                "oversized_unit_sentence_split"
                in member.source_issue_codes
            )
        )
    )

    fallback_split_ids = tuple(
        dict.fromkeys(
            member.unit_id
            for member in members
            if (
                "oversized_unit_fallback_split"
                in member.source_issue_codes
            )
        )
    )

    issues: list[
        PassageIssue
    ] = []

    if sentence_split_ids:
        issues.append(
            PassageIssue(
                code=(
                    "oversized_unit_sentence_split"
                ),
                message=(
                    "An oversized paragraph was "
                    "split at sentence boundaries."
                ),
                source_unit_ids=(
                    sentence_split_ids
                ),
            )
        )

    if fallback_split_ids:
        issues.append(
            PassageIssue(
                code=(
                    "oversized_unit_fallback_split"
                ),
                message=(
                    "An oversized paragraph required "
                    "a fixed-word fallback split."
                ),
                source_unit_ids=(
                    fallback_split_ids
                ),
            )
        )

    return tuple(issues)


def _materialize_passages(
    *,
    version_id: str,
    profile: str,
    include_in_index: bool,
    specs: tuple[
        _PassageSpec,
        ...,
    ],
    boundary_map: dict[
        str,
        PassageBoundary,
    ],
) -> tuple[Passage, ...]:
    """Assign stable IDs and same-run neighbor links."""

    passages: list[
        Passage
    ] = []

    for ordinal, spec in enumerate(
        specs,
        start=1,
    ):
        context_ids = (
            spec.section_path
            + spec.heading_path
        )

        boundaries = tuple(
            boundary_map[boundary_id]
            for boundary_id in context_ids
            if boundary_id in boundary_map
        )

        passages.append(
            Passage(
                passage_id=(
                    f"{version_id}:"
                    f"passage_{ordinal:06d}"
                ),
                version_id=version_id,
                profile=profile,
                kind=spec.kind,
                ordinal=ordinal,
                include_in_index=(
                    include_in_index
                ),
                members=spec.members,
                heading_path=(
                    spec.heading_path
                ),
                section_path=(
                    spec.section_path
                ),
                boundaries=boundaries,
                issues=spec.issues,
            )
        )

    linked: list[
        Passage
    ] = []

    for index, passage in enumerate(
        passages
    ):
        previous_id: str | None = None
        next_id: str | None = None

        if (
            index > 0
            and specs[index - 1].link_group
            == specs[index].link_group
        ):
            previous_id = (
                passages[index - 1]
                .passage_id
            )

        if (
            index + 1 < len(passages)
            and specs[index + 1].link_group
            == specs[index].link_group
        ):
            next_id = (
                passages[index + 1]
                .passage_id
            )

        linked.append(
            replace(
                passage,
                previous_passage_id=(
                    previous_id
                ),
                next_passage_id=(
                    next_id
                ),
            )
        )

    return tuple(linked)


def _members_word_count(
    members: tuple[
        PassageMember,
        ...,
    ],
) -> int:
    """Return total retrieval words for members."""

    return sum(
        member.word_count
        for member in members
    )


def _block_type_value(
    block_type: object,
) -> str:
    """Return the serialized value of a block type."""

    return str(
        getattr(
            block_type,
            "value",
            block_type,
        )
    )


__all__ = [
    "build_passages",
]
