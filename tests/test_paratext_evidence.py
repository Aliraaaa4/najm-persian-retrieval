"""Tests for structural paratext classification and evidence."""

from __future__ import annotations

from dataclasses import (
    FrozenInstanceError,
    replace,
)
from pathlib import Path

import pytest
import yaml

from najm_retrieval.retrieval import (
    ContentRole,
    HybridSearchHit,
    HybridSearchResult,
    ParatextCatalog,
    ParatextCatalogError,
    ParatextEvidenceExtractor,
    parse_passage_ordinal,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

CATALOG_PATH = (
    PROJECT_ROOT
    / "config"
    / "paratext_zones.yaml"
)

MAJALIS_VERSION = (
    "0672JalalDinRumi."
    "MajalisSabica."
    "AOCP202502141236-per1"
)


def load_catalog() -> ParatextCatalog:
    return ParatextCatalog.from_yaml(
        CATALOG_PATH
    )


def passage_id(
    version_id: str,
    ordinal: int,
) -> str:
    return (
        f"{version_id}:"
        f"passage_{ordinal:06d}"
    )


def make_hit(
    *,
    version_id: str,
    ordinal: int,
    rank: int,
) -> HybridSearchHit:
    return HybridSearchHit(
        passage_id=passage_id(
            version_id,
            ordinal,
        ),
        version_id=version_id,
        kind="mixed_prose",
        rank=rank,
        fusion_score=0.05 / rank,
        lexical_rank=rank,
        dense_rank=rank,
        lexical_bm25_score=(
            -5.0 + rank
        ),
        dense_cosine_score=(
            0.90 - rank / 100
        ),
    )


def hybrid_result(
    query_text: str,
    *items: tuple[
        str,
        int,
    ],
) -> HybridSearchResult:
    hits = tuple(
        make_hit(
            version_id=version_id,
            ordinal=ordinal,
            rank=rank,
        )
        for rank, (
            version_id,
            ordinal,
        ) in enumerate(
            items,
            start=1,
        )
    )

    return HybridSearchResult(
        query_text=query_text,
        hits=hits,
        latency_ms=3.0,
        lexical_latency_ms=1.0,
        dense_latency_ms=2.0,
        lexical_weight=2.0,
        dense_weight=1.0,
        rrf_constant=60.0,
        candidate_limit=100,
    )


def test_catalog_loads_one_reviewed_version() -> None:
    catalog = load_catalog()

    assert (
        catalog.configured_version_ids
        == (MAJALIS_VERSION,)
    )

    assert (
        catalog.expected_passage_count(
            MAJALIS_VERSION
        )
        == 239
    )


@pytest.mark.parametrize(
    ("ordinal", "expected_role", "expected_reason"),
    (
        (
            1,
            ContentRole.PARATEXT,
            "editorial_front_matter",
        ),
        (
            13,
            ContentRole.PARATEXT,
            "editorial_front_matter",
        ),
        (
            24,
            ContentRole.PARATEXT,
            "editorial_front_matter",
        ),
        (
            25,
            ContentRole.MIXED,
            "front_matter_to_authorial_transition",
        ),
        (
            26,
            ContentRole.AUTHORIAL,
            "main_work",
        ),
        (
            35,
            ContentRole.AUTHORIAL,
            "main_work",
        ),
        (
            239,
            ContentRole.AUTHORIAL,
            "main_work",
        ),
    ),
)
def test_reviewed_majalis_boundaries(
    ordinal: int,
    expected_role: ContentRole,
    expected_reason: str,
) -> None:
    evidence = load_catalog().classify(
        passage_id=passage_id(
            MAJALIS_VERSION,
            ordinal,
        ),
        version_id=MAJALIS_VERSION,
    )

    assert (
        evidence.role
        is expected_role
    )
    assert evidence.configured
    assert (
        evidence.reason
        == expected_reason
    )


def test_later_authorial_use_of_version_word_is_not_paratext() -> None:
    evidence = load_catalog().classify(
        passage_id=passage_id(
            MAJALIS_VERSION,
            35,
        ),
        version_id=MAJALIS_VERSION,
    )

    assert (
        evidence.role
        is ContentRole.AUTHORIAL
    )


def test_unconfigured_version_is_unknown() -> None:
    version_id = (
        "0672JalalDinRumi."
        "Mathnawi.PDL00048-per1"
    )

    evidence = load_catalog().classify(
        passage_id=passage_id(
            version_id,
            100,
        ),
        version_id=version_id,
    )

    assert (
        evidence.role
        is ContentRole.UNKNOWN
    )
    assert not evidence.configured
    assert evidence.reason is None


def test_parse_passage_ordinal() -> None:
    assert (
        parse_passage_ordinal(
            passage_id=passage_id(
                MAJALIS_VERSION,
                13,
            ),
            version_id=MAJALIS_VERSION,
        )
        == 13
    )


def test_passage_id_version_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        parse_passage_ordinal(
            passage_id=passage_id(
                MAJALIS_VERSION,
                13,
            ),
            version_id="other-version",
        )


def test_invalid_passage_id_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="passage_<ordinal>",
    ):
        parse_passage_ordinal(
            passage_id="invalid",
            version_id=MAJALIS_VERSION,
        )


def test_extractor_marks_paratext_top_hit() -> None:
    query = (
        "مولوی درباره دستگاه چاپ چه نظری داشته است؟"
    )

    result = hybrid_result(
        query,
        (
            MAJALIS_VERSION,
            13,
        ),
        (
            MAJALIS_VERSION,
            10,
        ),
        (
            MAJALIS_VERSION,
            35,
        ),
    )

    evidence = ParatextEvidenceExtractor(
        load_catalog()
    ).extract(
        result
    )

    assert (
        evidence.top_hit_role
        is ContentRole.PARATEXT
    )
    assert evidence.top_hit_is_paratext
    assert not evidence.top_hit_is_mixed
    assert (
        evidence.top_hit_is_structurally_non_authorial
    )

    assert evidence.paratext_hit_count == 2
    assert evidence.authorial_hit_count == 1
    assert evidence.mixed_hit_count == 0
    assert evidence.unknown_hit_count == 0
    assert (
        evidence.paratext_or_mixed_rate
        == pytest.approx(
            2 / 3
        )
    )


def test_extractor_marks_transition_top_hit_as_mixed() -> None:
    query = "مجلس اول از کجا آغاز می‌شود؟"

    evidence = ParatextEvidenceExtractor(
        load_catalog()
    ).extract(
        hybrid_result(
            query,
            (
                MAJALIS_VERSION,
                25,
            ),
        )
    )

    assert (
        evidence.top_hit_role
        is ContentRole.MIXED
    )
    assert evidence.top_hit_is_mixed
    assert not evidence.top_hit_is_paratext
    assert (
        evidence.top_hit_is_structurally_non_authorial
    )


def test_extractor_marks_main_text_as_authorial() -> None:
    query = (
        "مولوی درباره فساد امت چه گفته است؟"
    )

    evidence = ParatextEvidenceExtractor(
        load_catalog()
    ).extract(
        hybrid_result(
            query,
            (
                MAJALIS_VERSION,
                30,
            ),
            (
                MAJALIS_VERSION,
                35,
            ),
        )
    )

    assert (
        evidence.top_hit_role
        is ContentRole.AUTHORIAL
    )
    assert not evidence.top_hit_is_paratext
    assert not evidence.top_hit_is_mixed
    assert not (
        evidence.top_hit_is_structurally_non_authorial
    )

    assert evidence.authorial_hit_count == 2
    assert (
        evidence.paratext_or_mixed_rate
        == 0.0
    )


def test_extractor_preserves_unknown_for_unreviewed_versions() -> None:
    version_id = (
        "0672JalalDinRumi."
        "Mathnawi.PDL00048-per1"
    )

    evidence = ParatextEvidenceExtractor(
        load_catalog()
    ).extract(
        hybrid_result(
            "پرسش",
            (
                version_id,
                100,
            ),
        )
    )

    assert (
        evidence.top_hit_role
        is ContentRole.UNKNOWN
    )
    assert evidence.unknown_hit_count == 1
    assert not (
        evidence.top_hit_is_structurally_non_authorial
    )


def test_extractor_supports_empty_results() -> None:
    evidence = ParatextEvidenceExtractor(
        load_catalog()
    ).extract(
        hybrid_result(
            "پرسش"
        )
    )

    assert evidence.evaluated_hit_count == 0
    assert evidence.hits == ()
    assert evidence.top_hit_role is None
    assert (
        evidence.paratext_or_mixed_rate
        == 0.0
    )


def test_only_first_ten_hits_are_evaluated() -> None:
    items = tuple(
        (
            MAJALIS_VERSION,
            26 + index,
        )
        for index in range(12)
    )

    evidence = ParatextEvidenceExtractor(
        load_catalog()
    ).extract(
        hybrid_result(
            "پرسش",
            *items,
        )
    )

    assert evidence.evaluated_hit_count == 10
    assert len(evidence.hits) == 10


def test_overlapping_or_gapped_zones_are_rejected(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        CATALOG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw["versions"][
        MAJALIS_VERSION
    ]["zones"][2][
        "start_ordinal"
    ] = 27

    bad_path = (
        tmp_path
        / "paratext_zones.yaml"
    )

    bad_path.write_text(
        yaml.safe_dump(
            raw,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ParatextCatalogError,
        match="contiguous",
    ):
        ParatextCatalog.from_yaml(
            bad_path
        )


def test_expected_passage_count_must_match_zone_end(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        CATALOG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw["versions"][
        MAJALIS_VERSION
    ]["expected_passage_count"] = 240

    bad_path = (
        tmp_path
        / "paratext_zones.yaml"
    )

    bad_path.write_text(
        yaml.safe_dump(
            raw,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ParatextCatalogError,
        match="expected_passage_count",
    ):
        ParatextCatalog.from_yaml(
            bad_path
        )


def test_evidence_is_frozen_and_validated() -> None:
    evidence = ParatextEvidenceExtractor(
        load_catalog()
    ).extract(
        hybrid_result(
            "پرسش",
            (
                MAJALIS_VERSION,
                13,
            ),
        )
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        evidence.top_hit_is_paratext = False  # type: ignore[misc]

    with pytest.raises(
        ValueError,
        match="top_hit_is_paratext",
    ):
        replace(
            evidence,
            top_hit_is_paratext=False,
        )
