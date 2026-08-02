"""Tests for the corpus scope catalog and evidence extractor."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml

from najm_retrieval.retrieval import (
    CorpusScopeCatalog,
    HybridSearchHit,
    HybridSearchResult,
    ScopeCatalogError,
    ScopeEntityKind,
    ScopeEvidenceExtractor,
    normalize_scope_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "config"
    / "corpus_manifest.yaml"
)
ALIASES_PATH = (
    PROJECT_ROOT
    / "config"
    / "scope_aliases.yaml"
)


def load_catalog() -> CorpusScopeCatalog:
    return CorpusScopeCatalog.from_files(
        manifest_path=MANIFEST_PATH,
        aliases_path=ALIASES_PATH,
    )


def hybrid_hit(
    passage_id: str,
    version_id: str,
    *,
    rank: int,
) -> HybridSearchHit:
    return HybridSearchHit(
        passage_id=passage_id,
        version_id=version_id,
        kind="mixed_prose",
        rank=rank,
        fusion_score=0.05 / rank,
        lexical_rank=rank,
        dense_rank=rank,
        lexical_bm25_score=-5.0 + rank,
        dense_cosine_score=0.90 - rank / 100,
    )


def hybrid_result(
    query_text: str,
    *version_ids: str,
) -> HybridSearchResult:
    hits = tuple(
        hybrid_hit(
            f"p{rank}",
            version_id,
            rank=rank,
        )
        for rank, version_id in enumerate(
            version_ids,
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


def test_catalog_matches_manifest_scope() -> None:
    catalog = load_catalog()

    assert len(catalog.in_corpus_authors) == 4
    assert len(catalog.in_corpus_works) == 6

    assert {
        entity.entity_id
        for entity in catalog.in_corpus_works
    } == set(catalog.manifest.works)


def test_author_mention_resolves_to_all_indexed_author_versions() -> None:
    catalog = load_catalog()

    mentions = catalog.match_query(
        "مولوی درباره عشق چه گفته است؟"
    )

    rumi = next(
        mention
        for mention in mentions
        if mention.entity_id == "0672JalalDinRumi"
    )

    assert rumi.kind is ScopeEntityKind.AUTHOR
    assert rumi.in_corpus
    assert set(rumi.version_ids) == {
        "0672JalalDinRumi.Diwan.PDL00047-per1",
        "0672JalalDinRumi.MajalisSabica.AOCP202502141236-per1",
        "0672JalalDinRumi.Mathnawi.PDL00048-per1",
    }


def test_work_mention_resolves_to_one_indexed_version() -> None:
    catalog = load_catalog()

    mentions = catalog.match_query(
        "در مثنوی معنوی درباره اختیار چه آمده است؟"
    )

    work = next(
        mention
        for mention in mentions
        if mention.kind is ScopeEntityKind.WORK
    )

    assert (
        work.entity_id
        == "0672JalalDinRumi.Mathnawi"
    )
    assert work.version_ids == (
        "0672JalalDinRumi.Mathnawi.PDL00048-per1",
    )


def test_known_out_of_corpus_author_and_work_are_detected() -> None:
    catalog = load_catalog()

    mentions = catalog.match_query(
        "حافظ در دیوان حافظ درباره رندی چه می‌گوید؟"
    )

    assert {
        mention.entity_id
        for mention in mentions
        if not mention.in_corpus
    } == {
        "external:Hafez",
        "external:HafezDivan",
    }


def test_normalization_handles_arabic_variants_and_joiners() -> None:
    assert normalize_scope_text(
        "خواجه نصيرالدين طوسي"
    ) == normalize_scope_text(
        "خواجه نصیرالدین طوسی"
    )

    catalog = load_catalog()

    mentions = catalog.match_query(
        "خواجه نصيرالدين طوسي درباره اخلاق"
    )

    assert any(
        mention.entity_id
        == "0672NasirDinTusi"
        for mention in mentions
    )


def test_query_without_named_scope_has_no_mentions() -> None:
    catalog = load_catalog()

    assert catalog.match_query(
        "این بیت درباره چه مفهومی است؟"
    ) == ()


def test_scope_evidence_accepts_matching_top_source() -> None:
    query = "مولوی در مثنوی معنوی درباره اختیار چه می‌گوید؟"

    result = hybrid_result(
        query,
        "0672JalalDinRumi.Mathnawi.PDL00048-per1",
        "0672JalalDinRumi.Diwan.PDL00047-per1",
    )

    evidence = ScopeEvidenceExtractor(
        load_catalog()
    ).extract(
        query_text=query,
        hybrid_result=result,
    )

    assert evidence.explicit_scope
    assert evidence.in_corpus_scope_mentioned
    assert not evidence.known_out_of_corpus_scope_mentioned
    assert evidence.requested_work_ids == (
        "0672JalalDinRumi.Mathnawi",
    )
    assert evidence.requested_version_ids == (
        "0672JalalDinRumi.Mathnawi.PDL00048-per1",
    )
    assert evidence.top_hit_matches_requested_scope is True
    assert not evidence.source_attribution_conflict
    assert evidence.matching_hit_count_at_10 == 1
    assert evidence.matching_hit_rate_at_10 == 0.5


def test_work_scope_overrides_broader_author_scope() -> None:
    query = "مولوی در مثنوی درباره اختیار چه گفته است؟"

    result = hybrid_result(
        query,
        "0672JalalDinRumi.Diwan.PDL00047-per1",
    )

    evidence = ScopeEvidenceExtractor(
        load_catalog()
    ).extract(
        query_text=query,
        hybrid_result=result,
    )

    assert evidence.requested_author_ids == (
        "0672JalalDinRumi",
    )
    assert evidence.requested_work_ids == (
        "0672JalalDinRumi.Mathnawi",
    )
    assert evidence.top_hit_matches_requested_scope is False
    assert evidence.source_attribution_conflict


def test_known_ooc_scope_is_strong_evidence_without_version_target() -> None:
    query = "حافظ درباره رندی چه گفته است؟"

    result = hybrid_result(
        query,
        "0667BabaAfzal.Diwan.PDL00046-per1",
    )

    evidence = ScopeEvidenceExtractor(
        load_catalog()
    ).extract(
        query_text=query,
        hybrid_result=result,
    )

    assert evidence.explicit_scope
    assert evidence.known_out_of_corpus_scope_mentioned
    assert evidence.known_out_of_corpus_entity_ids == (
        "external:Hafez",
    )
    assert evidence.requested_version_ids == ()
    assert evidence.top_hit_matches_requested_scope is None
    assert not evidence.source_attribution_conflict
    assert evidence.matching_hit_rate_at_10 is None


def test_author_scope_detects_wrong_top_author() -> None:
    query = "خواجه نصیر درباره عدالت چه گفته است؟"

    result = hybrid_result(
        query,
        "0672JalalDinRumi.Diwan.PDL00047-per1",
    )

    evidence = ScopeEvidenceExtractor(
        load_catalog()
    ).extract(
        query_text=query,
        hybrid_result=result,
    )

    assert evidence.top_hit_matches_requested_scope is False
    assert evidence.source_attribution_conflict


def test_query_and_hybrid_result_must_align() -> None:
    extractor = ScopeEvidenceExtractor(
        load_catalog()
    )

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        extractor.extract(
            query_text="پرسش اول",
            hybrid_result=hybrid_result(
                "پرسش دوم"
            ),
        )


def test_reference_only_version_is_not_a_scope_target() -> None:
    catalog = load_catalog()

    entity = catalog.get_entity(
        "0670IbnAscadHanati.Masalik"
    )

    assert entity.version_ids == (
        "0670IbnAscadHanati.Masalik.AOCP202605141788-per1",
    )


def test_missing_manifest_work_alias_is_rejected(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        ALIASES_PATH.read_text(
            encoding="utf-8"
        )
    )

    del raw["in_corpus"]["works"][
        "0672JalalDinRumi.Mathnawi"
    ]

    bad_aliases = (
        tmp_path
        / "scope_aliases.yaml"
    )

    bad_aliases.write_text(
        yaml.safe_dump(
            raw,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ScopeCatalogError,
        match="exactly match manifest works",
    ):
        CorpusScopeCatalog.from_files(
            manifest_path=MANIFEST_PATH,
            aliases_path=bad_aliases,
        )


def test_normalized_alias_collision_is_rejected(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        ALIASES_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw["known_out_of_corpus"]["authors"][
        "external:Hafez"
    ]["aliases"].append(
        "مولوی"
    )

    bad_aliases = (
        tmp_path
        / "scope_aliases.yaml"
    )

    bad_aliases.write_text(
        yaml.safe_dump(
            raw,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ScopeCatalogError,
        match="alias collision",
    ):
        CorpusScopeCatalog.from_files(
            manifest_path=MANIFEST_PATH,
            aliases_path=bad_aliases,
        )


def test_scope_evidence_is_frozen() -> None:
    query = "مولوی در مثنوی درباره اختیار"

    evidence = ScopeEvidenceExtractor(
        load_catalog()
    ).extract(
        query_text=query,
        hybrid_result=hybrid_result(
            query,
            "0672JalalDinRumi.Mathnawi.PDL00048-per1",
        ),
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        evidence.explicit_scope = False  # type: ignore[misc]

    with pytest.raises(
        ValueError,
        match="top_hit_matches_requested_scope",
    ):
        replace(
            evidence,
            top_hit_matches_requested_scope=False,
        )
