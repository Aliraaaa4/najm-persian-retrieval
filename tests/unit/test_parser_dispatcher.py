"""Tests for profile-based parser dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from najm_retrieval.parsing.core import (
    split_openiti_source,
)
from najm_retrieval.parsing.dispatcher import (
    PARSER_HANDLERS,
    parse_source,
)
from najm_retrieval.parsing.profiles import (
    MIXED_PROSE_OCR,
    RAW_OCR_REFERENCE,
    STRUCTURED_POETRY,
)


def make_source(
    profile: str,
):
    """Create a minimal valid source for one parser profile."""

    if profile == STRUCTURED_POETRY:
        body = (
            "~~PageV01P001\r\n"
            "# بیت نخست %~% بیت دوم\r\n"
        )
    elif profile == MIXED_PROSE_OCR:
        body = (
            "PageV01P001\r\n"
            "# این یک بند نثر است.\r\n"
        )
    elif profile == RAW_OCR_REFERENCE:
        body = (
            "PageV00P001\r\n"
            "# heading-like OCR stays raw\r\n"
        )
    else:
        raise ValueError(
            f"Unsupported test profile: {profile}"
        )

    source_text = (
        "######OpenITI#\r\n"
        "#META# title: Dispatcher Test\r\n"
        "#META#Header#End#\r\n"
        + body
    )

    return split_openiti_source(
        source_text,
        source_path=Path(
            f"0001Author.Work.{profile}-per1"
        ),
    )


def test_registry_contains_all_supported_handlers() -> None:
    """Every production profile must have one handler."""

    assert set(PARSER_HANDLERS) == {
        STRUCTURED_POETRY,
        MIXED_PROSE_OCR,
        RAW_OCR_REFERENCE,
    }


@pytest.mark.parametrize(
    (
        "profile",
        "expected_parser_name",
    ),
    (
        (
            STRUCTURED_POETRY,
            "structured_poetry_state_machine",
        ),
        (
            MIXED_PROSE_OCR,
            "mixed_prose_ocr_state_machine",
        ),
        (
            RAW_OCR_REFERENCE,
            "conservative_raw_ocr_parser",
        ),
    ),
)
def test_dispatcher_selects_correct_handler(
    profile: str,
    expected_parser_name: str,
) -> None:
    """The profile must select its matching parser."""

    source = make_source(profile)

    document, metrics = parse_source(
        source=source,
        profile=profile,
    )

    assert document.profile == profile

    assert document.parser_name == (
        expected_parser_name
    )

    assert (
        document.reconstruct_body()
        == source.body_text
    )

    assert (
        metrics.passes_lossless_gate
        is True
    )


def test_dispatcher_is_deterministic() -> None:
    """Repeated parsing must produce equivalent output."""

    source = make_source(
        STRUCTURED_POETRY
    )

    first_document, first_metrics = (
        parse_source(
            source=source,
            profile=STRUCTURED_POETRY,
        )
    )

    second_document, second_metrics = (
        parse_source(
            source=source,
            profile=STRUCTURED_POETRY,
        )
    )

    assert (
        first_document.blocks
        == second_document.blocks
    )

    assert (
        first_document.reconstruct_body()
        == second_document.reconstruct_body()
    )

    assert (
        first_metrics.total_body_chars
        == second_metrics.total_body_chars
    )

    assert (
        first_metrics.covered_chars
        == second_metrics.covered_chars
    )


def test_dispatcher_rejects_unknown_profile() -> None:
    """Unknown profiles must fail before parsing."""

    source = make_source(
        MIXED_PROSE_OCR
    )

    with pytest.raises(
        ValueError,
        match="Unsupported parser profile",
    ):
        parse_source(
            source=source,
            profile="unknown_profile",
        )
