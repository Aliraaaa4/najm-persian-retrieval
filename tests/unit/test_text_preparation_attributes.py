"""Tests for parser attribute adapters."""

from __future__ import annotations

import pytest

from najm_retrieval.text_preparation.attributes import (
    attributes_to_dict,
)


def test_returns_empty_dictionary_for_none() -> None:
    assert attributes_to_dict(None) == {}


def test_converts_mapping_to_new_dictionary() -> None:
    source = {
        "group_id": "verse_0001",
        "continuation": True,
    }

    result = attributes_to_dict(source)

    assert result == source
    assert result is not source


def test_converts_json_list_pairs() -> None:
    source = [
        ["group_id", "verse_0001"],
        ["verse_number", 1],
        ["continuation", True],
    ]

    assert attributes_to_dict(source) == {
        "group_id": "verse_0001",
        "verse_number": 1,
        "continuation": True,
    }


def test_converts_internal_tuple_pairs() -> None:
    source = (
        ("group_id", "paragraph_0001"),
        ("continuation", False),
    )

    assert attributes_to_dict(source) == {
        "group_id": "paragraph_0001",
        "continuation": False,
    }


def test_converts_non_string_keys_to_strings() -> None:
    assert attributes_to_dict(
        [[1, "value"]]
    ) == {
        "1": "value",
    }


def test_rejects_malformed_attribute_entry() -> None:
    with pytest.raises(
        ValueError,
        match="exactly two values",
    ):
        attributes_to_dict(
            [["group_id"]]
        )


def test_rejects_duplicate_attribute_keys() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate attribute key",
    ):
        attributes_to_dict(
            [
                ["group_id", "verse_0001"],
                ["group_id", "verse_0002"],
            ]
        )


def test_rejects_mapping_keys_that_collide_after_conversion() -> None:
    source = {
        1: "numeric key",
        "1": "string key",
    }

    with pytest.raises(
        ValueError,
        match="Duplicate attribute key",
    ):
        attributes_to_dict(source)


def test_rejects_unsupported_top_level_value() -> None:
    with pytest.raises(
        TypeError,
        match="attributes must be",
    ):
        attributes_to_dict(
            123
        )
