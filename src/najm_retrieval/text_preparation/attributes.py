"""Adapters for parser attributes stored in Python or JSON form."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def attributes_to_dict(
    value: object,
) -> dict[str, Any]:
    """Convert serialized parser attributes to a dictionary.

    Parser models store attributes as tuples of key-value pairs.
    JSON serialization converts those tuples to lists. This
    function accepts both forms, as well as an ordinary mapping.

    Args:
        value:
            None, a mapping, or a sequence of two-item pairs.

    Returns:
        A new dictionary with string keys.

    Raises:
        TypeError:
            If the top-level value has an unsupported type.

        ValueError:
            If an attribute entry is malformed or a key occurs
            more than once.
    """

    if value is None:
        return {}

    if isinstance(value, Mapping):
        return {
            str(key): item_value
            for key, item_value in value.items()
        }

    if (
        isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        )
        or not isinstance(value, Sequence)
    ):
        raise TypeError(
            "attributes must be a mapping, "
            "a sequence of key-value pairs, or None."
        )

    result: dict[str, Any] = {}

    for index, item in enumerate(value):
        if (
            isinstance(
                item,
                (
                    str,
                    bytes,
                    bytearray,
                ),
            )
            or not isinstance(item, Sequence)
            or len(item) != 2
        ):
            raise ValueError(
                "Each attribute entry must contain "
                f"exactly two values; invalid entry "
                f"at index {index}."
            )

        key, item_value = item
        key_text = str(key)

        if key_text in result:
            raise ValueError(
                f"Duplicate attribute key: {key_text!r}."
            )

        result[key_text] = item_value

    return result
