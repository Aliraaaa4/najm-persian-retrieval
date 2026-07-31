"""Unicode-level normalization utilities."""

from __future__ import annotations

import re
import unicodedata


# Preserve horizontal tab (\u0009) and newline (\u000A).
# Other C0/C1 control characters are removed.
_CONTROL_CHARS_PATTERN = re.compile(
    r"[\u0000-\u0008\u000B-\u000C"
    r"\u000E-\u001F\u007F-\u009F]"
)

_HORIZONTAL_SPACE_PATTERN = re.compile(
    r"[ \t]+"
)

# Remove invisible formatting characters, but preserve:
# ZWNJ (\u200C) and ZWJ (\u200D).
_INVISIBLE_FORMAT_PATTERN = re.compile(
    r"[\u200B\u200E\u200F\u2060\uFEFF]"
)


def normalize_unicode(
    text: str,
) -> str:
    """Apply language-neutral Unicode normalization.

    The operation:
    - applies NFKC;
    - converts CRLF and CR to LF;
    - removes unsafe control characters;
    - removes selected invisible formatting characters;
    - collapses horizontal spaces per line;
    - preserves internal line and blank-line boundaries.

    Persian- or Arabic-specific characters are not converted here.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = (
        normalized
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    normalized = _CONTROL_CHARS_PATTERN.sub(
        "",
        normalized,
    )

    normalized = _INVISIBLE_FORMAT_PATTERN.sub(
        "",
        normalized,
    )

    normalized_lines: list[str] = []

    for line in normalized.split("\n"):
        line = _HORIZONTAL_SPACE_PATTERN.sub(
            " ",
            line,
        )

        normalized_lines.append(
            line.strip()
        )

    return "\n".join(
        normalized_lines
    ).strip()