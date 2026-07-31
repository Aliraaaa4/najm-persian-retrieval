# Golden Parser Annotation Guide

## 1. Purpose

Golden annotations define the expected structural interpretation of
representative OpenITI source excerpts.

They are used to:

- evaluate parser block types
- evaluate exact block boundaries
- evaluate page, image, and milestone extraction
- evaluate verse and paragraph reconstruction
- detect gaps and overlaps
- verify exact source reconstruction
- compare parser candidates fairly

Golden annotations must describe the source text exactly as it exists.
They must not silently correct OCR errors, spelling, punctuation, or
whitespace.

## 2. Annotation location

Each parser sample stores its annotations in the top-level
`annotations` object:

```json
{
  "annotations": {
    "schema_version": 1,
    "status": "draft",
    "blocks": [],
    "notes": ""
  }
}