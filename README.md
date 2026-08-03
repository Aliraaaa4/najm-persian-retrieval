# Historical Persian Text Retrieval System

A local semantic retrieval system for historical Persian and Islamic texts,
developed as part of the NAJM Junior AI Engineer (NLP) take-home project.

## Status

The repository contains an end-to-end local retrieval prototype with corpus
parsing, normalization, passage construction, lexical and dense indexing,
hybrid retrieval, trust-aware abstention, deterministic query suggestions, and
a FastAPI REST interface.

## Implemented Pipeline

1. Marker-aware OpenITI document parsing
2. Persian and Arabic text normalization
3. Structure-aware passage construction
4. SQLite FTS5 lexical indexing
5. Multilingual dense embedding and retrieval
6. Weighted hybrid retrieval
7. Trust-aware abstention and scope validation
8. Deterministic query suggestions
9. REST API using FastAPI

See [`docs/api.md`](docs/api.md) for the implemented API contract.

## Requirements

- Python 3.11
- Windows, Linux, or macOS
- CPU execution supported

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt
