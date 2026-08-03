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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the demo UI

From the project root, activate the virtual environment and start the local
FastAPI server:

```powershell
Set-Location D:\projects\najm-retrieval
.\.venv\Scripts\Activate.ps1
python -m uvicorn najm_retrieval.api.app:app --host 127.0.0.1 --port 8000
```

Then open the browser interface:

```text
http://127.0.0.1:8000/
```

The interface supports Persian right-to-left search, source-aware result cards,
trusted abstention, and clickable query suggestions.

The first request may take longer while the local dense model and indexes are
loaded.

## API documentation

Swagger UI is available while the server is running:

```text
http://127.0.0.1:8000/docs
```

See [`docs/api.md`](docs/api.md) for the public API contract.

## Tests

Run the full test suite:

```powershell
python -m pytest -q
```
