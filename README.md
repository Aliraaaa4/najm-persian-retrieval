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
python -m pip install -e .
```

## Runtime artifacts

The API needs the prepared lexical index, passage store, and dense index.
Download these two assets from the GitHub Release attached to `v0.1.0`:

- `najm-runtime-artifacts-v1.zip`
- `najm-runtime-artifacts-v1.zip.sha256`

Place both files in the repository root. On Windows, verify the archive:

```powershell
Get-FileHash .\najm-runtime-artifacts-v1.zip -Algorithm SHA256
Get-Content .\najm-runtime-artifacts-v1.zip.sha256
```

The two SHA-256 values must match. Extract the archive into the repository root:

```powershell
Expand-Archive `
    -LiteralPath .\najm-runtime-artifacts-v1.zip `
    -DestinationPath . `
    -Force
```

A cross-platform alternative is:

```bash
python -m zipfile -e najm-runtime-artifacts-v1.zip .
```

The extracted runtime layout is:

```text
artifacts/runtime/corpus-ad111acd912e/
├── lexical.sqlite3
├── passage_store.sqlite3
└── dense/intfloat__multilingual-e5-small/
```

The embedding model `intfloat/multilingual-e5-small` is loaded from the local
Hugging Face cache when available. On a fresh machine, the first retrieval may
download the model once and cache it for later local use.

For a fully offline run after the model is cached:

```powershell
$env:NAJM_DENSE_LOCAL_FILES_ONLY = "true"
```

Maintainers can rebuild the release bundle from validated local artifacts:

```powershell
python .\scripts\build_runtime_bundle.py --force
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

The first retrieval may take longer while the dense model is loaded or
downloaded into the local cache.

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
