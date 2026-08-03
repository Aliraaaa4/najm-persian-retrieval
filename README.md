# Historical Persian Text Retrieval System

A local, source-grounded retrieval system for historical Persian and Islamic
texts, developed for the NAJM Junior AI Engineer (NLP) take-home project.

The system retrieves relevant passages from a selected subset of the
OpenITI/KITAB corpus. It combines lexical and dense retrieval, applies hybrid
ranking and trust-aware abstention, and returns passages with explicit source
metadata. It does not use a generative language model to produce answers.

## Project status

The repository contains a tested local retrieval prototype with:

- marker-aware OpenITI parsing;
- Persian and Arabic text normalization;
- structure-aware passage construction;
- SQLite FTS5 lexical retrieval;
- multilingual dense retrieval;
- weighted hybrid ranking;
- corpus-scope and source validation;
- trusted abstention;
- deterministic query suggestions;
- a FastAPI REST interface;
- a Persian right-to-left demonstration UI.

The full automated test suite currently contains 451 passing tests.

## Indexed corpus

The project uses the specified subset of the OpenITI `PER0675AH` repository.

Indexed works:

- Baba Afzal â€” Diwan
- Ibn As'ad Hanati â€” Masalik
- Jalal al-Din Rumi â€” Diwan
- Jalal al-Din Rumi â€” Majalis-i Sab'a
- Jalal al-Din Rumi â€” Mathnawi
- Nasir al-Din Tusi â€” Akhlaq-i Muhtashami

The exact selected versions, quality profiles, and inclusion decisions are
recorded in:

```text
config/corpus_manifest.yaml
```

One additional OCR version of Ibn As'ad Hanati's Masalik is retained as a
reference-only source and is not included in the public retrieval index.

## Architecture

The implemented pipeline contains the following stages:

1. Corpus manifest and source-file audit
2. OpenITI parsing and metadata extraction
3. Unicode and text normalization
4. Logical-unit and boundary analysis
5. Structure-aware passage construction
6. SQLite FTS5 lexical indexing
7. Multilingual dense embedding
8. Weighted reciprocal-rank fusion
9. Scope and passage-kind validation
10. Trusted abstention
11. Passage-store enrichment
12. FastAPI REST service
13. Persian right-to-left demonstration interface

See [`docs/engineering_decisions.md`](docs/engineering_decisions.md) for the
architecture, technical decisions, trade-offs, limitations, and future work.

## Requirements

- Python 3.11
- Windows, Linux, or macOS
- CPU execution supported
- Approximately 8 GB of RAM is sufficient for the submitted corpus
- Internet access may be needed once to download the embedding model

The project uses only open-source libraries and models. No paid service or
proprietary API is required.

## Installation

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Installing the repository in editable mode is required because some integration
tests execute project scripts in separate Python processes.

## Obtain the dataset

Clone the pinned OpenITI repository into the expected local directory:

```powershell
git clone `
    https://github.com/OpenITI/PER0675AH.git `
    .\data\raw\PER0675AH
```

Check out the corpus revision recorded in the manifest:

```powershell
git -C .\data\raw\PER0675AH `
    checkout 7fff2ba2e59a086477b2c37b4d1d2f7f06e12929
```

The expected corpus data directory is:

```text
data/raw/PER0675AH/data
```

The repository does not commit the original OpenITI corpus files.

## Audit the corpus

Validate the configured source files and write a JSON audit report:

```powershell
python .\scripts\audit_corpus.py `
    --manifest .\config\corpus_manifest.yaml `
    --corpus-root .\data\raw\PER0675AH\data `
    --output .\data\processed\corpus_audit.json
```

The audit checks configured works, versions, files, and source metadata before
the parsing and indexing stages are allowed to continue.

## Parse the corpus

Parse the selected OpenITI versions into structured lossless JSON outputs:

```powershell
python .\scripts\parse_corpus.py `
    --manifest .\config\corpus_manifest.yaml `
    --corpus-root .\data\raw\PER0675AH\data `
    --output-dir .\data\processed\parser
```

The parser preserves document and version identifiers and separates metadata,
primary text, structural markers, and non-primary material.

## Build the indexes

The submitted system uses three runtime artifacts:

1. A SQLite FTS5 lexical index
2. A SQLite passage store
3. A dense embedding artifact

The lexical index and passage store have deterministic Python builders:

```python
from najm_retrieval.retrieval import (
    build_lexical_index,
    build_passage_store,
)

build_lexical_index(
    passage_root="PATH_TO_PREPARED_PASSAGE_JSONL_ROOT",
    database_path="artifacts/runtime/lexical.sqlite3",
    overwrite=True,
)

build_passage_store(
    "PATH_TO_PREPARED_PASSAGE_JSONL_ROOT",
    corpus_manifest_path="config/corpus_manifest.yaml",
    scope_aliases_path="config/scope_aliases.yaml",
    output_path="artifacts/runtime/passage_store.sqlite3",
)
```

The complete raw-corpus-to-dense-artifact process is not yet exposed as one
supported public CLI command. The submitted release therefore provides a
validated runtime bundle containing all required indexes. Turning the complete
pipeline into one configuration-driven build command is listed as future work.

For a reliable evaluation or handoff, use the published runtime artifacts
described below.

## Runtime artifacts

The API requires the prepared lexical index, passage store, and dense artifact.

Download these two files from the latest GitHub Release:

```text
najm-runtime-artifacts-v1.zip
najm-runtime-artifacts-v1.zip.sha256
```

Place both files in the repository root.

Verify the archive on Windows:

```powershell
$actualHash = (
    Get-FileHash `
        .\najm-runtime-artifacts-v1.zip `
        -Algorithm SHA256
).Hash.ToLower()

$expectedHash = (
    Get-Content `
        .\najm-runtime-artifacts-v1.zip.sha256
).Split()[0].ToLower()

if ($actualHash -ne $expectedHash) {
    throw "Runtime artifact checksum mismatch."
}

Write-Host "RUNTIME ARTIFACT HASH OK"
```

Extract the archive:

```powershell
Expand-Archive `
    -LiteralPath .\najm-runtime-artifacts-v1.zip `
    -DestinationPath . `
    -Force
```

Cross-platform extraction:

```bash
python -m zipfile -e najm-runtime-artifacts-v1.zip .
```

Expected layout:

```text
artifacts/runtime/corpus-ad111acd912e/
â”œâ”€â”€ lexical.sqlite3
â”œâ”€â”€ passage_store.sqlite3
â””â”€â”€ dense/
    â””â”€â”€ intfloat__multilingual-e5-small/
        â”œâ”€â”€ artifact_manifest.json
        â”œâ”€â”€ embeddings.npy
        â”œâ”€â”€ metadata.json
        â”œâ”€â”€ passages.jsonl
        â””â”€â”€ pilot_evaluation.json
```

Maintainers can recreate the ZIP from already validated local artifacts:

```powershell
python .\scripts\build_runtime_bundle.py --force
```

This command packages existing validated indexes; it does not rebuild the dense
embeddings from the raw corpus.

## Embedding model and offline mode

Dense retrieval uses the open-source model:

```text
intfloat/multilingual-e5-small
```

On a fresh computer, the first retrieval may download the model and store it in
the local Hugging Face cache.

After the model is cached, fully local loading can be enforced:

```powershell
$env:NAJM_DENSE_LOCAL_FILES_ONLY = "true"
```

The embedding model weights are not included in the runtime ZIP in order to
keep the release artifact small.

## Start the API and demo UI

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1

python -m uvicorn `
    najm_retrieval.api.app:app `
    --host 127.0.0.1 `
    --port 8000
```

Open the Persian demonstration interface:

```text
http://127.0.0.1:8000/
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

The first retrieval may take longer while the dense model is loaded or
downloaded.

## Example API request

Endpoint:

```http
POST /v1/retrieve
```

Request body:

```json
{
  "query": "Ø¯Ø± Ù…Ø«Ù†ÙˆÛŒ Ù…Ø¹Ù†ÙˆÛŒ Ø¯Ø±Ø¨Ø§Ø±Ù‡ Ø§Ø®ØªÛŒØ§Ø± Ú†Ù‡ Ø¢Ù…Ø¯Ù‡ Ø§Ø³ØªØŸ",
  "limit": 3
}
```

PowerShell example:

```powershell
$Body = @{
    query = "Ø¯Ø± Ù…Ø«Ù†ÙˆÛŒ Ù…Ø¹Ù†ÙˆÛŒ Ø¯Ø±Ø¨Ø§Ø±Ù‡ Ø§Ø®ØªÛŒØ§Ø± Ú†Ù‡ Ø¢Ù…Ø¯Ù‡ Ø§Ø³ØªØŸ"
    limit = 3
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/v1/retrieve" `
    -ContentType "application/json; charset=utf-8" `
    -Body $Body
```

When sufficient evidence is found, the response includes source passages with
fields such as author, title, work identifier, version identifier, passage
identifier, snippet, and ranking information.

When evidence is insufficient or the query explicitly refers to an
out-of-corpus source, the API returns an abstention response with no public
passages. It may also return safe query suggestions restricted to the indexed
corpus.

See [`docs/api.md`](docs/api.md) for the complete API contract.

## Assumptions

- The corpus scope is intentionally limited to the works specified in the
  take-home assignment.
- Canonical versions listed in `config/corpus_manifest.yaml` are used for public
  indexing.
- Reference-only versions are available for inspection but are excluded from
  public retrieval.
- Arabic quotations and headings may be genuine parts of Persian works and are
  preserved.
- Retriever scores are ranking signals and are not calibrated probabilities.
- The API returns source passages rather than generating synthesized answers.
- The runtime artifact and corpus manifest must belong to the same corpus
  identity and schema version.

## Known limitations

- The dense model is multilingual and is not fine-tuned for classical Persian,
  Persian poetry, historical spelling, or OpenITI OCR noise.
- Dense search uses exact NumPy similarity and is not intended for millions of
  passages.
- The current system does not use a cross-encoder reranker.
- Fusion weights and abstention thresholds were manually calibrated.
- The human-labeled evaluation set is limited.
- Parsing rules may not cover every OpenITI tagging variation.
- Index rebuilding is not incremental.
- The complete build pipeline is not exposed through one end-to-end CLI.
- The first run may need internet access to download the embedding model.
- The API does not include authentication, rate limiting, or production
  observability.
- The current UI is a demonstration interface.
- Full runtime validation was primarily performed on Windows.

See [`docs/engineering_decisions.md`](docs/engineering_decisions.md) for
detailed trade-offs and the planned production-oriented roadmap.

## Evaluation and supporting documentation

- [Engineering decisions](docs/engineering_decisions.md)
- [REST API documentation](docs/api.md)
- [Parser evaluation](docs/parser_evaluation.md)
- [Golden annotation guide](docs/golden_annotation_guide.md)
- [Abstention validation](docs/evaluation/abstention_validation_v1.md)

## Tests

Run the complete automated test suite:

```powershell
python -m pytest -q
```

Expected result for the submitted version:

```text
451 passed
```

A fresh-clone validation was also performed in a new virtual environment with
the project installed in editable mode.

## Delivery

The repository is intended to remain private during evaluation. The evaluator
must be invited as a GitHub collaborator.

The submitted release contains the runtime bundle and its SHA-256 checksum.
