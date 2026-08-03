# REST API

## Overview

NAJM provides a local FastAPI interface for trusted passage retrieval over a
selected corpus of historical Persian texts.

The API uses lexical and dense retrieval, hybrid ranking, scope validation, and
an abstention policy. It returns source passages and does not generate answers
with a large language model.

## Schema version

The current public API schema version is:

```text
1.1.0
```

Version `1.1.0` adds deterministic query suggestions to abstained responses.

## Start the server

```powershell
python -m uvicorn najm_retrieval.api.app:app --host 127.0.0.1 --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## Endpoints

### Health

```http
GET /health
```

Checks whether the HTTP application is running.

### Readiness

```http
GET /ready
```

Returns HTTP `200` when retrieval artifacts are ready and HTTP `503` when the
runtime could not be initialized.

### Trusted retrieval

```http
POST /v1/retrieve
```

Request:

```json
{
  "query": "اختیار در مثنوی معنوی",
  "limit": 3
}
```

Request fields:

- `query`: required text from 1 through 1000 characters.
- `limit`: number of public results from 1 through 10; default is 10.

Unknown request fields are rejected.

## Accepted response

When trusted evidence is sufficient, `return_results` is `true`, `results`
contains public passages, and `suggestions` is empty.

Retriever scores are ranking values, not calibrated probabilities.

## Abstained response

When evidence is insufficient, the API still returns HTTP `200`, but:

- `action` is `abstain`;
- `return_results` is `false`;
- `results` is empty;
- `top_passage_id` is `null`;
- safe query suggestions may be returned.

Diagnostic passages are never exposed publicly.

Example suggestion:

```json
{
  "query": "فقط بر اساس متن اصلی دیوان شمس: درباره عشق چه می‌گوید؟",
  "label": "جست‌وجوی همین پرسش در دیوان شمس",
  "kind": "replace_out_of_scope",
  "entity_id": "0672JalalDinRumi.Diwan",
  "entity_kind": "work",
  "version_ids": [
    "0672JalalDinRumi.Diwan.PDL00047-per1"
  ]
}
```

## Query suggestions

Suggestions are deterministic and based on the frozen corpus scope catalog.
They are produced only for abstained responses, refer only to indexed works,
do not modify the abstention decision, and never expose diagnostic passages.

Suggestion kinds:

- `replace_out_of_scope`
- `restrict_scope`
- `search_primary_text`
- `scope_query`

## Abstention reasons

Possible values include:

- `known_out_of_corpus_scope`
- `source_attribution_conflict`
- `top_hit_paratext`
- `top_hit_mixed`
- `no_hybrid_hits`
- `weak_cross_retriever_evidence`
- `baseline_evidence_passed`

## PowerShell example

```powershell
$Body = @{
    query = "در مثنوی معنوی درباره اختیار چه آمده است؟"
    limit = 3
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/v1/retrieve" `
    -ContentType "application/json; charset=utf-8" `
    -Body $Body
```

## Errors

- HTTP `422`: invalid request body.
- HTTP `503`: retrieval runtime is unavailable.
- HTTP `500`: retrieval execution failed.

Internal exception details are not returned to clients.
