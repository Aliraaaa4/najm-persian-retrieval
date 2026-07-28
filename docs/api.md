# API Documentation

## Overview

This document describes the REST API for the Historical Persian Text Retrieval System.

The API provides access to a local semantic search engine built for retrieving relevant passages from historical Persian and Islamic texts. The current document contains the planned API structure and will be updated as the implementation progresses.

## Base URL

```text
http://127.0.0.1:8000
```

## Interactive Documentation

After starting the API server, the automatically generated Swagger documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

The alternative ReDoc documentation will be available at:

```text
http://127.0.0.1:8000/redoc
```

## Planned Endpoints

### Health Check

```http
GET /health
```

This endpoint checks whether the API service is running correctly.

#### Example Request

```http
GET http://127.0.0.1:8000/health
```

#### Planned Response

```json
{
  "status": "ok"
}
```

---

### Semantic Search

```http
GET /search
```

This endpoint receives a natural-language query and returns the most relevant passages from the indexed historical texts.

#### Query Parameters

- `q`: The search query entered by the user.
- `top_k`: The maximum number of search results to return. The default value will be `5`.

#### Example Request

```http
GET http://127.0.0.1:8000/search?q=صبر در آثار مولانا&top_k=5
```

#### Planned Response

```json
{
  "query": "صبر در آثار مولانا",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "score": 0.87,
      "author": "Jalal al-Din Rumi",
      "title": "Mathnawi",
      "document_id": "rumi_mathnawi",
      "passage_id": "rumi_mathnawi_000123",
      "snippet": "A relevant passage from the retrieved document."
    }
  ]
}
```

## Search Result Fields

Each search result is planned to contain the following information:

- `rank`: The position of the result in the ranked result list.
- `score`: The semantic similarity score assigned to the passage.
- `author`: The author of the source document.
- `title`: The title of the source document.
- `document_id`: A unique identifier for the source document.
- `passage_id`: A unique identifier for the retrieved passage.
- `snippet`: A short excerpt from the retrieved text.

## Error Responses

The API will return an appropriate HTTP error response when the request is invalid or when the retrieval index is unavailable.

A planned validation error response may look like this:

```json
{
  "detail": "The search query must not be empty."
}
```

## Notes

- The API is designed to run locally.
- The search endpoint will use semantic retrieval rather than exact keyword matching.
- The API does not generate answers using a large language model.
- The returned results are passages retrieved directly from the indexed source texts.
- Endpoint details and response formats may be refined during implementation.

## Implementation Status

The API is currently under development. This document will be updated after the search service, response schemas, validation rules, and error-handling behavior are fully implemented.