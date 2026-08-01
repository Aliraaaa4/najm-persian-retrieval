"""Load and search one generated dense-retrieval artifact."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Sequence
import json

import numpy as np

from najm_retrieval.retrieval.dense_models import (
    DENSE_INDEX_SCHEMA_VERSION,
    DenseSearchHit,
    DenseSearchResult,
)


class DenseIndexError(RuntimeError):
    """Raised when a dense index cannot be loaded or searched."""


class _Encoder(Protocol):
    """Minimum encoder interface needed by DenseIndex."""

    def encode(
        self,
        inputs: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        """Encode text inputs."""


@dataclass(frozen=True)
class _PassageRecord:
    row_index: int
    passage_id: str
    version_id: str
    kind: str


class DenseIndex:
    """Read and search a normalized dense index stored as NumPy arrays."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        encoder: _Encoder | None = None,
        device: str = "cpu",
        local_files_only: bool = True,
        verify_hashes: bool = True,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.device = device
        self.local_files_only = local_files_only
        self._encoder = encoder

        if not self.artifact_root.is_dir():
            raise DenseIndexError(
                f"Dense artifact does not exist: {self.artifact_root}"
            )

        self._metadata_path = self.artifact_root / "metadata.json"
        self._embeddings_path = self.artifact_root / "embeddings.npy"
        self._passages_path = self.artifact_root / "passages.jsonl"

        for path in (
            self._metadata_path,
            self._embeddings_path,
            self._passages_path,
        ):
            if not path.is_file():
                raise DenseIndexError(f"Dense artifact file is missing: {path}")

        self._metadata = self._load_metadata()

        if verify_hashes:
            self._validate_file_hash(
                self._embeddings_path,
                "embeddings_sha256",
            )
            self._validate_file_hash(
                self._passages_path,
                "passages_sha256",
            )

        self._records = self._load_passage_records()
        self._embeddings = self._load_embeddings()
        self._validate_passage_id_digest()

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""

        return str(self._metadata["model_name"])

    @property
    def query_prefix(self) -> str:
        """Return the prefix expected by the embedding model."""

        return str(self._metadata["query_prefix"])

    @property
    def passage_count(self) -> int:
        """Return the number of indexed passages."""

        return int(self._metadata["passage_count"])

    @property
    def embedding_dimension(self) -> int:
        """Return the dense-vector dimension."""

        return int(self._metadata["embedding_dimension"])

    def search(
        self,
        query_text: str,
        *,
        limit: int = 10,
    ) -> DenseSearchResult:
        """Encode one query and return deterministic cosine-ranked hits."""

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must not be empty.")

        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")

        started_at = perf_counter()
        cleaned_query = query_text.strip()
        encoder = self._get_encoder()

        encoded = encoder.encode(
            [self.query_prefix + cleaned_query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(encoded, dtype=np.float32)

        if query_embedding.shape == (self.embedding_dimension,):
            query_embedding = query_embedding.reshape(1, -1)

        expected_shape = (1, self.embedding_dimension)

        if query_embedding.shape != expected_shape:
            raise DenseIndexError(
                "Encoder returned an unexpected shape: "
                f"{query_embedding.shape}; expected {expected_shape}."
            )

        if not np.isfinite(query_embedding).all():
            raise DenseIndexError("Query embedding contains NaN or Inf.")

        query_vector = query_embedding[0]
        query_norm = float(np.linalg.norm(query_vector))

        if query_norm <= 0:
            raise DenseIndexError("Query embedding has zero norm.")

        query_vector = query_vector / query_norm
        scores = np.asarray(
            self._embeddings @ query_vector,
            dtype=np.float32,
        )

        top_indexes = _top_k_indexes(
            scores,
            min(limit, self.passage_count),
        )

        hits = tuple(
            DenseSearchHit(
                passage_id=self._records[int(index)].passage_id,
                version_id=self._records[int(index)].version_id,
                kind=self._records[int(index)].kind,
                rank=rank,
                cosine_score=float(scores[int(index)]),
            )
            for rank, index in enumerate(top_indexes, start=1)
        )

        latency_ms = (perf_counter() - started_at) * 1000.0

        return DenseSearchResult(
            query_text=query_text,
            model_name=self.model_name,
            hits=hits,
            latency_ms=latency_ms,
        )

    def _load_metadata(self) -> dict[str, Any]:
        try:
            metadata = json.loads(
                self._metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise DenseIndexError(
                f"Cannot read dense metadata: {self._metadata_path}"
            ) from error

        if not isinstance(metadata, dict):
            raise DenseIndexError("Dense metadata must be a JSON object.")

        required = {
            "schema_version",
            "model_name",
            "query_prefix",
            "passage_prefix",
            "normalized_embeddings",
            "similarity",
            "dtype",
            "embedding_dimension",
            "passage_count",
            "passage_ids_sha256",
            "embeddings_sha256",
            "passages_sha256",
        }

        missing = sorted(required - metadata.keys())

        if missing:
            raise DenseIndexError(
                "Dense metadata is missing fields: " + ", ".join(missing)
            )

        if metadata["schema_version"] != DENSE_INDEX_SCHEMA_VERSION:
            raise DenseIndexError(
                "Unsupported dense-index schema version: "
                f"{metadata['schema_version']}"
            )

        for name in ("model_name", "query_prefix", "passage_prefix"):
            value = metadata[name]
            if not isinstance(value, str) or not value:
                raise DenseIndexError(
                    f"Dense metadata field {name!r} must be non-empty."
                )

        if metadata["normalized_embeddings"] is not True:
            raise DenseIndexError("Dense embeddings must be normalized.")

        if metadata["similarity"] != "cosine_via_inner_product":
            raise DenseIndexError("Unsupported dense similarity method.")

        if metadata["dtype"] != "float32":
            raise DenseIndexError("Dense index dtype must be float32.")

        for name in ("embedding_dimension", "passage_count"):
            value = metadata[name]
            if not isinstance(value, int) or value < 1:
                raise DenseIndexError(
                    f"Dense metadata field {name!r} must be positive."
                )

        for name in (
            "passage_ids_sha256",
            "embeddings_sha256",
            "passages_sha256",
        ):
            value = metadata[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DenseIndexError(
                    f"Dense metadata field {name!r} is not a SHA-256 digest."
                )

        return metadata

    def _load_passage_records(self) -> tuple[_PassageRecord, ...]:
        records: list[_PassageRecord] = []
        passage_ids: set[str] = set()

        try:
            with self._passages_path.open("r", encoding="utf-8") as file:
                for expected_index, line in enumerate(file):
                    if not line.strip():
                        continue

                    payload = json.loads(line)

                    if not isinstance(payload, dict):
                        raise DenseIndexError(
                            "Each passages.jsonl row must be an object."
                        )

                    row_index = payload.get("row_index")
                    passage_id = payload.get("passage_id")
                    version_id = payload.get("version_id")
                    kind = payload.get("kind")

                    if row_index != expected_index:
                        raise DenseIndexError(
                            "Dense passage row indexes must be contiguous "
                            "and start at zero."
                        )

                    for name, value in (
                        ("passage_id", passage_id),
                        ("version_id", version_id),
                        ("kind", kind),
                    ):
                        if not isinstance(value, str) or not value.strip():
                            raise DenseIndexError(
                                f"Dense passage field {name!r} is invalid."
                            )

                    if passage_id in passage_ids:
                        raise DenseIndexError(
                            f"Duplicate dense passage ID: {passage_id}"
                        )

                    passage_ids.add(passage_id)
                    records.append(
                        _PassageRecord(
                            row_index=row_index,
                            passage_id=passage_id,
                            version_id=version_id,
                            kind=kind,
                        )
                    )
        except DenseIndexError:
            raise
        except (OSError, json.JSONDecodeError) as error:
            raise DenseIndexError(
                f"Cannot read dense passage mapping: {self._passages_path}"
            ) from error

        if len(records) != self.passage_count:
            raise DenseIndexError(
                "Dense passage count does not match metadata: "
                f"{len(records)} != {self.passage_count}."
            )

        return tuple(records)

    def _load_embeddings(self) -> np.ndarray:
        try:
            embeddings = np.load(
                self._embeddings_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise DenseIndexError(
                f"Cannot load dense embeddings: {self._embeddings_path}"
            ) from error

        expected_shape = (
            self.passage_count,
            self.embedding_dimension,
        )

        if embeddings.shape != expected_shape:
            raise DenseIndexError(
                "Dense embedding shape does not match metadata: "
                f"{embeddings.shape} != {expected_shape}."
            )

        if embeddings.dtype != np.float32:
            raise DenseIndexError(
                f"Dense embeddings must use float32, found {embeddings.dtype}."
            )

        for start in range(0, self.passage_count, 2048):
            batch = embeddings[start : start + 2048]

            if not np.isfinite(batch).all():
                raise DenseIndexError("Dense embeddings contain NaN or Inf.")

            norms = np.linalg.norm(batch, axis=1)

            if not np.allclose(norms, 1.0, atol=1e-5):
                raise DenseIndexError("Dense embeddings are not normalized.")

        return embeddings

    def _validate_file_hash(self, path: Path, metadata_key: str) -> None:
        actual = _sha256_file(path)
        expected = str(self._metadata[metadata_key])

        if actual != expected:
            raise DenseIndexError(
                f"Dense artifact hash mismatch for {path.name}."
            )

    def _validate_passage_id_digest(self) -> None:
        digest = sha256()

        for record in self._records:
            digest.update(record.passage_id.encode("utf-8"))
            digest.update(b"\n")

        if digest.hexdigest() != self._metadata["passage_ids_sha256"]:
            raise DenseIndexError("Dense passage-ID digest mismatch.")

    def _get_encoder(self) -> _Encoder:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise DenseIndexError(
                    "Dense search requires the 'dense' optional dependency. "
                    "Install with: pip install -e \".[dense]\""
                ) from error

            self._encoder = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )

        dimension_getter = getattr(
            self._encoder,
            "get_embedding_dimension",
            None,
        )

        if callable(dimension_getter):
            actual_dimension = int(dimension_getter())

            if actual_dimension != self.embedding_dimension:
                raise DenseIndexError(
                    "Encoder dimension does not match the dense artifact: "
                    f"{actual_dimension} != {self.embedding_dimension}."
                )

        return self._encoder


def _top_k_indexes(
    scores: np.ndarray,
    limit: int,
) -> np.ndarray:
    """Return deterministic top-k indexes without a full-array sort."""

    if limit == len(scores):
        candidates = np.arange(len(scores))
    else:
        partition_start = len(scores) - limit
        partition = np.argpartition(scores, partition_start)[partition_start:]
        threshold = float(scores[partition].min())
        candidates = np.flatnonzero(scores >= threshold)

    ordering = np.lexsort(
        (
            candidates,
            -scores[candidates],
        )
    )

    return candidates[ordering[:limit]]


def _sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


__all__ = [
    "DenseIndex",
    "DenseIndexError",
]
