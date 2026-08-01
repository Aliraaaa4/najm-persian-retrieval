"""Tests for the memory-mapped dense retrieval index."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence
import json

import numpy as np
import pytest

from najm_retrieval.retrieval import (
    DenseIndex,
    DenseIndexError,
)


class _FakeEncoder:
    def __init__(
        self,
        vectors: dict[str, list[float]],
        *,
        dimension: int = 3,
    ) -> None:
        self.vectors = vectors
        self.dimension = dimension
        self.calls: list[str] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        inputs: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert batch_size == 1
        assert show_progress_bar is False
        assert convert_to_numpy is True
        assert normalize_embeddings is True

        self.calls.extend(inputs)

        return np.asarray(
            [self.vectors[text] for text in inputs],
            dtype=np.float32,
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_artifact(
    root: Path,
    *,
    row_indexes: tuple[int, ...] = (0, 1, 2),
    passage_count: int = 3,
) -> None:
    root.mkdir(parents=True)

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2**-0.5, 2**-0.5, 0.0],
        ],
        dtype=np.float32,
    )

    embeddings_path = root / "embeddings.npy"
    np.save(
        embeddings_path,
        embeddings,
        allow_pickle=False,
    )

    records = [
        {
            "row_index": row_indexes[0],
            "passage_id": "diwan:passage_000001",
            "version_id": "diwan",
            "kind": "diwan",
        },
        {
            "row_index": row_indexes[1],
            "passage_id": "mathnawi:passage_000001",
            "version_id": "mathnawi",
            "kind": "mathnawi",
        },
        {
            "row_index": row_indexes[2],
            "passage_id": "akhlaq:passage_000001",
            "version_id": "akhlaq",
            "kind": "mixed_prose",
        },
    ]

    passages_path = root / "passages.jsonl"
    passages_path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    passage_ids_digest = sha256(
        (
            "\n".join(
                record["passage_id"]
                for record in records
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()

    metadata = {
        "schema_version": "1.0.0",
        "model_name": "example/test-encoder",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "normalized_embeddings": True,
        "similarity": "cosine_via_inner_product",
        "dtype": "float32",
        "embedding_dimension": 3,
        "passage_count": passage_count,
        "source_file_count": 1,
        "source_files": ["sample.jsonl"],
        "passage_ids_sha256": passage_ids_digest,
        "embeddings_sha256": _sha256_file(
            embeddings_path
        ),
        "passages_sha256": _sha256_file(
            passages_path
        ),
    }

    (root / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_dense_search_returns_cosine_ranking(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(artifact_root)

    encoder = _FakeEncoder(
        {
            "query: Ø¯ÛŒÙˆØ§Ù†": [1.0, 0.0, 0.0],
        }
    )

    index = DenseIndex(
        artifact_root,
        encoder=encoder,
    )

    result = index.search(
        "Ø¯ÛŒÙˆØ§Ù†",
        limit=2,
    )

    assert result.model_name == "example/test-encoder"
    assert result.hits[0].passage_id == (
        "diwan:passage_000001"
    )
    assert result.hits[0].cosine_score == pytest.approx(
        1.0
    )
    assert result.hits[1].passage_id == (
        "akhlaq:passage_000001"
    )
    assert encoder.calls == ["query: Ø¯ÛŒÙˆØ§Ù†"]


def test_dense_index_uses_memory_mapping(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(artifact_root)

    index = DenseIndex(
        artifact_root,
        encoder=_FakeEncoder({}),
    )

    assert isinstance(
        index._embeddings,
        np.memmap,
    )
    assert index.passage_count == 3
    assert index.embedding_dimension == 3


def test_dense_ties_are_ordered_by_row_index(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(artifact_root)

    encoder = _FakeEncoder(
        {
            "query: Ø¨Ø±Ø§Ø¨Ø±": [0.0, 0.0, 1.0],
        }
    )

    result = DenseIndex(
        artifact_root,
        encoder=encoder,
    ).search(
        "Ø¨Ø±Ø§Ø¨Ø±",
        limit=3,
    )

    assert [
        hit.passage_id
        for hit in result.hits
    ] == [
        "diwan:passage_000001",
        "mathnawi:passage_000001",
        "akhlaq:passage_000001",
    ]


@pytest.mark.parametrize(
    ("query_text", "limit", "message"),
    [
        ("   ", 10, "must not be empty"),
        ("Ø¯ÛŒÙˆØ§Ù†", 0, "between 1 and 100"),
        ("Ø¯ÛŒÙˆØ§Ù†", 101, "between 1 and 100"),
    ],
)
def test_dense_search_validates_input(
    tmp_path: Path,
    query_text: str,
    limit: int,
    message: str,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(artifact_root)

    index = DenseIndex(
        artifact_root,
        encoder=_FakeEncoder(
            {
                "query: Ø¯ÛŒÙˆØ§Ù†": [
                    1.0,
                    0.0,
                    0.0,
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        index.search(
            query_text,
            limit=limit,
        )


def test_dense_index_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(artifact_root)

    with (
        artifact_root
        / "passages.jsonl"
    ).open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write("\n")

    with pytest.raises(
        DenseIndexError,
        match="hash mismatch",
    ):
        DenseIndex(
            artifact_root,
            encoder=_FakeEncoder({}),
        )


def test_dense_index_rejects_noncontiguous_rows(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(
        artifact_root,
        row_indexes=(0, 2, 3),
    )

    with pytest.raises(
        DenseIndexError,
        match="row indexes",
    ):
        DenseIndex(
            artifact_root,
            encoder=_FakeEncoder({}),
        )


def test_dense_index_rejects_shape_mismatch(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(
        artifact_root,
        passage_count=4,
    )

    with pytest.raises(
        DenseIndexError,
        match="passage count",
    ):
        DenseIndex(
            artifact_root,
            encoder=_FakeEncoder({}),
        )


def test_dense_index_rejects_encoder_dimension_mismatch(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "dense"
    _write_artifact(artifact_root)

    encoder = _FakeEncoder(
        {
            "query: Ø¯ÛŒÙˆØ§Ù†": [1.0, 0.0],
        },
        dimension=2,
    )

    index = DenseIndex(
        artifact_root,
        encoder=encoder,
    )

    with pytest.raises(
        DenseIndexError,
        match="dimension",
    ):
        index.search("Ø¯ÛŒÙˆØ§Ù†")
