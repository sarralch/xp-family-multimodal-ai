"""Embedding backends.

`SentenceTransformerEmbedder` is the production backend. `HashingEmbedder` is a
dependency-free, deterministic stand-in so tests and CI never download a model.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from xp_family.rag.text import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2-normalised vectors."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, batch_size=64, show_progress_bar=False)
        return _normalise(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """Bag-of-words hashed into a fixed number of buckets (feature hashing)."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hashing-{dim}"
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
                matrix[row, bucket] += 1.0
        return _normalise(matrix)
