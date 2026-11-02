"""Test helpers: a deterministic bag-of-words embedder, so tests need no model download or API key."""

import zlib

import numpy as np
import pytest

from assistant.embeddings import normalize
from assistant.retrieve import Retriever, index_text, tokenize


class HashingEmbedder:
    dim = 512

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for token in tokenize(text):
            vec[zlib.crc32(token.encode()) % self.dim] += 1.0
        return normalize(vec)

    def passages(self, texts):
        return np.stack([self._embed(t) for t in texts])

    def query(self, text):
        return self._embed(text)


class OverlapReranker:
    """Scores a passage by how many query words it contains."""

    def scores(self, query, passages):
        words = set(tokenize(query))
        return np.array([len(words & set(tokenize(p))) for p in passages], dtype=float)


CHUNKS = [
    {"id": "cf-p3-w0", "doc_id": "cf", "title": "Consumer Financing", "url": "u1", "page": 3,
     "text": "The maximum tenure of auto loans shall not exceed seven years."},
    {"id": "cf-p5-w0", "doc_id": "cf", "title": "Consumer Financing", "url": "u1", "page": 5,
     "text": "Banks shall obtain a copy of the borrower CNIC and verify income before approving credit cards."},
    {"id": "bb-p2-w0", "doc_id": "bb", "title": "Branchless Banking", "url": "u2", "page": 2,
     "text": "Level 0 accounts are basic branchless banking accounts with low transaction limits."},
]


@pytest.fixture
def retriever():
    embedder = HashingEmbedder()
    return Retriever(CHUNKS, embedder.passages([index_text(c) for c in CHUNKS]), embedder)
