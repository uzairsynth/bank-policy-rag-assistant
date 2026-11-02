"""Hybrid retrieval: BM25 keyword search + embedding search, merged with reciprocal rank fusion,
then re-ordered by a cross-encoder.

Keyword search is strong on exact terms ("Level 0 account", "CNIC"); embeddings catch
paraphrases ("how long do I keep records" -> "retention period"). Fusing the two rankings
is more robust than either alone and needs no score calibration. The cross-encoder then
re-reads the top candidates together with the question to put the best passage first.
"""

import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from assistant.sources import INDEX_DIR

TOKEN = re.compile(r"[a-z0-9]+")
MODES = ("bm25", "vector", "hybrid", "rerank")
RERANK_DEPTH = 20

# Questions whose best passage is less similar than this are treated as outside the documents.
# Calibrated on eval/questions.jsonl: every answerable question scored >= 0.60 and every
# non-banking question <= 0.52 (see reports/retrieval_eval.json).
MIN_SIMILARITY = 0.56


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def index_text(chunk: dict) -> str:
    """What gets indexed: the passage plus the regulation it comes from.

    Many pages share vocabulary (definitions, "customer", "account"), so the title tells
    both searches which rulebook a passage belongs to. Claude still receives the raw passage.
    """
    return f"{chunk['title']}\n{chunk['text']}"


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, embedder, reranker=None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.reranker = reranker
        self.bm25 = BM25Okapi([tokenize(index_text(c)) for c in chunks])

    @classmethod
    def load(cls, index_dir=INDEX_DIR, embedder=None, reranker=None, use_reranker: bool = True) -> "Retriever":
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError("No index found. Run: python -m assistant.ingest")
        from assistant.embeddings import Embedder, Reranker

        embedder = embedder or Embedder()
        if reranker is None and use_reranker:
            reranker = Reranker()
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        return cls(chunks, np.load(index_dir / "embeddings.npy"), embedder, reranker)

    @property
    def default_mode(self) -> str:
        return "rerank" if self.reranker is not None else "hybrid"

    def rank(self, query: str, mode: str | None = None, depth: int = 50) -> tuple[list[int], np.ndarray]:
        """Chunk indices, best first, plus every chunk's cosine similarity to the query."""
        mode = mode or self.default_mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "rerank" and self.reranker is None:
            raise ValueError("rerank mode needs a reranker")
        similarity = self.embeddings @ self.embedder.query(query)
        vector_rank = list(np.argsort(-similarity)[:depth])
        if mode == "vector":
            return vector_rank, similarity
        bm25_rank = list(np.argsort(-self.bm25.get_scores(tokenize(query)))[:depth])
        if mode == "bm25":
            return bm25_rank, similarity
        fused = reciprocal_rank_fusion([bm25_rank, vector_rank])
        if mode == "hybrid":
            return fused, similarity
        head = fused[:RERANK_DEPTH]
        scores = self.reranker.scores(query, [index_text(self.chunks[i]) for i in head])
        return [head[i] for i in np.argsort(-scores)] + fused[RERANK_DEPTH:], similarity

    def search(self, query: str, k: int = 5, mode: str | None = None) -> list[dict]:
        ranking, similarity = self.rank(query, mode)
        return [{**self.chunks[i], "similarity": float(similarity[i])} for i in ranking[:k]]

    def in_scope(self, query: str) -> tuple[bool, float]:
        best = float((self.embeddings @ self.embedder.query(query)).max())
        return best >= MIN_SIMILARITY, best
