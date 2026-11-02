"""Local sentence embeddings (no API key needed), via fastembed's ONNX runtime."""

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models are trained to see this prefix on short search queries (not on passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding  # imported here so tests can run without the model

        self.model = TextEmbedding(model_name)

    def passages(self, texts: list[str]) -> np.ndarray:
        return normalize(np.array(list(self.model.passage_embed(texts))))

    def query(self, text: str) -> np.ndarray:
        return normalize(np.array(list(self.model.query_embed(QUERY_INSTRUCTION + text)))[0])


RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder: reads the question and a passage together, so it judges relevance more
    precisely than comparing two separate embeddings. Too slow for the whole index, so it only
    re-orders the top candidates from hybrid search."""

    def __init__(self, model_name: str = RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name)

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        return np.array(list(self.model.rerank(query, passages)))
