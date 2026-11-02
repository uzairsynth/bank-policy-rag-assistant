"""Turn the SBP PDFs into page-level text chunks and an embedding index.

Usage:
    python -m assistant.ingest
"""

import json
import re
import time

import numpy as np
from pypdf import PdfReader

from assistant.embeddings import EMBED_MODEL, Embedder
from assistant.retrieve import index_text
from assistant.sources import DOCUMENTS, INDEX_DIR, pdf_path

CHUNK_WORDS = 180
OVERLAP_WORDS = 40


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def read_pages(path) -> list[str]:
    return [clean(page.extract_text() or "") for page in PdfReader(path).pages]


def chunk_pages(pages: list[str], doc: dict, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[dict]:
    """Chunk each page on its own so every chunk, and every citation, points to one page."""
    chunks = []
    step = size - overlap
    for page_no, text in enumerate(pages, start=1):
        words = text.split()
        if not words:
            continue
        for start in range(0, max(len(words) - overlap, 1), step):
            chunks.append({
                "id": f"{doc['id']}-p{page_no}-w{start}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "page": page_no,
                "text": " ".join(words[start:start + size]),
            })
    return chunks


def build_index(embedder: Embedder | None = None, index_dir=INDEX_DIR) -> dict:
    chunks = []
    for doc in DOCUMENTS:
        path = pdf_path(doc)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/download_docs.py")
        chunks += chunk_pages(read_pages(path), doc)

    embedder = embedder or Embedder()
    start = time.perf_counter()
    embeddings = embedder.passages([index_text(c) for c in chunks])
    seconds = time.perf_counter() - start

    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    np.save(index_dir / "embeddings.npy", embeddings.astype(np.float32))
    meta = {
        "embedding_model": EMBED_MODEL,
        "documents": len(DOCUMENTS),
        "chunks": len(chunks),
        "chunk_words": CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "embed_seconds": round(seconds, 1),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(build_index(), indent=2))
