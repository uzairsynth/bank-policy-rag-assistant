"""Answer questions with Claude, grounded in retrieved passages and returned with citations.

Each retrieved chunk is sent as a citable document. Claude's citations come back with the
exact quoted text and the document they came from, so every claim maps to a regulation page.
"""

import os

import anthropic

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """You answer questions about State Bank of Pakistan (SBP) regulations for bank staff.

Use only the regulation excerpts provided as documents in the user's message. Each excerpt's title names the regulation and page.
- Answer directly in the first sentence, then give the conditions or exceptions that matter.
- Quote limits, percentages, amounts and time periods exactly as the excerpt states them.
- If the excerpts do not contain the answer, say "The provided SBP regulations do not cover this." and stop. Do not fill gaps from general knowledge.
- If the excerpts only partly answer the question, answer that part and say what is missing.
- Keep the answer under 150 words. It is a study aid, not legal advice."""

OUT_OF_SCOPE_MESSAGE = "This question does not match anything in the indexed SBP regulations."


def build_user_content(question: str, hits: list[dict]) -> list[dict]:
    documents = [
        {
            "type": "document",
            "source": {"type": "content", "content": [{"type": "text", "text": hit["text"]}]},
            "title": f"{hit['title']}, page {hit['page']}",
            "citations": {"enabled": True},
        }
        for hit in hits
    ]
    return [*documents, {"type": "text", "text": question}]


def parse_response(content, hits: list[dict]) -> dict:
    """Join the answer text and turn Claude's citations into numbered sources."""
    sources: dict[str, dict] = {}
    parts = []
    for block in content:
        if block.type != "text":
            continue
        numbers = []
        for citation in block.citations or []:
            hit = hits[citation.document_index]
            if hit["id"] not in sources:
                sources[hit["id"]] = {
                    "n": len(sources) + 1,
                    "doc_id": hit["doc_id"],
                    "title": hit["title"],
                    "page": hit["page"],
                    "url": hit["url"],
                    "quotes": [],
                }
            source = sources[hit["id"]]
            quote = citation.cited_text.strip()
            if quote not in source["quotes"]:
                source["quotes"].append(quote)
            numbers.append(source["n"])
        parts.append(block.text + "".join(f" [{n}]" for n in sorted(set(numbers))))
    return {"answer": "".join(parts).strip(), "sources": list(sources.values())}


def answer(question: str, hits: list[dict], client: anthropic.Anthropic | None = None) -> dict:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        # If a safety classifier declines, the API retries on a fallback model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": build_user_content(question, hits)}],
    )
    if response.stop_reason == "refusal":
        return {"answer": "The model declined to answer this question.", "sources": [], "model": response.model}
    result = parse_response(response.content, hits)
    result["model"] = response.model
    result["usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return result


def ask(question: str, retriever, k: int = 5, client: anthropic.Anthropic | None = None) -> dict:
    """Full pipeline: scope check, hybrid retrieval, then a cited answer from Claude."""
    in_scope, best = retriever.in_scope(question)
    if not in_scope:
        return {"answer": OUT_OF_SCOPE_MESSAGE, "sources": [], "passages": [], "in_scope": False, "best_similarity": best}
    hits = retriever.search(question, k=k)
    return {**answer(question, hits, client), "passages": hits, "in_scope": True, "best_similarity": best}
