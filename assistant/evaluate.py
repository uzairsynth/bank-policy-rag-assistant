"""Measure the assistant on labelled questions.

Usage:
    python -m assistant.evaluate            # retrieval only: free, no API key needed
    python -m assistant.evaluate --answers  # also asks Claude every question (uses API credits)
"""

import argparse
import json
import statistics
import time

import numpy as np

from assistant.retrieve import MIN_SIMILARITY, MODES, Retriever
from assistant.sources import ROOT

QUESTIONS = ROOT / "eval" / "questions.jsonl"
REPORTS = ROOT / "reports"


def load_questions(path=QUESTIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(chunk: dict, question: dict) -> bool:
    return chunk["doc_id"] == question["doc_id"] and chunk["page"] in question["pages"]


def retrieval_metrics(retriever: Retriever, questions: list[dict], mode: str, ks=(1, 3, 5)) -> dict:
    hits = {k: 0 for k in ks}
    reciprocal_ranks, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        ranking, _ = retriever.rank(q["question"], mode)
        latencies.append(time.perf_counter() - start)
        first = next((r for r, idx in enumerate(ranking[:10]) if is_relevant(retriever.chunks[idx], q)), None)
        for k in ks:
            hits[k] += first is not None and first < k
        reciprocal_ranks.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(questions)
    return {
        **{f"hit@{k}": hits[k] / n for k in ks},
        "mrr@10": float(np.mean(reciprocal_ranks)),
        "median_latency_ms": round(statistics.median(latencies) * 1000, 1),
    }


def scope_gate_metrics(retriever: Retriever, in_scope: list[dict], out_of_scope: list[dict]) -> dict:
    inside = [retriever.in_scope(q["question"])[1] for q in in_scope]
    outside = [retriever.in_scope(q["question"])[1] for q in out_of_scope]
    return {
        "threshold": MIN_SIMILARITY,
        "in_scope_passed": float(np.mean([s >= MIN_SIMILARITY for s in inside])),
        "out_of_scope_blocked": float(np.mean([s < MIN_SIMILARITY for s in outside])),
        "in_scope_similarity": {"min": min(inside), "median": statistics.median(inside)},
        "out_of_scope_similarity": {"max": max(outside), "median": statistics.median(outside)},
    }


def answer_metrics(retriever: Retriever, in_scope: list[dict], out_of_scope: list[dict]) -> dict:
    from assistant.answer import OUT_OF_SCOPE_MESSAGE, ask

    rows = []
    for q in in_scope + out_of_scope:
        result = ask(q["question"], retriever)
        cited_correct = any(
            s["doc_id"] == q.get("doc_id") and s["page"] in q.get("pages", []) for s in result["sources"]
        )
        declined = result["answer"] == OUT_OF_SCOPE_MESSAGE or "do not cover" in result["answer"]
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "cited_pages": [(s["doc_id"], s["page"]) for s in result["sources"]],
            "has_citation": bool(result["sources"]),
            "cites_labelled_page": cited_correct,
            "declined": declined,
            "out_of_scope": bool(q.get("out_of_scope")),
        })
    answered = [r for r in rows if not r["out_of_scope"]]
    unanswerable = [r for r in rows if r["out_of_scope"]]
    return {
        "answers_with_citation": float(np.mean([r["has_citation"] for r in answered])),
        "answers_citing_labelled_page": float(np.mean([r["cites_labelled_page"] for r in answered])),
        "out_of_scope_declined": float(np.mean([r["declined"] for r in unanswerable])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", action="store_true", help="also evaluate Claude's answers (costs API credits)")
    args = parser.parse_args()

    retriever = Retriever.load()
    questions = load_questions()
    in_scope = [q for q in questions if not q.get("out_of_scope")]
    out_of_scope = [q for q in questions if q.get("out_of_scope")]

    report = {
        "questions": {"in_scope": len(in_scope), "out_of_scope": len(out_of_scope)},
        "chunks_indexed": len(retriever.chunks),
        "default_mode": retriever.default_mode,
        "retrieval": {
            mode: retrieval_metrics(retriever, in_scope, mode)
            for mode in MODES
            if mode != "rerank" or retriever.reranker is not None
        },
        "scope_gate": scope_gate_metrics(retriever, in_scope, out_of_scope),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retrieval_eval.json").write_text(json.dumps(report, indent=2))

    print(f"{len(in_scope)} labelled questions, {len(retriever.chunks)} chunks\n")
    print(f"{'mode':<8}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR@10':>8}{'ms':>7}")
    for mode, m in report["retrieval"].items():
        print(f"{mode:<8}{m['hit@1']:>8.1%}{m['hit@3']:>8.1%}{m['hit@5']:>8.1%}{m['mrr@10']:>8.3f}{m['median_latency_ms']:>7}")
    gate = report["scope_gate"]
    print(f"\nScope gate @ {gate['threshold']}: {gate['in_scope_passed']:.0%} of in-scope passed, "
          f"{gate['out_of_scope_blocked']:.0%} of out-of-scope blocked")

    if args.answers:
        answers = answer_metrics(retriever, in_scope, out_of_scope)
        (REPORTS / "answer_eval.json").write_text(json.dumps(answers, indent=2, ensure_ascii=False))
        print(f"\nAnswers with a citation:          {answers['answers_with_citation']:.0%}")
        print(f"Answers citing the labelled page: {answers['answers_citing_labelled_page']:.0%}")
        print(f"Out-of-scope questions declined:  {answers['out_of_scope_declined']:.0%}")


if __name__ == "__main__":
    main()
