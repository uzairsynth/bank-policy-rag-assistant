# SBP Regulations Assistant (RAG)

![tests](https://github.com/uzairsynth/bank-policy-rag-assistant/actions/workflows/ci.yml/badge.svg)

Ask a question about State Bank of Pakistan (SBP) regulations and get a short answer in which every statement cites the exact regulation page it came from. Built with retrieval-augmented generation (RAG): local hybrid search finds the right passages, and Claude answers only from them.

Covers 4 public regulations (108 pages): Prudential Regulations for Consumer Financing, Branchless Banking Regulations, Regulations for Payment Card Security, and AML/CFT/CPF Regulations.

Output format (illustrative answer; the quoted source text is the real regulation wording):

```
Q: What is the minimum down payment when financing a car through a bank?
A: The minimum down payment must not fall below 15% of the value of the vehicle [1].

[1] Prudential Regulations for Consumer Financing (2016), page 23
    "the banks/DFIs shall ensure that the minimum down payment does not fall below 15% of the value of vehicle."
```

## How it works

```
SBP PDFs ──> page-level chunks (180 words, 40 overlap, 275 chunks)
                 │
                 ├── BM25 keyword index
                 └── bge-small embeddings (local, no API cost)
                          │
question ──> relevance gate ──> hybrid search (reciprocal rank fusion)
                                     │
                                     └── cross-encoder re-ranks top 20
                                              │
                                     top 5 passages ──> Claude (citations enabled)
                                                            │
                                     answer + numbered sources with exact quotes
```

- **Page-level chunks**, so every citation names a single page a reviewer can open.
- **Hybrid search.** BM25 catches exact terms ("Level 0", "CNIC"); embeddings catch paraphrases ("how long do I keep records" vs "retention period"). Both rankings are merged with reciprocal rank fusion, which needs no score calibration.
- **Cross-encoder re-ranking.** A small model reads the question and each candidate passage together and puts the best one first.
- **Grounded answers.** Passages are sent to Claude as citable documents, so citations come back as pointers to the exact text Claude used, not citations the model writes itself. The prompt tells it to answer only from the excerpts and to say "The provided SBP regulations do not cover this" otherwise.
- **Relevance gate.** Questions that are nowhere near the documents (weather, sports) are answered without calling the LLM at all.

## Results

Retrieval is measured on [36 labelled questions](eval/questions.jsonl), written in everyday language and each tied to the page that answers it. A hit means a chunk from the correct page is ranked within the top *k*.

| Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| BM25 only | 38.9% | 55.6% | 61.1% | 0.489 |
| Embeddings only | 36.1% | 52.8% | 61.1% | 0.448 |
| Hybrid (BM25 + embeddings) | 38.9% | 58.3% | 63.9% | 0.515 |
| **Hybrid + cross-encoder (default)** | **52.8%** | **72.2%** | **80.6%** | **0.638** |

Re-ranking is the change that clearly matters: +17 points at Hit@5. With 36 questions, one question is 2.8 points, so smaller differences are within noise. For example, adding document titles to the index and the BGE query instruction raised embeddings-only Hit@5 from 55.6% to 61.1% but moved hybrid from 69.4% to 63.9% ([earlier run](reports/retrieval_eval_baseline.json)).

**Relevance gate** (similarity threshold 0.56, calibrated on the same questions): all 36 answerable questions pass, and all 4 non-banking questions are blocked. Banking questions that these regulations do not cover (bank profit rates, tax deadlines, deposit insurance) are semantically close to the documents, so 7 of those 8 pass the gate and rely on Claude's instruction to say the regulations do not cover them.

Re-ranking 20 passages takes a few seconds per question on a laptop CPU (4.4 s median in [this run](reports/retrieval_eval.json)); lowering `RERANK_DEPTH` trades accuracy for speed.

## Run it

```bash
pip install -r requirements.txt
python scripts/download_docs.py     # 4 PDFs from SBP's official archive site
python -m assistant.ingest          # chunks + embeddings (local models download once: 67 MB + 80 MB)
python -m assistant.evaluate        # retrieval metrics, free, no API key
streamlit run app/streamlit_app.py  # chat UI
pytest                              # unit tests, no model download or API key needed
```

Answers need an Anthropic API key (`ANTHROPIC_API_KEY`). Without one, the app still shows the most relevant passages. `python -m assistant.evaluate --answers` also checks Claude's answers: whether each one is cited, whether it cites the labelled page, and whether uncovered questions are declined. It uses API credits.

The model defaults to `claude-opus-5-5`; set `CLAUDE_MODEL` to change it. Requests opt into the API's server-side fallback, so a safety-classifier refusal is retried on a fallback model instead of failing.

## Project layout

```
assistant/sources.py     the 4 SBP documents and their URLs
assistant/ingest.py      PDF text -> page-level chunks -> embedding index
assistant/embeddings.py  local embedding model and cross-encoder
assistant/retrieve.py    BM25, embeddings, rank fusion, re-ranking, relevance gate
assistant/answer.py      Claude call with citable documents, citation parsing
assistant/evaluate.py    retrieval metrics and optional answer evaluation
app/streamlit_app.py     chat interface with sources and quotes
eval/questions.jsonl     36 labelled questions + 12 out-of-scope questions
```

## Limitations

- **Document versions.** The indexed versions date from 2016-2020. SBP amends regulations through circulars, so answers can be out of date. Always check [sbp.org.pk](https://www.sbp.org.pk) for the current text.
- **PDF text quality.** Some words come out of the PDFs split ("acce pt", "Payment C ards"), which hurts keyword search.
- **Small evaluation set.** 36 answerable questions written for this project. A production system would need a larger set reviewed by compliance staff.
- **Not legal advice.** A study and lookup aid for bank staff, not a compliance opinion.

## License

MIT. The SBP documents are not included in this repository; they are downloaded from SBP's website.
