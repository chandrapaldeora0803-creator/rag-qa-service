# rag-qa-service

[![CI](https://github.com/chandrapaldeora0803-creator/rag-qa-service/actions/workflows/ci.yml/badge.svg)](https://github.com/chandrapaldeora0803-creator/rag-qa-service/actions/workflows/ci.yml)

A retrieval-augmented question answering (RAG) service in Python. It answers questions **only** from your documents, cites the exact chunk every answer came from, refuses when the documents don't contain the answer, and ships with an **evaluation step that runs in CI** so retrieval or answer quality can't silently regress.

Everything runs offline by default (BM25 retrieval + a deterministic mock LLM), so tests and CI need no API keys. Set `LLM_API_KEY` to switch to a real OpenAI-compatible model.

## Architecture

```
            ingest                                   ask
  docs ──► chunker ──► BM25 index        question ──► BM25 search (top-k)
          (overlapping                                  │  drop hits below min_score
           word windows)                                ▼
                                             no hits? ──► "I don't know" (LLM never called)
                                                        │
                                                        ▼
                                     prompt = rules + [chunk_id] context + question
                                                        │
                                                        ▼
                                       LLM (MockLLM | OpenAI-compatible API)
                                                        │
                                                        ▼
                          citation check: keep ids we actually sent, flag the rest
```

| Module | Responsibility |
|---|---|
| `rag/chunking.py` | Splits text into overlapping word windows; chunk id = `<doc_id>#<n>` |
| `rag/retriever.py` | BM25 implemented from scratch (tf saturation `k1`, length normalisation `b`, idf) |
| `rag/llm.py` | `LLMClient` protocol, offline `MockLLM`, `OpenAICompatibleLLM` over httpx |
| `rag/pipeline.py` | Retrieval threshold, prompt building, refusal handling, citation validation |
| `rag/api.py` | FastAPI app factory: `/health`, `/ingest`, `/ask` with Pydantic validation |
| `rag/evaluation.py` | recall@k, MRR, citation accuracy, keyword match, refusal rate, invalid citations |

## Design decisions

| Decision | Why |
|---|---|
| BM25 instead of embeddings (for now) | Zero dependencies, deterministic, strong baseline for keyword-heavy support docs. The retriever sits behind one method, so a vector or hybrid retriever can replace it. |
| Score threshold before calling the LLM | Cheapest hallucination guard: if nothing relevant is retrieved, refuse without spending a model call. |
| Citations validated against retrieved chunk ids | A model can invent a source. Unknown ids are moved to `invalid_citations` and counted by the evaluation. |
| Mock LLM with the same contract as a real one | Tests and the CI evaluation gate are fast, free and reproducible. |
| Evaluation as a CI gate | Retrieval changes (chunk size, stopwords, k) are measured, not guessed. CI fails if recall@k drops below the threshold. |
| App factory (`create_app`) | Tests inject their own pipeline/LLM; no global state to patch. |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                   # offline tests
python -m rag.evaluation --min-recall 0.8   # evaluation report (JSON)
uvicorn rag.api:app --reload                # http://localhost:8000/docs
```

With Docker:

```bash
docker build -t rag-qa-service .
docker run -p 8000:8000 rag-qa-service
```

Use a real model (any OpenAI-compatible endpoint):

```bash
export LLM_API_KEY=...
export LLM_BASE_URL=https://api.openai.com/v1   # optional
export LLM_MODEL=gpt-4o-mini                    # optional
```

## API

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "Are digital gift cards refundable?"}'
```

```json
{
  "answer": "Digital gift cards are not refundable. [refund-policy#0]",
  "answered": true,
  "citations": ["refund-policy#0"],
  "sources": [{"chunk_id": "refund-policy#0", "score": 3.1, "text": "..."}]
}
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status and number of indexed chunks |
| POST | `/ingest` | `{doc_id, text}` -> chunks and indexes a document (201) |
| POST | `/ask` | `{question}` -> answer, citations, retrieved sources |

## Evaluation

`data/eval_set.jsonl` holds labelled questions: the document that should be retrieved, keywords the answer must contain, and out-of-scope questions (`expected_doc: null`) that must be refused.

| Metric | Meaning |
|---|---|
| `recall_at_k` | Share of questions whose expected document is in the top-k results |
| `mrr` | Mean reciprocal rank of the expected document |
| `citation_accuracy` | Share of answers that cite the expected document |
| `keyword_match` | Share of answers containing all expected keywords |
| `refusal_rate` | Share of out-of-scope questions correctly refused |
| `invalid_citations` | Citations pointing at chunks that were never retrieved |

Every failing case is listed in `failures` with the question, answer and citations, so a regression can be debugged directly from the CI log.

## Next steps

- Hybrid retrieval (BM25 + embeddings) with reciprocal rank fusion, compared using the same evaluation set
- Re-ingesting a `doc_id` should replace its chunks instead of appending
- Persist the index (SQLite / pgvector) instead of keeping it in memory
- LLM-as-judge faithfulness scoring alongside the keyword checks
- Streaming responses and per-request latency/token metrics
