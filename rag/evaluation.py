"""Offline evaluation of the RAG pipeline.

Retrieval metrics : recall@k and MRR (did the right document come back, and how high?)
Answer metrics    : citation accuracy, keyword match, refusal rate on out-of-scope questions,
                    and hallucinated (invalid) citations.

Usage: python -m rag.evaluation --docs data/docs --cases data/eval_set.jsonl --min-recall 0.8
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rag.llm import llm_from_env
from rag.pipeline import RAGPipeline


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_doc: str | None  # None = out of scope, the pipeline should refuse
    expected_keywords: tuple[str, ...] = ()


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(EvalCase(row["question"], row.get("expected_doc"), tuple(row.get("expected_keywords", []))))
    return cases


def recall_at_k(retrieved_docs: list[str], expected_doc: str, k: int) -> float:
    return 1.0 if expected_doc in retrieved_docs[:k] else 0.0


def reciprocal_rank(retrieved_docs: list[str], expected_doc: str) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == expected_doc:
            return 1.0 / rank
    return 0.0


def evaluate(pipeline: RAGPipeline, cases: list[EvalCase], k: int = 3) -> dict:
    in_scope = [c for c in cases if c.expected_doc]
    out_of_scope = [c for c in cases if not c.expected_doc]
    recall = mrr = citation_ok = keyword_ok = 0.0
    invalid_citations = 0
    failures = []

    for case in in_scope:
        retrieved = [hit.chunk.doc_id for hit in pipeline.retriever.search(case.question, k)]
        recall += recall_at_k(retrieved, case.expected_doc, k)
        mrr += reciprocal_rank(retrieved, case.expected_doc)

        answer = pipeline.ask(case.question)
        invalid_citations += len(answer.invalid_citations)
        cited_docs = {citation.rsplit("#", 1)[0] for citation in answer.citations}
        cited = case.expected_doc in cited_docs
        keywords = all(kw.lower() in answer.text.lower() for kw in case.expected_keywords)
        citation_ok += cited
        keyword_ok += keywords
        if not (cited and keywords):
            failures.append({"question": case.question, "answer": answer.text,
                             "citations": answer.citations, "expected_doc": case.expected_doc})

    refused = 0
    for case in out_of_scope:
        answer = pipeline.ask(case.question)
        if answer.answered:
            failures.append({"question": case.question, "answer": answer.text, "expected": "refusal"})
        else:
            refused += 1

    n = len(in_scope) or 1
    return {
        "cases": len(cases),
        "k": k,
        "recall_at_k": round(recall / n, 3),
        "mrr": round(mrr / n, 3),
        "citation_accuracy": round(citation_ok / n, 3),
        "keyword_match": round(keyword_ok / n, 3),
        "refusal_rate": round(refused / (len(out_of_scope) or 1), 3),
        "invalid_citations": invalid_citations,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a JSONL question set.")
    parser.add_argument("--docs", type=Path, default=Path("data/docs"))
    parser.add_argument("--cases", type=Path, default=Path("data/eval_set.jsonl"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.0, help="exit with code 1 if recall@k is lower")
    args = parser.parse_args()

    pipeline = RAGPipeline(llm_from_env(), top_k=args.k)
    pipeline.ingest_directory(args.docs)
    report = evaluate(pipeline, load_cases(args.cases), args.k)
    print(json.dumps(report, indent=2))
    if report["recall_at_k"] < args.min_recall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
