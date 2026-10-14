from pathlib import Path

import pytest

from rag.evaluation import evaluate, load_cases, recall_at_k, reciprocal_rank
from rag.llm import NO_ANSWER, MockLLM
from rag.pipeline import RAGPipeline

DATA = Path(__file__).resolve().parent.parent / "data"


class FixedLLM:
    """Returns a canned reply and records every prompt it receives."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def pipeline() -> RAGPipeline:
    rag = RAGPipeline(MockLLM())
    rag.ingest_directory(DATA / "docs")
    return rag


def test_answers_with_citation_from_the_right_document(pipeline):
    answer = pipeline.ask("Are digital gift cards refundable?")
    assert answer.answered
    assert "not refundable" in answer.text
    assert answer.citations == ["refund-policy#0"]


def test_out_of_scope_question_is_refused_without_calling_the_llm():
    llm = FixedLLM("should not be used")
    rag = RAGPipeline(llm)
    rag.ingest_directory(DATA / "docs")
    answer = rag.ask("Do you offer student discounts?")
    assert not answer.answered
    assert answer.text == NO_ANSWER
    assert llm.prompts == []


def test_llm_refusal_is_passed_through():
    rag = RAGPipeline(FixedLLM(NO_ANSWER))
    rag.ingest("shipping", "Express shipping delivers within one business day.")
    assert not rag.ask("How fast is express shipping?").answered


def test_hallucinated_citations_are_separated_from_valid_ones():
    rag = RAGPipeline(FixedLLM("One business day [shipping#0] [made-up#7]"))
    rag.ingest("shipping", "Express shipping delivers within one business day.")
    answer = rag.ask("How fast is express shipping?")
    assert answer.citations == ["shipping#0"]
    assert answer.invalid_citations == ["made-up#7"]


def test_prompt_contains_question_and_chunk_ids():
    llm = FixedLLM("ok [shipping#0]")
    rag = RAGPipeline(llm)
    rag.ingest("shipping", "Express shipping delivers within one business day.")
    rag.ask("How fast is express shipping?")
    assert "[shipping#0] Express shipping" in llm.prompts[0]
    assert "Question: How fast is express shipping?" in llm.prompts[0]


# ---------- evaluation ----------

def test_recall_and_reciprocal_rank():
    assert recall_at_k(["a", "b", "c"], "b", k=2) == 1.0
    assert recall_at_k(["a", "b", "c"], "c", k=2) == 0.0
    assert reciprocal_rank(["a", "b"], "b") == 0.5
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_bundled_eval_set_passes_quality_bar(pipeline):
    report = evaluate(pipeline, load_cases(DATA / "eval_set.jsonl"), k=3)
    assert report["cases"] == 8
    assert report["recall_at_k"] == 1.0
    assert report["citation_accuracy"] == 1.0
    assert report["keyword_match"] == 1.0
    assert report["refusal_rate"] == 1.0
    assert report["invalid_citations"] == 0
    assert report["failures"] == []
