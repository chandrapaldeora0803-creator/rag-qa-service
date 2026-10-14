"""RAG pipeline: chunk -> index -> retrieve -> build prompt -> generate -> validate citations."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import chunk_text
from rag.llm import NO_ANSWER, LLMClient
from rag.retriever import BM25Retriever, ScoredChunk

PROMPT_TEMPLATE = """You answer questions using ONLY the context below.
Cite the source id in square brackets after each fact, for example [refund-policy#0].
If the context does not contain the answer, reply exactly: {no_answer}

Context:
{context}

Question: {question}
Answer:"""

_CITATION = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    sources: list[ScoredChunk] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.text != NO_ANSWER


class RAGPipeline:
    def __init__(self, llm: LLMClient, top_k: int = 3, min_score: float = 0.5,
                 chunk_size: int = 120, chunk_overlap: int = 30):
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever = BM25Retriever()

    def ingest(self, doc_id: str, text: str) -> int:
        chunks = chunk_text(doc_id, text, self.chunk_size, self.chunk_overlap)
        self.retriever.add(chunks)
        return len(chunks)

    def ingest_directory(self, directory: Path) -> int:
        """Ingest every .md file in a directory; the file name (without extension) becomes the doc id."""
        total = 0
        for path in sorted(Path(directory).glob("*.md")):
            total += self.ingest(path.stem, path.read_text(encoding="utf-8"))
        return total

    def build_prompt(self, question: str, hits: list[ScoredChunk]) -> str:
        context = "\n".join(f"[{hit.chunk.chunk_id}] {hit.chunk.text}" for hit in hits)
        return PROMPT_TEMPLATE.format(no_answer=NO_ANSWER, context=context, question=question)

    def ask(self, question: str) -> Answer:
        # 1. Retrieve, and drop weak matches so the model is not asked to answer from noise.
        hits = [hit for hit in self.retriever.search(question, self.top_k) if hit.score >= self.min_score]
        if not hits:
            return Answer(question, NO_ANSWER)

        # 2. Generate.
        raw = self.llm.complete(self.build_prompt(question, hits)).strip()
        if not raw or NO_ANSWER in raw:
            return Answer(question, NO_ANSWER, sources=hits)

        # 3. Keep only citations that point at chunks we actually sent; anything else is hallucinated.
        allowed = {hit.chunk.chunk_id for hit in hits}
        cited = list(dict.fromkeys(_CITATION.findall(raw)))
        return Answer(
            question=question,
            text=raw,
            citations=[c for c in cited if c in allowed],
            sources=hits,
            invalid_citations=[c for c in cited if c not in allowed],
        )
