"""FastAPI service: /health, /ingest and /ask.

Run locally:  uvicorn rag.api:app --reload
"""
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.llm import llm_from_env
from rag.pipeline import RAGPipeline

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


class IngestRequest(BaseModel):
    # '#' is reserved as the chunk separator in citation ids, so doc ids are restricted.
    doc_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    text: str = Field(min_length=1, max_length=200_000)


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    citations: list[str]
    sources: list[Source]


def create_app(pipeline: RAGPipeline | None = None, docs_dir: Path | None = DOCS_DIR) -> FastAPI:
    """App factory: tests pass their own pipeline; production uses the env-configured LLM."""
    pipeline = pipeline if pipeline is not None else RAGPipeline(llm_from_env())
    if docs_dir is not None and docs_dir.exists():
        pipeline.ingest_directory(docs_dir)

    app = FastAPI(title="RAG Q&A Service", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks_indexed": len(pipeline.retriever)}

    @app.post("/ingest", response_model=IngestResponse, status_code=201)
    def ingest(request: IngestRequest) -> IngestResponse:
        return IngestResponse(doc_id=request.doc_id, chunks=pipeline.ingest(request.doc_id, request.text))

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        answer = pipeline.ask(request.question)
        return AskResponse(
            answer=answer.text,
            answered=answer.answered,
            citations=answer.citations,
            sources=[
                Source(chunk_id=hit.chunk.chunk_id, score=round(hit.score, 3), text=hit.chunk.text)
                for hit in answer.sources
            ],
        )

    return app


app = create_app()
