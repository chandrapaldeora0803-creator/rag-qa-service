import pytest
from fastapi.testclient import TestClient

from rag.api import DOCS_DIR, create_app
from rag.llm import MockLLM
from rag.pipeline import RAGPipeline


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(RAGPipeline(MockLLM()), docs_dir=DOCS_DIR))


def test_health_reports_indexed_chunks(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks_indexed": 3}


def test_ask_returns_cited_answer_and_sources(client):
    response = client.post("/ask", json={"question": "How long does express shipping take?"})
    body = response.json()
    assert response.status_code == 200
    assert body["answered"] is True
    assert "1 business day" in body["answer"]
    assert body["citations"] == ["shipping#0"]
    assert body["sources"][0]["chunk_id"] == "shipping#0"


def test_out_of_scope_question_is_refused(client):
    body = client.post("/ask", json={"question": "Do you offer student discounts?"}).json()
    assert body["answered"] is False
    assert body["citations"] == []


def test_ingested_document_becomes_searchable(client):
    created = client.post("/ingest", json={
        "doc_id": "loyalty",
        "text": "Loyalty points expire after 12 months of inactivity.",
    })
    assert created.status_code == 201
    assert created.json() == {"doc_id": "loyalty", "chunks": 1}

    body = client.post("/ask", json={"question": "When do loyalty points expire?"}).json()
    assert body["citations"] == ["loyalty#0"]
    assert "12 months" in body["answer"]


@pytest.mark.parametrize("path,payload", [
    ("/ask", {"question": "hi"}),
    ("/ask", {}),
    ("/ingest", {"doc_id": "bad#id", "text": "some text"}),
    ("/ingest", {"doc_id": "ok", "text": ""}),
])
def test_invalid_requests_are_rejected(client, path, payload):
    assert client.post(path, json=payload).status_code == 422
