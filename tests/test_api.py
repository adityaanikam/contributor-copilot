import httpx
import pytest
from fastapi.testclient import TestClient

from contributor_copilot.api import create_app
from contributor_copilot.config import Settings
from contributor_copilot.fake_llm import FakeChatModel
from contributor_copilot.service import build_copilot


@pytest.fixture
def settings(sample_corpus, tmp_path):
    return Settings(
        corpus_dir=sample_corpus,
        issues_path=tmp_path / "none.jsonl",
        index_dir=tmp_path / "index",
        log_path=tmp_path / "logs" / "requests.jsonl",
        llm_provider="fake",
        embeddings="none",
        rerank=False,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client


def test_health_reports_bm25_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["retrieval"] == "bm25"
    assert body["chunks"] > 100


def test_ask_returns_answer_with_file_citation(client):
    response = client.post("/ask", json={"question": "How long can the apm doctor network probe take?"})
    assert response.status_code == 200
    body = response.json()
    assert "five" in body["answer"]
    [citation] = body["citations"]
    assert citation["source"] == "docs/src/content/docs/reference/cli/doctor.md"
    assert citation["location"].startswith(citation["source"] + ":")
    assert citation["url"].startswith("https://github.com/microsoft/apm/blob/main/")
    assert len(body["retrieved"]) == 5


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_unreachable_llm_returns_503(settings):
    copilot = build_copilot(settings)

    def refuse(messages):
        raise httpx.ConnectError("Connection refused")

    copilot.llm = FakeChatModel(respond=refuse)
    with TestClient(create_app(copilot=copilot)) as client:
        response = client.post("/ask", json={"question": "What does apm doctor check?"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]
