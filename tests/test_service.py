import pytest
from fakes import HashingEmbedder, KeywordReranker

from contributor_copilot import __main__ as cli
from contributor_copilot import service
from contributor_copilot.config import Settings
from contributor_copilot.fake_llm import FakeChatModel


@pytest.fixture
def settings(sample_corpus, tmp_path):
    return Settings(
        corpus_dir=sample_corpus,
        issues_path=tmp_path / "none.jsonl",
        index_dir=tmp_path / "index",
        llm_provider="fake",
        embeddings="none",
        rerank=False,
        top_k=3,
    )


def test_bm25_only_settings_build_no_models(settings):
    retriever = service.build_retriever(settings)
    assert retriever.mode == "bm25"
    assert retriever.dense is None and retriever.reranker is None
    assert retriever.top_k == 3


def test_default_settings_wire_dense_store_and_reranker(settings, monkeypatch):
    # Swap the model classes for offline fakes; the wiring is what is tested.
    monkeypatch.setattr(service, "SentenceTransformerEmbedder", lambda name: HashingEmbedder())
    monkeypatch.setattr(service, "CrossEncoderReranker", lambda name: KeywordReranker("apm"))
    hybrid = Settings(**{**settings.__dict__, "embeddings": "bge", "rerank": True})
    retriever = service.build_retriever(hybrid)
    assert retriever.mode == "hybrid+rerank"
    assert retriever.dense.count() == len(retriever.bm25.chunks)
    assert (settings.index_dir / "chroma").is_dir()


def test_ask_returns_answer_citations_and_retrieved(settings):
    copilot = service.build_copilot(settings)
    result = copilot.ask("Which variable holds the token for GitLab packages?")
    assert "GITLAB_APM_PAT" in result.answer
    assert result.citations[0].source.endswith("consumer/authentication.md")
    assert len(result.retrieved) == 3


def test_cli_prints_answer_and_citation_urls(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_copilot", lambda _: service.build_copilot(settings))
    cli.main(["Which variable holds the token for GitLab packages?"])
    out = capsys.readouterr().out
    assert "GITLAB_APM_PAT" in out
    assert "consumer/authentication.md:" in out
    assert "https://github.com/microsoft/apm/blob/main/" in out


def test_ask_passes_llm_errors_through(settings):
    copilot = service.build_copilot(settings)

    def boom(messages):
        raise RuntimeError("model crashed")

    copilot.llm = FakeChatModel(respond=boom)
    with pytest.raises(RuntimeError):
        copilot.ask("What does apm doctor check?")
