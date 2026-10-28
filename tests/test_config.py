from pathlib import Path

import pytest

from contributor_copilot.config import Settings


def test_defaults_use_local_ollama_and_hybrid_retrieval():
    settings = Settings.from_env({})
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embeddings == "bge"
    assert settings.rerank is True


def test_env_overrides():
    settings = Settings.from_env(
        {
            "COPILOT_LLM_PROVIDER": "FAKE",
            "COPILOT_EMBEDDINGS": "none",
            "COPILOT_RERANK": "0",
            "COPILOT_CHUNK_SIZE": "500",
            "COPILOT_CHUNK_OVERLAP": "50",
            "COPILOT_CORPUS_DIR": "tests/sample_corpus",
            "COPILOT_TOP_K": "3",
        }
    )
    assert settings.llm_provider == "fake"
    assert settings.embeddings == "none"
    assert settings.rerank is False
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.corpus_dir == Path("tests/sample_corpus")
    assert settings.top_k == 3


def test_empty_value_falls_back_to_default():
    assert Settings.from_env({"COPILOT_OLLAMA_MODEL": ""}).ollama_model == "llama3.2:3b"


@pytest.mark.parametrize(
    "env",
    [
        {"COPILOT_LLM_PROVIDER": "openai"},
        {"COPILOT_EMBEDDINGS": "ada"},
        {"COPILOT_CHUNK_SIZE": "100", "COPILOT_CHUNK_OVERLAP": "100"},
    ],
)
def test_invalid_values_are_rejected(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)
