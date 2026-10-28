"""Runtime settings, read from COPILOT_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

LLM_PROVIDERS = ("ollama", "fake")
EMBEDDING_MODES = ("bge", "none")


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = Path("data/corpus")
    issues_path: Path = Path("data/issues.jsonl")
    index_dir: Path = Path("data/index")
    log_path: Path = Path("logs/requests.jsonl")
    repo_url: str = "https://github.com/microsoft/apm"

    llm_provider: str = "ollama"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192

    # "bge" = hybrid BM25 + dense retrieval, "none" = BM25 only.
    embeddings: str = "bge"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    candidates: int = 20  # hits taken from each retriever before fusion
    top_k: int = 5  # chunks shown to the LLM

    def __post_init__(self):
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"COPILOT_LLM_PROVIDER must be one of {LLM_PROVIDERS}")
        if self.embeddings not in EMBEDDING_MODES:
            raise ValueError(f"COPILOT_EMBEDDINGS must be one of {EMBEDDING_MODES}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        defaults = cls()

        def get(name, default, cast=str):
            raw = env.get(f"COPILOT_{name}")
            return default if raw is None or raw == "" else cast(raw)

        return cls(
            corpus_dir=get("CORPUS_DIR", defaults.corpus_dir, Path),
            issues_path=get("ISSUES_PATH", defaults.issues_path, Path),
            index_dir=get("INDEX_DIR", defaults.index_dir, Path),
            log_path=get("LOG_PATH", defaults.log_path, Path),
            repo_url=get("REPO_URL", defaults.repo_url),
            llm_provider=get("LLM_PROVIDER", defaults.llm_provider).lower(),
            ollama_model=get("OLLAMA_MODEL", defaults.ollama_model),
            ollama_base_url=get("OLLAMA_BASE_URL", defaults.ollama_base_url),
            ollama_num_ctx=get("OLLAMA_NUM_CTX", defaults.ollama_num_ctx, int),
            embeddings=get("EMBEDDINGS", defaults.embeddings).lower(),
            embedding_model=get("EMBEDDING_MODEL", defaults.embedding_model),
            rerank=get("RERANK", defaults.rerank, _as_bool),
            rerank_model=get("RERANK_MODEL", defaults.rerank_model),
            chunk_size=get("CHUNK_SIZE", defaults.chunk_size, int),
            chunk_overlap=get("CHUNK_OVERLAP", defaults.chunk_overlap, int),
            candidates=get("CANDIDATES", defaults.candidates, int),
            top_k=get("TOP_K", defaults.top_k, int),
        )
