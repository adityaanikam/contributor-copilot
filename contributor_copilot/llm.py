"""Choose the chat model from COPILOT_LLM_PROVIDER."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from contributor_copilot.config import Settings
from contributor_copilot.fake_llm import make_fake_llm


def get_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "fake":
        return make_fake_llm()
    from langchain_ollama import ChatOllama

    # temperature 0 and a fixed seed make answers repeatable for evaluation.
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        seed=0,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
    )
