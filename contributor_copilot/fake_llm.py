"""A rule-based chat model for tests and offline runs. No network, no model.

It is a real LangChain BaseChatModel, so the code under test calls it exactly
like ChatOllama. It recognises our prompts by their system message:

- answer: returns the source sentence that shares the most words with the
  question, followed by its citation, or "I don't know" when nothing matches.

Token counts in usage_metadata are whitespace word counts, not real tokens.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from contributor_copilot import prompts
from contributor_copilot.bm25 import tokenize

SOURCE_RE = re.compile(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
QUESTION_RE = re.compile(r"^Question: (.*)$", re.MULTILINE)
PARAGRAPH_RE = re.compile(r"\n\s*\n")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _word_count(text: Any) -> int:
    return len(str(text).split())


class FakeChatModel(BaseChatModel):
    respond: Callable[[list[BaseMessage]], str]
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-rule-based"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        text = self.respond(messages)
        input_tokens = sum(_word_count(m.content) for m in messages)
        output_tokens = _word_count(text)
        message = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def sentences(text: str) -> list[str]:
    """Split into sentences, joining lines that markdown wrapped mid-sentence."""
    found = []
    for paragraph in PARAGRAPH_RE.split(text):
        joined = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        found.extend(s for s in SENTENCE_END_RE.split(joined) if s)
    return found


def parse_sources(user_message: str) -> dict[int, str]:
    return {int(n): text for n, text in SOURCE_RE.findall(user_message)}


def _answer(user_message: str) -> str:
    question = QUESTION_RE.search(user_message).group(1)
    question_tokens = set(tokenize(question))
    needed = min(2, len(question_tokens))
    best_overlap, best = 0, None
    for number, text in parse_sources(user_message).items():
        for sentence in sentences(text):
            overlap = len(question_tokens & set(tokenize(sentence)))
            if overlap > best_overlap:
                best_overlap, best = overlap, (number, sentence)
    if best is None or best_overlap < needed:
        return prompts.I_DONT_KNOW
    number, sentence = best
    return f"{sentence} [{number}]"


def scripted_respond(messages: list[BaseMessage]) -> str:
    system, user = messages[0].content, messages[-1].content
    if system == prompts.ANSWER_SYSTEM:
        return _answer(user)
    raise ValueError("FakeChatModel got a prompt it does not know")


def make_fake_llm() -> FakeChatModel:
    return FakeChatModel(respond=scripted_respond)
