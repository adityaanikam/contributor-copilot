"""Ask the LLM to answer from retrieved chunks and resolve its [n] citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from contributor_copilot.bm25 import Hit
from contributor_copilot.prompts import I_DONT_KNOW, answer_messages
from contributor_copilot.telemetry import RequestTrace

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    number: int
    source: str
    location: str
    section: str
    url: str
    kind: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)


def is_idk(text: str) -> bool:
    normalized = text.strip().lower().replace("’", "'")
    return normalized.startswith(("i don't know", "i do not know"))


def cited_numbers(text: str, source_count: int) -> list[int]:
    """Source numbers cited in the text, in order, without duplicates or out-of-range ones."""
    numbers = []
    for match in CITATION_RE.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= source_count and number not in numbers:
            numbers.append(number)
    return numbers


def resolve_citations(text: str, hits: list[Hit]) -> list[Citation]:
    citations = []
    for number in cited_numbers(text, len(hits)):
        chunk = hits[number - 1].chunk
        citations.append(
            Citation(
                number=number,
                source=chunk.source,
                location=chunk.location,
                section=chunk.section,
                url=chunk.url,
                kind=chunk.kind,
            )
        )
    return citations


def generate_answer(
    question: str, hits: list[Hit], llm: BaseChatModel, trace: RequestTrace | None = None
) -> Answer:
    if not hits:
        return Answer(I_DONT_KNOW)
    messages = answer_messages(question, hits)
    reply = trace.invoke(llm, messages, "generate") if trace else llm.invoke(messages)
    text = str(reply.content).strip()
    if is_idk(text):
        return Answer(text)
    return Answer(text, resolve_citations(text, hits))
