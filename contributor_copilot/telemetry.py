"""Per-request token counts and latency, appended to a JSONL log."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


@dataclass
class StageTiming:
    stage: str
    latency_ms: float


@dataclass
class LLMCall:
    stage: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class Usage:
    request_id: str
    input_tokens: int
    output_tokens: int
    llm_calls: int
    latency_ms: float
    stages: list[StageTiming]


@dataclass
class RequestTrace:
    """Collects stage timings and LLM token usage for one request."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stages: list[StageTiming] = field(default_factory=list)
    calls: list[LLMCall] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append(StageTiming(name, (time.perf_counter() - started) * 1000))

    def invoke(self, llm: BaseChatModel, messages: list[BaseMessage], stage: str):
        """Call the LLM and record its token usage and latency."""
        started = time.perf_counter()
        reply = llm.invoke(messages)
        usage = reply.usage_metadata or {}
        self.calls.append(
            LLMCall(
                stage=stage,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
        return reply

    def usage(self) -> Usage:
        return Usage(
            request_id=self.request_id,
            input_tokens=sum(c.input_tokens for c in self.calls),
            output_tokens=sum(c.output_tokens for c in self.calls),
            llm_calls=len(self.calls),
            latency_ms=(time.perf_counter() - self._started) * 1000,
            stages=list(self.stages),
        )

    def record(self, **fields) -> dict:
        """One log line: timestamp, usage totals, per-call detail and extra fields."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **asdict(self.usage()),
            "calls": [asdict(c) for c in self.calls],
            **fields,
        }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
