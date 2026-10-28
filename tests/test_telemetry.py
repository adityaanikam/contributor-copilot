import json

import pytest
from langchain_core.messages import HumanMessage

from contributor_copilot.bm25 import BM25Index
from contributor_copilot.fake_llm import FakeChatModel, make_fake_llm
from contributor_copilot.ingest import build_chunks
from contributor_copilot.prompts import ANSWER_SYSTEM, GRADE_SYSTEM
from contributor_copilot.retriever import HybridRetriever
from contributor_copilot.service import Copilot
from contributor_copilot.telemetry import RequestTrace, append_jsonl

REPO = "https://github.com/microsoft/apm"


def test_trace_records_tokens_and_latency_per_call():
    trace = RequestTrace()
    llm = FakeChatModel(respond=lambda messages: "three word reply")
    trace.invoke(llm, [HumanMessage("one two")], "rewrite")
    trace.invoke(llm, [HumanMessage("a b c d")], "generate")
    usage = trace.usage()
    assert [c.stage for c in trace.calls] == ["rewrite", "generate"]
    assert (usage.input_tokens, usage.output_tokens, usage.llm_calls) == (6, 6, 2)
    assert usage.latency_ms >= sum(c.latency_ms for c in trace.calls)


def test_stage_is_recorded_even_when_it_raises():
    trace = RequestTrace()
    with pytest.raises(RuntimeError):
        with trace.stage("retrieve"):
            raise RuntimeError("boom")
    assert [s.stage for s in trace.stages] == ["retrieve"]


def test_missing_usage_metadata_counts_as_zero():
    class NoUsage(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            result = super()._generate(messages, stop, run_manager, **kwargs)
            result.generations[0].message.usage_metadata = None
            return result

    trace = RequestTrace()
    trace.invoke(NoUsage(respond=lambda m: "x"), [HumanMessage("hi")], "check")
    assert trace.calls[0].input_tokens == 0 and trace.calls[0].output_tokens == 0


def test_append_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "logs" / "requests.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})
    assert [json.loads(line)["a"] for line in path.read_text().splitlines()] == [1, 2]


@pytest.fixture(scope="module")
def retriever():
    from conftest import SAMPLE_CORPUS

    return HybridRetriever(BM25Index(build_chunks(SAMPLE_CORPUS, REPO, 1000, 150)), top_k=5)


def test_ask_returns_usage_and_writes_one_log_line(retriever, tmp_path):
    log = tmp_path / "requests.jsonl"
    copilot = Copilot(retriever, make_fake_llm(), log_path=log)
    result = copilot.ask("How long can the apm doctor network probe take?")

    usage = result.usage
    assert usage.llm_calls == 3  # rewrite, generate, check
    assert usage.input_tokens > usage.output_tokens > 0
    assert [s.stage for s in usage.stages] == ["rewrite", "retrieve", "generate", "check"]

    [line] = log.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["request_id"] == usage.request_id
    assert record["input_tokens"] == usage.input_tokens
    assert record["latency_ms"] > 0
    assert [c["stage"] for c in record["calls"]] == ["rewrite", "generate", "check"]
    assert record["citations"][0].startswith("docs/src/content/docs/reference/cli/doctor.md:")
    assert record["grounded"] is True


def test_retry_counts_every_llm_call(retriever):
    answers = iter(["Made up [1].", "Also made up [1]."])

    def respond(messages):
        system = messages[0].content
        if system == ANSWER_SYSTEM:
            return next(answers)
        return "NO" if system == GRADE_SYSTEM else "doctor probe"

    result = Copilot(retriever, FakeChatModel(respond=respond)).ask("How long can the probe take?")
    assert result.attempts == 2
    assert result.usage.llm_calls == 6  # (rewrite, generate, check) x 2
    stages = [s.stage for s in result.usage.stages]
    assert stages.count("rewrite") == 2 and stages[-1] == "give_up"


def test_no_hits_means_no_answer_call(retriever):
    result = Copilot(retriever, make_fake_llm()).ask("What is the capital of France?")
    # Nothing matched, so generate answers "I don't know" without the LLM
    # and the grader is skipped: only the two rewrites call the model.
    assert result.attempts == 2
    assert result.usage.llm_calls == 2


def test_failed_request_is_logged_with_the_error(retriever, tmp_path):
    log = tmp_path / "requests.jsonl"

    def boom(messages):
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        Copilot(retriever, FakeChatModel(respond=boom), log_path=log).ask("anything")
    record = json.loads(log.read_text())
    assert "model crashed" in record["error"]
    assert record["question"] == "anything"
