import json
import os

import pytest
from conftest import ROOT, SAMPLE_CORPUS

from contributor_copilot.bm25 import BM25Index, Hit
from contributor_copilot.evaluation import (
    citation_match,
    evaluate_answers,
    evaluate_retrieval,
    hit_at_k,
    key_fact_recall,
    load_questions,
    log_run,
    reciprocal_rank,
    unique_sources,
)
from contributor_copilot.fake_llm import make_fake_llm
from contributor_copilot.ingest import Chunk, build_chunks
from contributor_copilot.retriever import HybridRetriever
from contributor_copilot.service import Copilot

QUESTIONS = ROOT / "eval" / "questions.jsonl"
REPO = "https://github.com/microsoft/apm"


def _normalized(text):
    return " ".join(text.lower().replace("`", "").replace("*", "").split())


def test_shipped_questions_are_answerable_from_the_sample_corpus():
    questions = load_questions(QUESTIONS)
    assert len(questions) == 8
    for q in questions:
        in_sample = [s for s in q.expected_sources if (SAMPLE_CORPUS / s).is_file()]
        assert in_sample, f"{q.id}: no expected source in the sample corpus"
        text = _normalized(" ".join((SAMPLE_CORPUS / s).read_text(encoding="utf-8") for s in in_sample))
        for alternatives in q.key_facts:
            assert any(_normalized(alt) in text for alt in alternatives), (q.id, alternatives)


def _write(tmp_path, *rows):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


GOOD = {"id": "a", "question": "Q?", "expected_sources": ["A.md"], "key_facts": ["x", ["y", "z"]]}


def test_key_facts_accept_strings_or_lists_of_alternatives(tmp_path):
    [q] = load_questions(_write(tmp_path, GOOD))
    assert q.key_facts == (("x",), ("y", "z"))
    assert q.answerable is True


def test_unanswerable_questions_need_no_sources(tmp_path):
    [q] = load_questions(_write(tmp_path, {"id": "u", "question": "Q?", "answerable": False}))
    assert q.answerable is False and q.expected_sources == ()


@pytest.mark.parametrize(
    "rows, message",
    [
        (["{not json"], "invalid JSON"),
        ([{"question": "Q?"}], "'id'"),
        ([GOOD, GOOD], "duplicate id"),
        ([{**GOOD, "expected_sources": []}], "need expected_sources"),
        ([{**GOOD, "key_facts": [[]]}], "empty list"),
        ([""], "no questions"),
    ],
)
def test_invalid_question_files_fail_with_line_numbers(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_questions(_write(tmp_path, *rows))


def _hit(source):
    return Hit(Chunk(id=source, text="", source=source, kind="doc", title="", section="", url=""), 0.0)


def test_unique_sources_keeps_first_occurrence_order():
    assert unique_sources([_hit("a"), _hit("b"), _hit("a"), _hit("c")]) == ["a", "b", "c"]


def test_hit_at_k_and_reciprocal_rank():
    ranked = ["x", "y", "a", "b"]
    assert hit_at_k(ranked, ("a",), 1) == 0.0
    assert hit_at_k(ranked, ("a",), 3) == 1.0
    assert reciprocal_rank(ranked, ("b", "a")) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ("z",)) == 0.0


def test_key_fact_recall_ignores_case_and_markdown():
    facts = (("apm install --frozen",), ("five seconds", "5 seconds"), ("missing",))
    answer = "Run `APM install --frozen`; it waits up to 5   seconds."
    assert key_fact_recall(answer, facts) == pytest.approx(2 / 3)
    assert key_fact_recall("anything", ()) == 0.0


def test_citation_match_needs_one_expected_file():
    assert citation_match(["a.md", "b.md"], ("b.md",)) == 1.0
    assert citation_match(["a.md"], ("b.md",)) == 0.0
    assert citation_match([], ("b.md",)) == 0.0


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(BM25Index(build_chunks(SAMPLE_CORPUS, REPO, 1000, 150)), top_k=5)


def test_evaluate_retrieval_on_sample(retriever):
    metrics, rows = evaluate_retrieval(retriever, load_questions(QUESTIONS))
    assert set(metrics) == {"hit_at_1", "hit_at_3", "hit_at_5", "mrr"}
    assert len(rows) == 8
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert all(len(row["ranked_sources"]) <= 5 for row in rows)


def test_evaluate_answers_scores_each_question(retriever, tmp_path):
    questions = load_questions(QUESTIONS) + load_questions(
        _write(tmp_path, {"id": "off-topic", "question": "What is the capital of France?", "answerable": False})
    )
    metrics, rows = evaluate_answers(Copilot(retriever, make_fake_llm()), questions)
    assert {"key_fact", "citation_match", "idk", "grounded", "attempts", "latency_ms"} <= set(metrics)
    assert metrics["input_tokens"] > 0 and metrics["output_tokens"] > 0
    off_topic = rows[-1]
    assert off_topic["idk"] == 1.0 and off_topic["key_fact"] == 1.0 and off_topic["citation_match"] == 1.0
    doctor = next(r for r in rows if r["id"] == "doctor-timeout")
    assert doctor["key_fact"] == 1.0 and doctor["citation_match"] == 1.0


def test_log_run_writes_a_local_mlflow_file_store(tmp_path):
    import mlflow

    run_id = log_run(
        tmp_path / "mlruns",
        "test-exp",
        "chunk1000-rerank-off",
        {"chunk_size": 1000, "rerank": False},
        {"hit_at_1": 0.5, "mrr": 0.75},
        {"retrieval": [{"id": "a", "hit_at_1": 0.5}]},
    )
    run = mlflow.get_run(run_id)
    assert run.data.params["chunk_size"] == "1000"
    assert run.data.metrics["mrr"] == 0.75
    artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
    assert artifacts == ["retrieval.json"]
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
