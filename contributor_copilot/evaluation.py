"""Retrieval and answer metrics over eval/questions.jsonl, logged to MLflow.

Retrieval is scored on the question alone (no LLM): hit@k and MRR over the
ranked list of unique files. Answers go through the full LangGraph flow and
are scored on key facts and on whether a cited file is an expected one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from contributor_copilot.answer import is_idk
from contributor_copilot.bm25 import Hit
from contributor_copilot.retriever import HybridRetriever
from contributor_copilot.service import Copilot

K_VALUES = (1, 3, 5)
RETRIEVAL_DEPTH = 20  # chunks retrieved per question before taking unique files


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_sources: tuple[str, ...]
    key_facts: tuple[tuple[str, ...], ...]  # each fact = acceptable alternatives
    answerable: bool = True


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read and validate the JSONL question file. See eval/README.md for the format."""
    questions, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: invalid JSON ({error})") from error
        for field in ("id", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{where}: '{field}' must be a non-empty string")
        if row["id"] in seen:
            raise ValueError(f"{where}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        answerable = row.get("answerable", True)
        sources = tuple(row.get("expected_sources", []))
        facts = tuple(
            (fact,) if isinstance(fact, str) else tuple(fact) for fact in row.get("key_facts", [])
        )
        if answerable and (not sources or not facts):
            raise ValueError(f"{where}: answerable questions need expected_sources and key_facts")
        if not all(alternatives for alternatives in facts):
            raise ValueError(f"{where}: a key fact cannot be an empty list")
        questions.append(EvalQuestion(row["id"], row["question"], sources, facts, answerable))
    if not questions:
        raise ValueError(f"{path}: no questions")
    return questions


def unique_sources(hits: list[Hit]) -> list[str]:
    """Files in the order they first appear in the hit list."""
    return list(dict.fromkeys(h.chunk.source for h in hits))


def hit_at_k(ranked: list[str], expected: tuple[str, ...], k: int) -> float:
    return 1.0 if any(source in expected for source in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], expected: tuple[str, ...]) -> float:
    for rank, source in enumerate(ranked, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("`", "").replace("*", "").split())


def key_fact_recall(answer: str, key_facts: tuple[tuple[str, ...], ...]) -> float:
    """Fraction of key facts found in the answer (case and markdown insensitive)."""
    if not key_facts:
        return 0.0
    text = _normalize(answer)
    found = sum(any(_normalize(alt) in text for alt in facts) for facts in key_facts)
    return found / len(key_facts)


def citation_match(cited: list[str], expected: tuple[str, ...]) -> float:
    """1.0 if the answer cites at least one expected file."""
    return 1.0 if set(cited) & set(expected) else 0.0


def _means(rows: list[dict], keys: list[str]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in keys} if rows else {}


def evaluate_retrieval(
    retriever: HybridRetriever, questions: list[EvalQuestion]
) -> tuple[dict[str, float], list[dict]]:
    rows = []
    for q in questions:
        if not q.answerable:
            continue
        ranked = unique_sources(retriever.retrieve(q.question, top_k=RETRIEVAL_DEPTH))
        row = {"id": q.id, "ranked_sources": ranked[: max(K_VALUES)]}
        for k in K_VALUES:
            row[f"hit_at_{k}"] = hit_at_k(ranked, q.expected_sources, k)
        row["mrr"] = reciprocal_rank(ranked, q.expected_sources)
        rows.append(row)
    return _means(rows, [f"hit_at_{k}" for k in K_VALUES] + ["mrr"]), rows


def evaluate_answers(
    copilot: Copilot, questions: list[EvalQuestion]
) -> tuple[dict[str, float], list[dict]]:
    rows = []
    for q in questions:
        result = copilot.ask(q.question)
        cited = [c.source for c in result.citations]
        if q.answerable:
            key_fact = key_fact_recall(result.answer, q.key_facts)
            citation = citation_match(cited, q.expected_sources)
        else:
            # The right answer to an unanswerable question is "I don't know".
            key_fact = 1.0 if is_idk(result.answer) else 0.0
            citation = 1.0 if not cited else 0.0
        rows.append(
            {
                "id": q.id,
                "answer": result.answer,
                "cited": cited,
                "key_fact": key_fact,
                "citation_match": citation,
                "idk": 1.0 if is_idk(result.answer) else 0.0,
                "grounded": 1.0 if result.grounded else 0.0,
                "attempts": result.attempts,
                "latency_ms": result.usage.latency_ms,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            }
        )
    keys = ["key_fact", "citation_match", "idk", "grounded", "attempts", "latency_ms"]
    keys += ["input_tokens", "output_tokens"]
    return _means(rows, keys), rows


def log_run(
    tracking_dir: Path,
    experiment: str,
    run_name: str,
    params: dict,
    metrics: dict[str, float],
    rows: dict[str, list[dict]],
) -> str:
    """Log one configuration as an MLflow run in a local file store. Returns the run id."""
    # MLflow 3.x keeps the file store behind an opt-in flag; we want plain files.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for name, table in rows.items():
            mlflow.log_dict({"rows": table}, f"{name}.json")
    return run.info.run_id
