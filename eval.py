"""Evaluate retrieval and answers over eval/questions.jsonl; log each config to MLflow.

    python eval.py                                   # chunk 500/1000/2000 x rerank on/off
    python eval.py --retrieval-only                  # skip the LLM
    python eval.py --chunk-sizes 1000 --rerank off --corpus tests/sample_corpus

Other settings (LLM provider, model, embeddings) come from COPILOT_* env vars.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from contributor_copilot.config import Settings
from contributor_copilot.dense import SentenceTransformerEmbedder
from contributor_copilot.evaluation import (
    K_VALUES,
    evaluate_answers,
    evaluate_retrieval,
    load_questions,
    log_run,
)
from contributor_copilot.llm import get_llm
from contributor_copilot.rerank import CrossEncoderReranker
from contributor_copilot.service import Copilot, build_retriever

COLUMNS = [
    ("chunk", "chunk_size"),
    ("rerank", "rerank"),
    ("retrieval", "retrieval_mode"),
    *[(f"hit@{k}", f"hit_at_{k}") for k in K_VALUES],
    ("MRR", "mrr"),
    ("key fact", "key_fact"),
    ("citation", "citation_match"),
    ("IDK", "idk"),
    ("grounded", "grounded"),
    ("latency ms", "latency_ms"),
    ("tokens in", "input_tokens"),
    ("tokens out", "output_tokens"),
]


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--corpus", type=Path, help="defaults to COPILOT_CORPUS_DIR")
    parser.add_argument("--chunk-sizes", default="500,1000,2000")
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--rerank", default="on,off", help="on, off, or on,off")
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer metrics")
    parser.add_argument("--mlruns", type=Path, default=Path("mlruns"))
    parser.add_argument("--experiment", default="contributor-copilot")
    parser.add_argument("--out", type=Path, help="also write the markdown table here")
    args = parser.parse_args(argv)
    args.chunk_sizes = [int(size) for size in _csv(args.chunk_sizes)]
    options = _csv(args.rerank)
    if not options or set(options) - {"on", "off"}:
        parser.error("--rerank takes on, off, or on,off")
    args.rerank = [option == "on" for option in options]
    return args


def format_table(results: list[dict]) -> str:
    def cell(value):
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, float):
            return f"{value:.0f}" if value >= 100 else f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(title for title, _ in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for result in results:
        lines.append("| " + " | ".join(cell(result.get(key)) for _, key in COLUMNS) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace, base: Settings) -> list[dict]:
    if args.corpus:
        base = replace(base, corpus_dir=args.corpus)
    questions = load_questions(args.questions)
    if not args.retrieval_only and base.llm_provider == "fake":
        print("Note: answer metrics below come from the rule-based fake LLM, not a real model.")
    # Load each model once and share it across configurations.
    embedder = SentenceTransformerEmbedder(base.embedding_model) if base.embeddings == "bge" else None
    reranker = CrossEncoderReranker(base.rerank_model) if any(args.rerank) else None
    llm = None if args.retrieval_only else get_llm(base)

    results = []
    for size in args.chunk_sizes:
        for rerank in args.rerank:
            settings = replace(
                base,
                chunk_size=size,
                chunk_overlap=round(size * args.overlap_ratio),
                rerank=rerank,
            )
            retriever = build_retriever(settings, embedder=embedder, reranker=reranker)
            metrics, retrieval_rows = evaluate_retrieval(retriever, questions)
            rows = {"retrieval": retrieval_rows}
            if llm is not None:
                answer_metrics, answer_rows = evaluate_answers(Copilot(retriever, llm), questions)
                metrics.update(answer_metrics)
                rows["answers"] = answer_rows
            params = {
                "corpus": str(settings.corpus_dir),
                "questions": str(args.questions),
                "n_questions": len(questions),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "rerank": settings.rerank,
                "retrieval_mode": retriever.mode,
                "embeddings": settings.embeddings,
                "embedding_model": settings.embedding_model if settings.embeddings == "bge" else "",
                "rerank_model": settings.rerank_model if settings.rerank else "",
                "llm_provider": "" if llm is None else settings.llm_provider,
                "llm_model": settings.ollama_model if llm is not None and settings.llm_provider == "ollama" else "",
                "top_k": settings.top_k,
                "candidates": settings.candidates,
            }
            run_name = f"chunk{size}-rerank-{'on' if rerank else 'off'}"
            run_id = log_run(args.mlruns, args.experiment, run_name, params, metrics, rows)
            print(f"logged {run_name} as MLflow run {run_id}")
            results.append({**params, **metrics})
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    table = format_table(run(args, Settings.from_env()))
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
