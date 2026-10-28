import importlib.util

import pytest
from conftest import ROOT, SAMPLE_CORPUS

from contributor_copilot.config import Settings

spec = importlib.util.spec_from_file_location("eval_cli", ROOT / "eval.py")
eval_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_cli)


def test_parse_args_defaults_to_full_grid():
    args = eval_cli.parse_args([])
    assert args.chunk_sizes == [500, 1000, 2000]
    assert args.rerank == [True, False]
    assert args.retrieval_only is False


def test_parse_args_rejects_bad_rerank_option():
    with pytest.raises(SystemExit):
        eval_cli.parse_args(["--rerank", "maybe"])


def test_format_table_marks_missing_metrics_na():
    table = eval_cli.format_table(
        [{"chunk_size": 500, "rerank": False, "retrieval_mode": "bm25", "hit_at_1": 0.5, "latency_ms": 1234.5}]
    )
    header, rule, row = table.splitlines()
    assert header.startswith("| chunk | rerank | retrieval | hit@1")
    assert row.startswith("| 500 | off | bm25 | 0.500 |")
    assert "| 1234 |" in row or "| 1235 |" in row
    assert "n/a" in row


def test_run_logs_one_mlflow_run_per_config(tmp_path, capsys):
    args = eval_cli.parse_args(
        [
            "--corpus", str(SAMPLE_CORPUS),
            "--chunk-sizes", "500,1000",
            "--rerank", "off",
            "--mlruns", str(tmp_path / "mlruns"),
            "--out", str(tmp_path / "table.md"),
        ]
    )  # fmt: skip
    base = Settings(llm_provider="fake", embeddings="none", index_dir=tmp_path / "index")
    results = eval_cli.run(args, base)
    assert [r["chunk_size"] for r in results] == [500, 1000]
    assert all(r["chunk_overlap"] == round(r["chunk_size"] * 0.15) for r in results)
    assert all("key_fact" in r and "hit_at_5" in r for r in results)
    assert "fake LLM" in capsys.readouterr().out


def test_main_writes_table_file(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_EMBEDDINGS", "none")
    monkeypatch.setenv("COPILOT_INDEX_DIR", str(tmp_path / "index"))
    out = tmp_path / "table.md"
    eval_cli.main(
        [
            "--corpus", str(SAMPLE_CORPUS),
            "--chunk-sizes", "1000",
            "--rerank", "off",
            "--retrieval-only",
            "--mlruns", str(tmp_path / "mlruns"),
            "--out", str(out),
        ]
    )  # fmt: skip
    lines = out.read_text().splitlines()
    assert len(lines) == 3 and lines[2].startswith("| 1000 | off | bm25 |")
