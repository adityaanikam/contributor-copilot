# Evaluation questions

`questions.jsonl` holds one JSON object per line. `eval.py` reads it.

```json
{"id": "doctor-timeout", "question": "How long can the network probe in apm doctor take before it times out?", "expected_sources": ["docs/src/content/docs/reference/cli/doctor.md"], "key_facts": [["five seconds", "5 seconds"]]}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique short name. Shows up in the per-question MLflow artifacts. |
| `question` | yes | Asked exactly as written. |
| `expected_sources` | yes, unless `answerable` is false | Files that state the answer, as paths relative to the corpus root (the same form as a citation's `source`), or `issue #N` / `PR #N` for exported issues. List every file that states the fact, not just one. |
| `key_facts` | yes, unless `answerable` is false | Short phrases a correct answer must contain. Each item is a string, or a list of acceptable alternatives, for example `["five seconds", "5 seconds"]`. |
| `answerable` | no, default `true` | Set `false` for a question the corpus does not answer. The correct answer is then "I don't know", with no citations. |

## How each question is scored

Retrieval (no LLM): the question is run through the retriever, the top 20
chunks are reduced to unique files in rank order, and then

- `hit@k` is 1 if an expected file is in the first k files (k = 1, 3, 5),
- `MRR` is 1 / rank of the first expected file, or 0.

Answers (full LangGraph flow, needs an LLM):

- `key fact`: fraction of `key_facts` found in the answer. Matching ignores
  case, backticks, asterisks and repeated whitespace.
- `citation`: 1 if the answer cites at least one expected file.
- `IDK`, `grounded`, `latency ms`: share of "I don't know" answers, share that
  passed the grounding check, and mean time per question.

Unanswerable questions are skipped for retrieval. For answers they score 1 on
`key fact` when the answer is "I don't know" and 1 on `citation` when it
cites nothing.

## Adding your own questions

1. Find the fact in the corpus first. Copy the exact phrase into `key_facts`.
2. Search the whole corpus (`data/corpus`) for other files that state the same
   fact and add them all to `expected_sources`. If you leave one out, a
   retrieval that finds it is counted as a miss.
3. Keep key facts short and specific. `"start"` matches far more answers than
   `"apm install --frozen"` does.
4. Run `python eval.py --retrieval-only --chunk-sizes 1000 --rerank off` to
   check the file loads before a full run.

`tests/test_evaluation.py` checks that each shipped question's key facts
appear in its expected files in `tests/sample_corpus/`. Questions you add for
the full corpus do not need to pass that test; keep them in a separate file
(for example `eval/my_questions.jsonl`) and pass `--questions`.
