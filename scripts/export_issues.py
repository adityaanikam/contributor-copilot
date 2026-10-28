"""Export a repo's issues and merged PRs to data/issues.jsonl with the gh CLI.

Needs the GitHub CLI, logged in (gh auth login). Run it locally:

    python scripts/export_issues.py                  # microsoft/apm, 300 of each
    python scripts/export_issues.py --limit 1000 --comments

Each output line is one JSON object with kind "issue" or "pr", which
contributor_copilot.ingest.load_issues reads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "issues.jsonl"
ISSUE_FIELDS = ["number", "title", "body", "state", "url", "labels", "createdAt", "closedAt", "author"]
PR_FIELDS = ["number", "title", "body", "state", "url", "labels", "createdAt", "mergedAt", "author"]


def list_command(kind: str, repo: str, limit: int, fields: list[str]) -> list[str]:
    state = "all" if kind == "issue" else "merged"
    return [
        "gh", kind, "list",
        "--repo", repo,
        "--state", state,
        "--limit", str(limit),
        "--json", ",".join(fields),
    ]  # fmt: skip


def _login(user: dict | None) -> str | None:
    return (user or {}).get("login")


def to_record(kind: str, item: dict) -> dict:
    """Flatten gh's JSON into the shape the ingester expects."""
    return {
        "kind": kind,
        "number": item["number"],
        "title": item["title"],
        "body": item.get("body") or "",
        "state": item.get("state"),
        "url": item["url"],
        "labels": [label["name"] for label in item.get("labels") or []],
        "author": _login(item.get("author")),
        "created_at": item.get("createdAt"),
        "closed_at": item.get("closedAt"),
        "merged_at": item.get("mergedAt"),
        "comments": [
            {"author": _login(c.get("author")), "body": c.get("body") or ""}
            for c in item.get("comments") or []
        ],
    }


def export(
    repo: str, limit: int, out: Path, comments: bool = False, run=subprocess.run
) -> int:
    """Write issues then merged PRs to out. Returns the number of records."""
    records = []
    for kind, fields in (("issue", ISSUE_FIELDS), ("pr", PR_FIELDS)):
        fields = [*fields, "comments"] if comments else fields
        result = run(
            list_command(kind, repo, limit, fields),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        records += [to_record(kind, item) for item in json.loads(result.stdout)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="microsoft/apm")
    parser.add_argument("--limit", type=int, default=300, help="max issues and max PRs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--comments", action="store_true", help="also fetch comments (slower, larger output)"
    )
    args = parser.parse_args(argv)
    count = export(args.repo, args.limit, args.out, comments=args.comments)
    print(f"Wrote {count} records to {args.out}")


if __name__ == "__main__":
    main()
