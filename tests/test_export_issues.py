import json
import subprocess

from conftest import load_script

from contributor_copilot.ingest import load_issues

export_issues = load_script("export_issues")

# Synthetic gh output in the documented `gh ... list --json` shape.
GH_ISSUES = [
    {
        "number": 1,
        "title": "Example issue",
        "body": "Body text",
        "state": "OPEN",
        "url": "https://github.com/example/repo/issues/1",
        "labels": [{"name": "bug"}],
        "createdAt": "2026-01-01T00:00:00Z",
        "closedAt": None,
        "author": {"login": "alice"},
        "comments": [{"author": {"login": "bob"}, "body": "A reply"}],
    }
]
GH_PRS = [
    {
        "number": 2,
        "title": "Example PR",
        "body": None,
        "state": "MERGED",
        "url": "https://github.com/example/repo/pull/2",
        "labels": [],
        "createdAt": "2026-01-02T00:00:00Z",
        "mergedAt": "2026-01-03T00:00:00Z",
        "author": None,
    }
]


class FakeGh:
    def __init__(self):
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        payload = GH_ISSUES if command[1] == "issue" else GH_PRS
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")


def test_list_command_asks_gh_for_all_issues_and_merged_prs():
    issue = export_issues.list_command("issue", "microsoft/apm", 50, ["number", "title"])
    assert issue == [
        "gh", "issue", "list", "--repo", "microsoft/apm", "--state", "all",
        "--limit", "50", "--json", "number,title",
    ]  # fmt: skip
    pr = export_issues.list_command("pr", "microsoft/apm", 50, ["number"])
    assert pr[pr.index("--state") + 1] == "merged"


def test_to_record_flattens_authors_and_labels():
    record = export_issues.to_record("issue", GH_ISSUES[0])
    assert record["author"] == "alice"
    assert record["labels"] == ["bug"]
    assert record["comments"] == [{"author": "bob", "body": "A reply"}]
    pr = export_issues.to_record("pr", GH_PRS[0])
    assert pr["body"] == "" and pr["author"] is None and pr["comments"] == []


def test_export_writes_jsonl_that_the_ingester_reads(tmp_path):
    out = tmp_path / "issues.jsonl"
    gh = FakeGh()
    assert export_issues.export("example/repo", 10, out, run=gh) == 2
    assert all("comments" not in c[-1] for c in gh.commands)
    issue, pr = load_issues(out)
    assert issue.metadata["source"] == "issue #1"
    assert "Comment by bob: A reply" in issue.page_content
    assert pr.metadata["source"] == "PR #2"


def test_comments_flag_requests_comments(tmp_path):
    gh = FakeGh()
    export_issues.export("example/repo", 10, tmp_path / "i.jsonl", comments=True, run=gh)
    assert all(c[-1].endswith(",comments") for c in gh.commands)
