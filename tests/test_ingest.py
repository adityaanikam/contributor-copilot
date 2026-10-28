import json

import pytest

from contributor_copilot.ingest import (
    build_chunks,
    clean_markdown,
    corpus_ref,
    discover_files,
    heading_index,
    load_documents,
    load_issues,
    section_at,
    split_documents,
)

REPO = "https://github.com/microsoft/apm"


def test_discover_files_finds_the_15_sample_files(sample_corpus):
    files = discover_files(sample_corpus)
    names = [f.relative_to(sample_corpus).as_posix() for f in files]
    assert len(files) == 15
    assert names[:3] == ["README.md", "CONTRIBUTING.md", "GOVERNANCE.md"]
    assert "docs/src/content/docs/consumer/update-and-refresh.mdx" in names
    assert "LICENSE" not in names


def test_clean_markdown_blanks_frontmatter_and_mdx_imports_but_keeps_lines():
    raw = "---\ntitle: Run scripts\nsidebar:\n  order: 2\n---\n\nimport { Aside } from '@astrojs/starlight/components';\n\n## Usage\nText\n"
    text, title = clean_markdown(raw)
    assert title == "Run scripts"
    assert "sidebar" not in text and "import" not in text
    assert text.count("\n") == raw.count("\n")
    assert text.split("\n")[8] == "## Usage"


def test_heading_index_ignores_comments_inside_code_fences():
    text = "# Title\n```yaml\n# not a heading\n```\n## Real\n"
    assert heading_index(text) == [(1, 1, "Title"), (5, 2, "Real")]


def test_section_at_builds_the_heading_path():
    headings = [(1, 1, "Guide"), (3, 2, "Install"), (6, 3, "Windows"), (9, 2, "Usage")]
    assert section_at(headings, 7) == "Guide > Install > Windows"
    assert section_at(headings, 10) == "Guide > Usage"
    assert section_at(headings, 0) == ""


def test_load_documents_sets_citation_metadata(sample_corpus):
    docs = {d.metadata["source"]: d for d in load_documents(sample_corpus, REPO)}
    doctor = docs["docs/src/content/docs/reference/cli/doctor.md"]
    assert doctor.metadata["title"] == "apm doctor"
    assert doctor.metadata["kind"] == "doc"
    assert doctor.metadata["url"] == f"{REPO}/blob/main/docs/src/content/docs/reference/cli/doctor.md"
    # Files without frontmatter fall back to their first H1.
    assert docs["CONTRIBUTING.md"].metadata["title"] == "Contributing to APM"


def test_chunks_respect_size_and_point_at_real_lines(sample_corpus):
    chunks = build_chunks(sample_corpus, REPO, chunk_size=500, chunk_overlap=50)
    assert all(len(c.text) <= 500 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    for chunk in chunks:
        original = (sample_corpus / chunk.source).read_text(encoding="utf-8").split("\n")
        first_line = chunk.text.split("\n")[0]
        assert first_line in original[chunk.start_line - 1]
        assert chunk.url.endswith(f"?plain=1#L{chunk.start_line}-L{chunk.end_line}")


def test_chunk_section_comes_from_the_heading_above(sample_corpus):
    chunks = build_chunks(sample_corpus, REPO, chunk_size=1000, chunk_overlap=150)
    small = [c for c in chunks if "standing preapproval" in c.text]
    assert small and small[0].source == "CONTRIBUTING.md"
    assert small[0].section.endswith("Small corrections")
    assert small[0].location.startswith("CONTRIBUTING.md:")


def test_larger_chunk_size_gives_fewer_chunks(sample_corpus):
    small = build_chunks(sample_corpus, REPO, 500, 75)
    large = build_chunks(sample_corpus, REPO, 2000, 300)
    assert len(large) < len(small)


def test_corpus_ref_is_main_without_git(sample_corpus):
    assert corpus_ref(sample_corpus) == "main"


def test_build_chunks_fails_clearly_on_empty_corpus(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_corpus"):
        build_chunks(tmp_path, REPO, 1000, 150)


def _write_issues(path):
    # Synthetic records in the shape scripts/export_issues.py writes.
    records = [
        {
            "kind": "issue",
            "number": 7,
            "title": "Example issue title",
            "body": "Example issue body.",
            "url": "https://github.com/example/repo/issues/7",
            "comments": [{"author": "someone", "body": "Example comment."}],
        },
        {
            "kind": "pr",
            "number": 8,
            "title": "Example PR title",
            "body": None,
            "url": "https://github.com/example/repo/pull/8",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_load_issues_labels_issues_and_prs(tmp_path):
    path = tmp_path / "issues.jsonl"
    _write_issues(path)
    issue, pr = load_issues(path)
    assert issue.metadata["source"] == "issue #7"
    assert "Comment by someone: Example comment." in issue.page_content
    assert pr.metadata["source"] == "PR #8"
    assert pr.metadata["url"].endswith("/pull/8")


def test_load_issues_rejects_unknown_kind(tmp_path):
    path = tmp_path / "issues.jsonl"
    path.write_text('{"kind": "discussion", "number": 1, "title": "x"}\n')
    with pytest.raises(ValueError, match="issues.jsonl:1"):
        load_issues(path)


def test_issue_chunks_have_no_line_numbers(tmp_path):
    path = tmp_path / "issues.jsonl"
    _write_issues(path)
    chunks = split_documents(load_issues(path), 1000, 100)
    assert {c.source for c in chunks} == {"issue #7", "PR #8"}
    assert all(c.start_line is None for c in chunks)
    assert chunks[0].location == "issue #7"
    assert chunks[0].url == "https://github.com/example/repo/issues/7"


def test_build_chunks_includes_issues_when_present(sample_corpus, tmp_path):
    path = tmp_path / "issues.jsonl"
    _write_issues(path)
    chunks = build_chunks(sample_corpus, REPO, 1000, 150, issues_path=path)
    assert {"issue #7", "PR #8"} <= {c.source for c in chunks}
