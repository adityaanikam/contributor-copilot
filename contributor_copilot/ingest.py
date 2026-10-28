"""Load the corpus (docs and optional issues) and split it into citable chunks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from typing import Iterator

import yaml
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

ROOT_FILES = ("README.md", "CONTRIBUTING.md", "GOVERNANCE.md")
DOC_SUFFIXES = (".md", ".mdx")

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
MDX_IMPORT_RE = re.compile(r"^import .+ from ['\"].+['\"];?[ \t]*$", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str  # repo-relative path, or "issue #12" / "PR #34"
    kind: str  # "doc", "issue" or "pr"
    title: str
    section: str  # heading path, e.g. "Contributing to APM > Small corrections"
    url: str
    start_line: int | None = None
    end_line: int | None = None

    @property
    def search_text(self) -> str:
        """Text that BM25 and embeddings see: title and section give context."""
        return f"{self.title}\n{self.section}\n{self.text}"

    @property
    def location(self) -> str:
        if self.start_line is None:
            return self.source
        return f"{self.source}:{self.start_line}-{self.end_line}"


def discover_files(corpus_dir: Path) -> list[Path]:
    """Root README/CONTRIBUTING/GOVERNANCE plus every .md/.mdx under docs/."""
    files = [corpus_dir / name for name in ROOT_FILES if (corpus_dir / name).is_file()]
    docs_dir = corpus_dir / "docs"
    if docs_dir.is_dir():
        files += sorted(p for p in docs_dir.rglob("*") if p.suffix in DOC_SUFFIXES and p.is_file())
    return files


def corpus_ref(corpus_dir: Path) -> str:
    """Commit SHA of a cloned corpus, used in citation URLs. 'main' if unknown."""
    if not (corpus_dir / ".git").exists():
        return "main"
    try:
        result = subprocess.run(
            ["git", "-C", str(corpus_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    return result.stdout.strip()


def _blank_out(match: re.Match) -> str:
    # Keep the line count so line numbers still match the original file.
    return "\n" * match.group(0).count("\n")


def clean_markdown(text: str) -> tuple[str, str | None]:
    """Blank out YAML frontmatter and MDX import lines. Returns (text, title)."""
    text = text.replace("\r\n", "\n")
    title = None
    match = FRONTMATTER_RE.match(text)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        if isinstance(meta, dict) and meta.get("title"):
            title = str(meta["title"])
        text = _blank_out(match) + text[match.end() :]
    text = MDX_IMPORT_RE.sub("", text)
    return text, title


def heading_index(text: str) -> list[tuple[int, int, str]]:
    """(line_number, level, title) for each markdown heading outside code fences."""
    headings = []
    in_fence = False
    for number, line in enumerate(text.split("\n"), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            headings.append((number, len(match.group(1)), match.group(2)))
    return headings


def section_at(headings: list[tuple[int, int, str]], line: int) -> str:
    """Heading path in effect at the given line."""
    stack: list[tuple[int, str]] = []
    for number, level, title in headings:
        if number > line:
            break
        stack = [entry for entry in stack if entry[0] < level]
        stack.append((level, title))
    return " > ".join(title for _, title in stack)


class MarkdownCorpusLoader(BaseLoader):
    """LangChain document loader for the corpus markdown files.

    Yields one Document per file with the metadata a citation needs. It uses
    LangChain's BaseLoader interface instead of langchain-community's
    TextLoader, because langchain-community is being sunset.
    """

    def __init__(self, corpus_dir: Path, repo_url: str, ref: str = "main"):
        self.corpus_dir = corpus_dir
        self.repo_url = repo_url
        self.ref = ref

    def lazy_load(self) -> Iterator[Document]:
        for path in discover_files(self.corpus_dir):
            text, title = clean_markdown(path.read_text(encoding="utf-8"))
            source = path.relative_to(self.corpus_dir).as_posix()
            if title is None:
                first_h1 = next((h[2] for h in heading_index(text) if h[1] == 1), None)
                title = first_h1 or path.stem
            yield Document(
                page_content=text,
                metadata={
                    "source": source,
                    "kind": "doc",
                    "title": title,
                    "url": f"{self.repo_url}/blob/{self.ref}/{source}",
                },
            )


def load_documents(corpus_dir: Path, repo_url: str, ref: str = "main") -> list[Document]:
    return MarkdownCorpusLoader(corpus_dir, repo_url, ref).load()


def load_issues(path: Path) -> list[Document]:
    """Load issues and merged PRs written by scripts/export_issues.py."""
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind not in ("issue", "pr"):
                raise ValueError(f"{path}:{line_number}: kind must be 'issue' or 'pr'")
            label = "issue" if kind == "issue" else "PR"
            parts = [record["title"], record.get("body") or ""]
            for comment in record.get("comments") or []:
                parts.append(f"Comment by {comment.get('author', 'unknown')}: {comment.get('body', '')}")
            documents.append(
                Document(
                    page_content="\n\n".join(parts),
                    metadata={
                        "source": f"{label} #{record['number']}",
                        "kind": kind,
                        "title": record["title"],
                        "url": record.get("url", ""),
                    },
                )
            )
    return documents


def _chunk_id(source: str, start: int, text: str) -> str:
    return hashlib.sha1(f"{source}\0{start}\0{text}".encode("utf-8")).hexdigest()[:16]


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split with LangChain's markdown-aware recursive splitter.

    The splitter prefers to break at headings, then code fences, then blank
    lines. add_start_index gives each chunk's character offset, which we turn
    into line numbers and the heading path for the citation.
    """
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    plain_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = []
    for document in documents:
        meta = document.metadata
        is_doc = meta["kind"] == "doc"
        splitter = markdown_splitter if is_doc else plain_splitter
        headings = heading_index(document.page_content) if is_doc else []
        for piece in splitter.split_documents([document]):
            start = piece.metadata["start_index"]
            start_line = end_line = None
            section = meta["title"]
            url = meta["url"]
            if is_doc:
                start_line = document.page_content.count("\n", 0, start) + 1
                end_line = start_line + piece.page_content.count("\n")
                section = section_at(headings, start_line) or meta["title"]
                url = f"{url}?plain=1#L{start_line}-L{end_line}"
            chunks.append(
                Chunk(
                    id=_chunk_id(meta["source"], start, piece.page_content),
                    text=piece.page_content,
                    source=meta["source"],
                    kind=meta["kind"],
                    title=meta["title"],
                    section=section,
                    url=url,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
    return chunks


def build_chunks(
    corpus_dir: Path,
    repo_url: str,
    chunk_size: int,
    chunk_overlap: int,
    issues_path: Path | None = None,
) -> list[Chunk]:
    """Load docs (and issues, if the file exists) and split them into chunks."""
    documents = load_documents(corpus_dir, repo_url, corpus_ref(corpus_dir))
    if issues_path is not None and issues_path.is_file():
        documents += load_issues(issues_path)
    if not documents:
        raise FileNotFoundError(
            f"No documents found in {corpus_dir}. Run scripts/fetch_corpus.py first."
        )
    return split_documents(documents, chunk_size, chunk_overlap)
