"""Shallow-clone the microsoft/apm repository into data/corpus/.

Only the root files and docs/ are checked out (git sparse checkout). The
loader in contributor_copilot.ingest then reads README.md, CONTRIBUTING.md,
GOVERNANCE.md and every .md/.mdx file under docs/.

    python scripts/fetch_corpus.py                 # clone into data/corpus
    python scripts/fetch_corpus.py --force         # delete and clone again
    python scripts/fetch_corpus.py --write-sample  # refresh tests/sample_corpus
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/microsoft/apm.git"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data" / "corpus"
SAMPLE_DIR = ROOT / "tests" / "sample_corpus"

# The 15 files committed under tests/sample_corpus/. LICENSE is copied too,
# because the MIT license requires it to travel with copies of the files.
SAMPLE_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "docs/src/content/docs/concepts/what-is-apm.md",
    "docs/src/content/docs/concepts/the-three-promises.md",
    "docs/src/content/docs/concepts/glossary.md",
    "docs/src/content/docs/consumer/install-packages.md",
    "docs/src/content/docs/consumer/authentication.md",
    "docs/src/content/docs/consumer/run-scripts.md",
    "docs/src/content/docs/consumer/update-and-refresh.mdx",
    "docs/src/content/docs/reference/cli/doctor.md",
    "docs/src/content/docs/reference/cli/uninstall.md",
    "docs/src/content/docs/reference/common-errors.md",
    "docs/src/content/docs/troubleshooting/ssl-issues.md",
    "docs/src/content/docs/contributing/development-guide.md",
)
LICENSE_FILE = "LICENSE"


def clone_commands(dest: Path, repo_url: str = REPO_URL) -> list[list[str]]:
    """Commands for a depth-1 clone that checks out only root files and docs/."""
    return [
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, str(dest)],
        ["git", "-C", str(dest), "sparse-checkout", "set", "docs"],
    ]


def _make_writable(func, path, _exc):
    # Git marks pack files read-only, which makes rmtree fail on Windows.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def fetch(dest: Path, force: bool = False, run=subprocess.run) -> bool:
    """Clone the corpus into dest. Returns False if dest exists and force is off."""
    if dest.exists():
        if not force:
            print(f"{dest} already exists. Use --force to clone again.")
            return False
        shutil.rmtree(dest, onexc=_make_writable)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for command in clone_commands(dest):
        run(command, check=True)
    head = run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Cloned {REPO_URL} at {head.stdout.strip()} into {dest}")
    return True


def write_sample(corpus: Path, sample_dir: Path = SAMPLE_DIR) -> list[Path]:
    """Copy SAMPLE_FILES and LICENSE from a cloned corpus into sample_dir."""
    written = []
    for relative in (*SAMPLE_FILES, LICENSE_FILE):
        source = corpus / relative
        if not source.is_file():
            raise FileNotFoundError(f"{source} is missing; run fetch_corpus.py first")
        target = sample_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="delete dest and clone again")
    parser.add_argument(
        "--write-sample",
        action="store_true",
        help="copy the 15 sample files from dest into tests/sample_corpus",
    )
    args = parser.parse_args(argv)
    if args.write_sample:
        for path in write_sample(args.dest):
            print(f"wrote {path.relative_to(ROOT)}")
        return
    fetch(args.dest, force=args.force)


if __name__ == "__main__":
    main()
