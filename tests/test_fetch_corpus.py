import subprocess

from conftest import SAMPLE_CORPUS, load_script

fetch_corpus = load_script("fetch_corpus")


class RecordingRun:
    def __init__(self):
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")


def test_clone_is_shallow_and_sparse(tmp_path):
    dest = tmp_path / "corpus"
    clone, sparse = fetch_corpus.clone_commands(dest)
    assert clone[:2] == ["git", "clone"]
    assert "--depth" in clone and clone[clone.index("--depth") + 1] == "1"
    assert "--sparse" in clone
    assert clone[-2:] == [fetch_corpus.REPO_URL, str(dest)]
    assert sparse == ["git", "-C", str(dest), "sparse-checkout", "set", "docs"]


def test_fetch_runs_clone_then_reads_head(tmp_path):
    run = RecordingRun()
    assert fetch_corpus.fetch(tmp_path / "corpus", run=run) is True
    assert run.commands[0][:2] == ["git", "clone"]
    assert run.commands[1][3:] == ["sparse-checkout", "set", "docs"]
    assert run.commands[2][3:] == ["rev-parse", "HEAD"]


def test_fetch_skips_existing_dest_without_force(tmp_path):
    dest = tmp_path / "corpus"
    dest.mkdir()
    run = RecordingRun()
    assert fetch_corpus.fetch(dest, run=run) is False
    assert run.commands == []


def test_fetch_force_replaces_existing_dest(tmp_path):
    dest = tmp_path / "corpus"
    dest.mkdir()
    (dest / "old.md").write_text("old")
    run = RecordingRun()
    assert fetch_corpus.fetch(dest, force=True, run=run) is True
    assert not (dest / "old.md").exists()


def test_write_sample_copies_the_listed_files_and_license(tmp_path):
    out = tmp_path / "sample"
    written = fetch_corpus.write_sample(SAMPLE_CORPUS, out)
    assert len(written) == len(fetch_corpus.SAMPLE_FILES) + 1
    assert (out / "LICENSE").read_text().startswith("MIT License")


def test_committed_sample_matches_the_list():
    assert len(fetch_corpus.SAMPLE_FILES) == 15
    for relative in fetch_corpus.SAMPLE_FILES:
        assert (SAMPLE_CORPUS / relative).is_file(), relative
