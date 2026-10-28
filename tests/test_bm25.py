import pytest

from contributor_copilot.bm25 import BM25Index, tokenize
from contributor_copilot.ingest import build_chunks

REPO = "https://github.com/microsoft/apm"


def test_tokenize_keeps_compound_words_and_their_parts():
    tokens = tokenize("Set GITLAB_APM_PAT for apm self-update")
    assert "gitlab_apm_pat" in tokens
    assert {"gitlab", "apm", "pat", "self-update", "self", "update"} <= set(tokens)
    assert "for" not in tokens


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("What does `apm run` do?") == ["apm", "run"]


@pytest.fixture(scope="module")
def index():
    from conftest import SAMPLE_CORPUS

    return BM25Index(build_chunks(SAMPLE_CORPUS, REPO, 1000, 150))


def test_search_finds_the_file_that_defines_the_term(index):
    hits = index.search("GITLAB_APM_PAT token for GitLab packages", k=5)
    assert hits[0].chunk.source.endswith("consumer/authentication.md")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_returns_at_most_k_and_only_positive_scores(index):
    hits = index.search("apm", k=3)
    assert len(hits) <= 3
    assert all(h.score > 0 for h in hits)


def test_search_with_no_known_terms_returns_nothing(index):
    assert index.search("zzzqqq", k=5) == []
    assert index.search("the and of", k=5) == []


def test_empty_index_is_rejected():
    with pytest.raises(ValueError):
        BM25Index([])
