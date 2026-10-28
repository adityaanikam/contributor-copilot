from fakes import HashingEmbedder

from contributor_copilot.dense import ChromaStore, SentenceTransformerEmbedder, collection_name
from contributor_copilot.ingest import build_chunks

REPO = "https://github.com/microsoft/apm"


def test_collection_name_is_valid_for_chroma():
    name = collection_name("BAAI/bge-small-en-v1.5", 1000, 150)
    assert name == "c1000-o150-baai-bge-small-en-v1.5"
    assert len(collection_name("x" * 200, 500, 50)) <= 63


def test_store_embeds_once_and_finds_relevant_chunk(sample_corpus, tmp_path):
    chunks = build_chunks(sample_corpus, REPO, 1000, 150)
    embedder = HashingEmbedder()
    store = ChromaStore(chunks, embedder, tmp_path / "chroma", "test")
    assert store.count() == len(chunks)
    assert embedder.calls == len(chunks)

    hits = store.search("GITLAB_APM_PAT read_repository scope", k=5)
    assert len(hits) == 5
    assert any(h.chunk.source.endswith("consumer/authentication.md") for h in hits)
    assert hits[0].score >= hits[-1].score

    # A second store on the same directory reuses the stored embeddings.
    again = HashingEmbedder()
    ChromaStore(chunks, again, tmp_path / "chroma", "test")
    assert again.calls == 0


def test_store_deletes_chunks_that_left_the_corpus(sample_corpus, tmp_path):
    chunks = build_chunks(sample_corpus, REPO, 1000, 150)
    ChromaStore(chunks, HashingEmbedder(), tmp_path / "chroma", "test")
    smaller = ChromaStore(chunks[:10], HashingEmbedder(), tmp_path / "chroma", "test")
    assert smaller.count() == 10
    assert len(smaller.search("apm", k=50)) == 10


class RecordingModel:
    def __init__(self):
        self.inputs = []

    def encode(self, texts, normalize_embeddings, **kwargs):
        self.inputs.append((list(texts), normalize_embeddings))
        return [[1.0, 0.0] for _ in texts]


def test_embedder_prefixes_queries_but_not_documents():
    model = RecordingModel()
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", model=model)
    assert embedder.embed_documents(["doc text"]) == [[1.0, 0.0]]
    assert embedder.embed_query("question") == [1.0, 0.0]
    assert model.inputs[0] == (["doc text"], True)
    assert model.inputs[1] == ([SentenceTransformerEmbedder.QUERY_INSTRUCTION + "question"], True)
