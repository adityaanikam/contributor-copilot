from contributor_copilot.ingest import Chunk
from contributor_copilot.rerank import CrossEncoderReranker


def _chunk(i, text):
    return Chunk(id=str(i), text=text, source=f"f{i}.md", kind="doc", title="T", section="S", url="")


class LengthModel:
    """Scores a pair by the length of the chunk text."""

    def __init__(self):
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return [float(len(text)) for _, text in pairs]


def test_rerank_sorts_by_model_score_and_pairs_query_with_search_text():
    model = LengthModel()
    reranker = CrossEncoderReranker(model=model)
    chunks = [_chunk(1, "short"), _chunk(2, "a much longer chunk"), _chunk(3, "medium len")]
    hits = reranker.rerank("q", chunks)
    assert [h.chunk.id for h in hits] == ["2", "3", "1"]
    assert model.pairs[0] == ("q", chunks[0].search_text)


def test_rerank_empty_list_skips_the_model():
    model = LengthModel()
    assert CrossEncoderReranker(model=model).rerank("q", []) == []
    assert model.pairs is None
