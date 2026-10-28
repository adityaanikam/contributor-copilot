"""Run the real sentence-transformers wrappers against tiny random models.

The real bge and ms-marco models need a download. These tests build a
two-layer BERT with random weights in a temp dir instead, so they check that
our wrappers call the library correctly without any network. They say
nothing about retrieval quality. Skipped when the "models" extra is missing.
"""

import pytest

pytest.importorskip("sentence_transformers")

from contributor_copilot.dense import SentenceTransformerEmbedder  # noqa: E402
from contributor_copilot.ingest import Chunk  # noqa: E402
from contributor_copilot.rerank import CrossEncoderReranker  # noqa: E402

WORDS = "apm install run doctor network token gitlab scope script start the a what".split()


def _tiny_bert(path, classifier):
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertModel,
        BertTokenizerFast,
    )

    torch.manual_seed(0)
    path.mkdir()
    vocab = path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(path)
    config = BertConfig(
        vocab_size=5 + len(WORDS),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
    )
    model = BertForSequenceClassification(config) if classifier else BertModel(config)
    model.save_pretrained(path)
    return str(path)


def test_embedder_returns_unit_vectors(tmp_path):
    embedder = SentenceTransformerEmbedder(_tiny_bert(tmp_path / "embed", classifier=False))
    docs = embedder.embed_documents(["apm install", "run the doctor"])
    query = embedder.embed_query("what is apm")
    assert len(docs) == 2 and len(docs[0]) == len(query) == 16
    for vector in [*docs, query]:
        assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-5)


def test_cross_encoder_scores_every_chunk(tmp_path):
    reranker = CrossEncoderReranker(_tiny_bert(tmp_path / "rerank", classifier=True))
    chunks = [
        Chunk(id=str(i), text=t, source=f"{i}.md", kind="doc", title="t", section="s", url="")
        for i, t in enumerate(["apm install", "gitlab token scope", "run start script"])
    ]
    hits = reranker.rerank("gitlab token", chunks)
    assert sorted(h.chunk.id for h in hits) == ["0", "1", "2"]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
