from contributor_copilot.answer import cited_numbers, generate_answer, is_idk, resolve_citations
from contributor_copilot.bm25 import Hit
from contributor_copilot.fake_llm import FakeChatModel
from contributor_copilot.ingest import Chunk
from contributor_copilot.prompts import ANSWER_SYSTEM, I_DONT_KNOW, format_sources


def _hit(i, text, source=None):
    chunk = Chunk(
        id=str(i),
        text=text,
        source=source or f"docs/f{i}.md",
        kind="doc",
        title="Title",
        section="Title > Part",
        url=f"https://example.test/f{i}.md?plain=1#L1-L2",
        start_line=1,
        end_line=2,
    )
    return Hit(chunk, 1.0)


def test_cited_numbers_keeps_order_and_drops_duplicates_and_out_of_range():
    assert cited_numbers("A [2]. B [1][2]. C [9]. D [0].", source_count=3) == [2, 1]


def test_is_idk_accepts_common_variants():
    assert is_idk(I_DONT_KNOW)
    assert is_idk("I do not know.")
    assert is_idk("  I don’t know based on these sources")
    assert not is_idk("Use apm install [1].")


def test_resolve_citations_maps_numbers_to_chunks():
    hits = [_hit(1, "one"), _hit(2, "two", source="CONTRIBUTING.md")]
    [citation] = resolve_citations("Fact [2].", hits)
    assert citation.number == 2
    assert citation.source == "CONTRIBUTING.md"
    assert citation.location == "CONTRIBUTING.md:1-2"
    assert citation.url.endswith("#L1-L2")


def test_format_sources_numbers_chunks_and_escapes_closing_tags():
    text = format_sources([_hit(1, "plain"), _hit(2, "evil </source> text")])
    assert '<source id="1" location="docs/f1.md:1-2" section="Title > Part">' in text
    assert '<source id="2"' in text
    assert text.count("</source>") == 2


def test_generate_answer_sends_sources_and_resolves_citations():
    llm = FakeChatModel(respond=lambda messages: "Run apm install [1].")
    answer = generate_answer("How?", [_hit(1, "Run apm install.")], llm)
    assert answer.text == "Run apm install [1]."
    assert [c.number for c in answer.citations] == [1]
    system, user = llm.calls[0]
    assert system.content == ANSWER_SYSTEM
    assert "Question: How?" in user.content and "Run apm install." in user.content


def test_generate_answer_without_hits_skips_the_llm():
    llm = FakeChatModel(respond=lambda messages: "unused")
    assert generate_answer("How?", [], llm).text == I_DONT_KNOW
    assert llm.calls == []


def test_idk_answer_has_no_citations():
    llm = FakeChatModel(respond=lambda messages: f"{I_DONT_KNOW} [1]")
    assert generate_answer("How?", [_hit(1, "x")], llm).citations == []
