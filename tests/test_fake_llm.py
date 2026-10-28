import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from contributor_copilot.bm25 import Hit
from contributor_copilot.config import Settings
from contributor_copilot.fake_llm import FakeChatModel, make_fake_llm, sentences
from contributor_copilot.ingest import Chunk
from contributor_copilot.llm import get_llm
from contributor_copilot.prompts import I_DONT_KNOW, answer_messages, grade_messages, rewrite_messages


def _hits(*texts):
    return [
        Hit(Chunk(id=str(i), text=t, source=f"{i}.md", kind="doc", title="T", section="S", url=""), 1.0)
        for i, t in enumerate(texts, start=1)
    ]


def test_answers_with_best_matching_sentence_and_citation():
    llm = make_fake_llm()
    hits = _hits("Unrelated text here.", "The probe can take up to five seconds. Other stuff.")
    reply = llm.invoke(answer_messages("How many seconds can the probe take?", hits))
    assert reply.content == "The probe can take up to five seconds. [2]"


def test_grade_accepts_extracted_sentence_and_rejects_others():
    llm = make_fake_llm()
    hits = _hits("The probe can take up to five\nseconds. Other stuff.")
    assert llm.invoke(grade_messages(hits, "The probe can take up to five seconds. [1]")).content == "YES"
    assert llm.invoke(grade_messages(hits, "The probe takes an hour. [1]")).content == "NO"
    assert llm.invoke(grade_messages(hits, "The probe can take up to five seconds.")).content == "NO"
    assert llm.invoke(grade_messages(hits, "The probe can take up to five seconds. [4]")).content == "NO"


def test_rewrite_returns_question_then_keywords_on_retry():
    llm = make_fake_llm()
    assert llm.invoke(rewrite_messages("How do I run tests?")).content == "How do I run tests?"
    retry = llm.invoke(rewrite_messages("How do I run tests?", previous_query="How do I run tests?"))
    assert retry.content == "run tests"


def test_says_idk_when_nothing_matches():
    reply = make_fake_llm().invoke(answer_messages("Who wrote zzzqqq?", _hits("apm install")))
    assert reply.content == I_DONT_KNOW


def test_reports_word_counts_as_usage():
    llm = FakeChatModel(respond=lambda messages: "two words")
    reply = llm.invoke([SystemMessage("a b"), HumanMessage("c d e")])
    assert reply.usage_metadata == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}


def test_unknown_prompt_is_an_error():
    with pytest.raises(ValueError):
        make_fake_llm().invoke([SystemMessage("something else"), HumanMessage("hi")])


def test_sentences_join_wrapped_lines_and_split_on_punctuation():
    assert sentences("One. Two\nwraps here!\n\nThree") == ["One.", "Two wraps here!", "Three"]


def test_get_llm_picks_provider_from_settings():
    assert isinstance(get_llm(Settings(llm_provider="fake")), FakeChatModel)
    ollama = get_llm(Settings(llm_provider="ollama", ollama_model="qwen2.5:3b"))
    assert type(ollama).__name__ == "ChatOllama"
    assert ollama.model == "qwen2.5:3b"
    assert ollama.temperature == 0
