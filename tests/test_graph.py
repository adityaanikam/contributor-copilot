import pytest
from fakes import KeywordReranker

from contributor_copilot.bm25 import BM25Index
from contributor_copilot.fake_llm import FakeChatModel, make_fake_llm, scripted_respond
from contributor_copilot.graph import build_graph, citations_are_valid
from contributor_copilot.ingest import build_chunks
from contributor_copilot.prompts import ANSWER_SYSTEM, GRADE_SYSTEM, I_DONT_KNOW, REWRITE_SYSTEM
from contributor_copilot.retriever import HybridRetriever

REPO = "https://github.com/microsoft/apm"
QUESTION = "How long can the apm doctor network probe take?"


@pytest.fixture(scope="module")
def retriever():
    from conftest import SAMPLE_CORPUS

    return HybridRetriever(BM25Index(build_chunks(SAMPLE_CORPUS, REPO, 1000, 150)), top_k=5)


def _system(messages):
    return messages[0].content


def scripted(answers, grades=None):
    """Fake LLM that rewrites like the default fake but uses fixed answers/grades in turn."""
    answers, grades = iter(answers), iter(grades or [])

    def respond(messages):
        system = _system(messages)
        if system == ANSWER_SYSTEM:
            return next(answers)
        if system == GRADE_SYSTEM:
            return next(grades)
        return scripted_respond(messages)

    return FakeChatModel(respond=respond)


def _prompts(llm):
    names = {REWRITE_SYSTEM: "rewrite", ANSWER_SYSTEM: "answer", GRADE_SYSTEM: "grade"}
    return [names[_system(call)] for call in llm.calls]


def test_grounded_answer_on_first_attempt(retriever):
    llm = make_fake_llm()
    state = build_graph(retriever, llm).invoke({"question": QUESTION})
    assert state["grounded"] is True
    assert state["attempts"] == 1
    assert "five" in state["answer"]
    assert state["citations"][0].source.endswith("reference/cli/doctor.md")
    assert _prompts(llm) == ["rewrite", "answer", "grade"]


def test_retry_once_then_grounded(retriever):
    llm = scripted(["Made up claim [1].", "Real claim [1]."], grades=["NO", "YES"])
    state = build_graph(retriever, llm).invoke({"question": QUESTION})
    assert state["grounded"] is True
    assert state["attempts"] == 2
    assert state["answer"] == "Real claim [1]."
    assert _prompts(llm) == ["rewrite", "answer", "grade"] * 2
    # The retry's rewrite prompt carries the failed query.
    retry_prompt = llm.calls[3][-1].content
    assert "did not find sources" in retry_prompt and QUESTION in retry_prompt
    assert len(state["queries"]) == 2 and state["queries"][0] != state["queries"][1]


def test_keeps_the_last_cited_answer_flagged_ungrounded_after_two_rejections(retriever):
    llm = scripted(["Wrong [1].", "Still wrong [2]."], grades=["NO", "NO"])
    state = build_graph(retriever, llm).invoke({"question": QUESTION})
    assert state["grounded"] is False
    assert state["attempts"] == 2
    assert state["answer"] == "Still wrong [2]."
    assert [c.number for c in state["citations"]] == [2]


def test_max_attempts_one_disables_retry(retriever):
    llm = scripted(["Wrong [1]."], grades=["NO"])
    state = build_graph(retriever, llm, max_attempts=1).invoke({"question": QUESTION})
    assert state["attempts"] == 1
    assert state["grounded"] is False
    assert state["answer"] == "Wrong [1]."


def test_gives_up_when_the_rejected_answer_has_no_valid_citation(retriever):
    llm = scripted(["Uncited claim.", "Cites a missing source [9]."], grades=[])
    state = build_graph(retriever, llm).invoke({"question": QUESTION})
    assert state["grounded"] is False
    assert state["answer"] == I_DONT_KNOW
    assert state["citations"] == []


def test_idk_and_bad_citations_skip_the_grader(retriever):
    llm = scripted([I_DONT_KNOW, "Cites a missing source [9]."])
    state = build_graph(retriever, llm).invoke({"question": QUESTION})
    assert state["answer"] == I_DONT_KNOW
    assert "grade" not in _prompts(llm)


def test_retrieve_fuses_question_and_rewrite_and_reranks_on_question():
    from conftest import SAMPLE_CORPUS

    reranker = KeywordReranker("five")
    retriever = HybridRetriever(
        BM25Index(build_chunks(SAMPLE_CORPUS, REPO, 1000, 150)), reranker=reranker, top_k=5
    )

    def respond(messages):
        if _system(messages) == REWRITE_SYSTEM:
            return "doctor probe timeout\nextra lines are ignored"
        return scripted_respond(messages)

    state = build_graph(retriever, FakeChatModel(respond=respond)).invoke({"question": QUESTION})
    assert state["queries"] == ["doctor probe timeout"]
    assert reranker.queries == [QUESTION]


def test_empty_rewrite_falls_back_to_the_question(retriever):
    def respond(messages):
        return "" if _system(messages) == REWRITE_SYSTEM else scripted_respond(messages)

    state = build_graph(retriever, FakeChatModel(respond=respond)).invoke({"question": QUESTION})
    assert state["queries"][0] == QUESTION


@pytest.mark.parametrize(
    "answer, count, expected",
    [
        ("Fact [1]. Other [2].", 2, True),
        ("No citation.", 2, False),
        ("Out of range [3].", 2, False),
        ("Zero [0].", 2, False),
    ],
)
def test_citations_are_valid(answer, count, expected):
    assert citations_are_valid(answer, count) is expected
