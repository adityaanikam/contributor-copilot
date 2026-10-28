"""LangGraph flow: rewrite, retrieve, answer, check grounding, retry once or give up.

    START -> rewrite -> retrieve -> generate -> check --grounded--> END
                ^                                 |
                +------------- retry -------------+
                                                  +--out of attempts--> give_up -> END
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from contributor_copilot.answer import CITATION_RE, Citation, generate_answer, is_idk
from contributor_copilot.bm25 import Hit
from contributor_copilot.prompts import I_DONT_KNOW, grade_messages, rewrite_messages
from contributor_copilot.retriever import HybridRetriever


class State(TypedDict, total=False):
    question: str
    query: str  # latest rewritten query
    queries: list[str]  # every rewritten query, in order
    hits: list[Hit]
    answer: str
    citations: list[Citation]
    grounded: bool
    attempts: int


def citations_are_valid(answer: str, source_count: int) -> bool:
    """At least one citation, and every [n] points at a source that exists."""
    numbers = [int(n) for n in CITATION_RE.findall(answer)]
    return bool(numbers) and all(1 <= n <= source_count for n in numbers)


def build_graph(retriever: HybridRetriever, llm: BaseChatModel, max_attempts: int = 2):
    """Compile the graph. max_attempts=2 means the first try plus one retry."""

    def rewrite(state: State) -> State:
        reply = llm.invoke(rewrite_messages(state["question"], state.get("query")))
        lines = str(reply.content).strip().splitlines()
        query = lines[0].strip() if lines else ""
        query = query or state["question"]
        return {
            "query": query,
            "queries": [*state.get("queries", []), query],
            "attempts": state.get("attempts", 0) + 1,
        }

    def retrieve(state: State) -> State:
        # Fuse the original question with the rewrite so a bad rewrite cannot
        # lose what the question alone would find. Rerank against the question.
        queries = [state["question"], state["query"]]
        return {"hits": retriever.retrieve(queries, rerank_query=state["question"])}

    def generate(state: State) -> State:
        answer = generate_answer(state["question"], state["hits"], llm)
        return {"answer": answer.text, "citations": answer.citations}

    def check(state: State) -> State:
        answer, hits = state["answer"], state["hits"]
        if is_idk(answer) or not citations_are_valid(answer, len(hits)):
            return {"grounded": False}
        verdict = str(llm.invoke(grade_messages(hits, answer)).content)
        return {"grounded": verdict.strip().upper().startswith("YES")}

    def give_up(state: State) -> State:
        return {"answer": I_DONT_KNOW, "citations": []}

    def route(state: State) -> str:
        if state["grounded"]:
            return "done"
        return "retry" if state["attempts"] < max_attempts else "give_up"

    graph = StateGraph(State)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("check", check)
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check")
    graph.add_conditional_edges(
        "check", route, {"done": END, "retry": "rewrite", "give_up": "give_up"}
    )
    graph.add_edge("give_up", END)
    return graph.compile()
