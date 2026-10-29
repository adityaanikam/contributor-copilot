"""LangGraph flow: rewrite, retrieve, answer, check grounding, retry once or give up.

    START -> rewrite -> retrieve -> generate -> check --grounded--> END
                ^                                 |
                +------------- retry -------------+
                                                  +--out of attempts--> give_up -> END

Each node is timed, and every LLM call goes through the request's
RequestTrace, passed in as config["configurable"]["trace"].
"""

from __future__ import annotations

from typing import Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from contributor_copilot.answer import CITATION_RE, Citation, generate_answer, is_idk
from contributor_copilot.bm25 import Hit
from contributor_copilot.guardrails import sanitize_hits
from contributor_copilot.prompts import I_DONT_KNOW, grade_messages, rewrite_messages
from contributor_copilot.retriever import HybridRetriever
from contributor_copilot.telemetry import RequestTrace


class State(TypedDict, total=False):
    question: str
    query: str  # latest rewritten query
    queries: list[str]  # every rewritten query, in order
    hits: list[Hit]
    flagged_sources: list[str]  # chunks the guardrail redacted
    answer: str
    citations: list[Citation]
    grounded: bool
    attempts: int


def citations_are_valid(answer: str, source_count: int) -> bool:
    """At least one citation, and every [n] points at a source that exists."""
    numbers = [int(n) for n in CITATION_RE.findall(answer)]
    return bool(numbers) and all(1 <= n <= source_count for n in numbers)


def _timed(name: str, node: Callable[[State, RequestTrace], State]):
    def run(state: State, config: RunnableConfig) -> State:
        trace = config.get("configurable", {}).get("trace") or RequestTrace()
        with trace.stage(name):
            return node(state, trace)

    return run


def build_graph(
    retriever: HybridRetriever,
    llm: BaseChatModel,
    max_attempts: int = 2,
    sanitize_sources: bool = True,
):
    """Compile the graph. max_attempts=2 means the first try plus one retry.

    sanitize_sources=False turns the injection guardrail off; only tests do that.
    """

    def rewrite(state: State, trace: RequestTrace) -> State:
        messages = rewrite_messages(state["question"], state.get("query"))
        lines = str(trace.invoke(llm, messages, "rewrite").content).strip().splitlines()
        query = lines[0].strip() if lines else ""
        query = query or state["question"]
        return {
            "query": query,
            "queries": [*state.get("queries", []), query],
            "attempts": state.get("attempts", 0) + 1,
        }

    def retrieve(state: State, trace: RequestTrace) -> State:
        # Fuse the original question with the rewrite so a bad rewrite cannot
        # lose what the question alone would find. Rerank against the question.
        queries = [state["question"], state["query"]]
        hits = retriever.retrieve(queries, rerank_query=state["question"])
        flagged: list[str] = []
        if sanitize_sources:
            hits, flagged = sanitize_hits(hits)
        seen = state.get("flagged_sources", [])
        return {"hits": hits, "flagged_sources": seen + [f for f in flagged if f not in seen]}

    def generate(state: State, trace: RequestTrace) -> State:
        answer = generate_answer(state["question"], state["hits"], llm, trace)
        return {"answer": answer.text, "citations": answer.citations}

    def check(state: State, trace: RequestTrace) -> State:
        answer, hits = state["answer"], state["hits"]
        if is_idk(answer) or not citations_are_valid(answer, len(hits)):
            return {"grounded": False}
        verdict = str(trace.invoke(llm, grade_messages(hits, answer), "check").content)
        return {"grounded": verdict.strip().upper().startswith("YES")}

    def give_up(state: State, trace: RequestTrace) -> State:
        # The grader rejected the answer, but it is a real answer with valid citations. A small
        # local model is an unreliable judge, so keep the answer and leave grounded False
        # instead of throwing it away. Answers that abstain or cite missing sources are dropped.
        answer = state["answer"]
        if not is_idk(answer) and citations_are_valid(answer, len(state["hits"])):
            return {"answer": answer, "citations": state["citations"]}
        return {"answer": I_DONT_KNOW, "citations": []}

    def route(state: State) -> str:
        if state["grounded"]:
            return "done"
        return "retry" if state["attempts"] < max_attempts else "give_up"

    graph = StateGraph(State)
    for name, node in [
        ("rewrite", rewrite),
        ("retrieve", retrieve),
        ("generate", generate),
        ("check", check),
        ("give_up", give_up),
    ]:
        graph.add_node(name, _timed(name, node))
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check")
    graph.add_conditional_edges(
        "check", route, {"done": END, "retry": "rewrite", "give_up": "give_up"}
    )
    graph.add_edge("give_up", END)
    return graph.compile()
