"""Graph control-flow tests with a scripted planner. No LLM is called."""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from inkference.agent import graph as graph_mod
from inkference.agent import nodes
from inkference.rag.llm import LLMUnavailable


class FakeIndex:
    """Stands in for RagIndex: always "built", returns one hit per query."""

    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [_Hit(1, "Plymouth harbour text", 0.8)]
        self.queries = []

    def exists(self, doc_id):
        return True

    def query(self, doc_id, question, top_k=None):
        self.queries.append(question)
        return self.hits


class _Hit:
    def __init__(self, page_number, text, score):
        self.page_number, self.text, self.score = page_number, text, score


@pytest.fixture
def scripted(monkeypatch):
    """Feed the planner a fixed list of replies; record how many times it was asked."""
    state = {"replies": [], "calls": 0}

    def fake_complete(system, prompt, cfg=None, **kw):
        i = state["calls"]
        state["calls"] += 1
        if i >= len(state["replies"]):
            raise AssertionError("planner called more times than the script allows")
        reply = state["replies"][i]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(nodes, "complete", fake_complete)
    monkeypatch.setattr(
        nodes, "generate_answer",
        lambda q, contexts, cfg=None, persona=None, history=None:
            f"ANSWER[{len(contexts)} contexts]",
    )
    return state


def run(store, index, scripted, replies, question="What happened at Plymouth?", **kw):
    scripted["replies"] = replies
    g = graph_mod.build_graph(store, index, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": kw.pop("thread", "t1")}, "recursion_limit": 30}
    return g.invoke({"doc_id": store.doc_id, "question": question,
                     "max_steps": kw.pop("max_steps", 3), "steps": 0, **kw}, cfg)


def test_plan_act_compose_happy_path(store, scripted):
    index = FakeIndex()
    out = run(store, index, scripted, [
        '{"standalone_question":"What happened at Plymouth?","action":"search","query":"Plymouth"}',
        '{"action":"answer"}',
    ])
    assert out["steps"] == 1
    assert out["answer"] == "ANSWER[1 contexts]"
    assert out["source_pages"] == [1]
    assert [t["action"] for t in out["trace"]] == ["search"]


def test_standalone_question_is_applied(store, scripted):
    out = run(store, FakeIndex(), scripted, [
        '{"standalone_question":"What was the weather at Plymouth?","action":"answer"}',
    ])
    assert out["standalone_question"] == "What was the weather at Plymouth?"


def test_step_cap_forces_compose(store, scripted):
    index = FakeIndex()
    # The planner keeps asking for more work; max_steps=2 must stop it anyway.
    out = run(store, index, scripted, [
        '{"action":"search","query":"a"}',
        '{"action":"search","query":"b"}',
    ], max_steps=2)
    assert out["steps"] == 2
    assert out["stop_reason"] in ("max_steps", "answer")
    assert out["answer"].startswith("ANSWER[")


def test_repeated_action_short_circuits(store, scripted):
    # An identical repeat can never add evidence, so the graph must stop asking.
    out = run(store, FakeIndex(), scripted, [
        '{"action":"search","query":"Plymouth"}',
        '{"action":"search","query":"plymouth "}',   # same, modulo case/space
    ], max_steps=3)
    assert out["steps"] == 1
    assert out["stop_reason"] == "repeated_action"


def test_unparseable_reply_spends_one_repair_then_answers(store, scripted):
    out = run(store, FakeIndex(), scripted, [
        "I will now search the journal.",   # unparseable
        '{"action":"search","query":"Plymouth"}',  # repair succeeds
        '{"action":"answer"}',
    ])
    assert scripted["calls"] == 3
    assert [t["action"] for t in out["trace"]] == ["search"]


def test_unparseable_twice_degrades_to_answer_without_raising(store, scripted):
    out = run(store, FakeIndex(), scripted, ["nope", "still nope"])
    assert out["steps"] == 0
    assert out["answer"].startswith("ANSWER[") or "No transcribed text" in out["answer"]


def test_planner_unavailable_degrades_to_one_deterministic_search(store, scripted):
    """With no API key the agent must still answer — same behaviour as POST /ask."""
    index = FakeIndex()
    out = run(store, index, scripted, [
        LLMUnavailable("no LLM provider configured"),
        LLMUnavailable("no LLM provider configured"),
    ])
    assert out["stop_reason"] == "llm_unavailable"
    assert index.queries == ["What happened at Plymouth?"]
    assert out["source_pages"] == [1]


def test_no_evidence_returns_a_clear_message(store, scripted):
    out = run(store, FakeIndex(hits=[]), scripted, [
        '{"action":"search","query":"nothing"}',
        '{"action":"answer"}',
    ])
    assert "No transcribed text" in out["answer"]
    assert out["source_pages"] == []


def test_overview_is_not_cited_as_a_page(store, scripted):
    out = run(store, FakeIndex(), scripted, [
        '{"action":"overview"}',
        '{"action":"answer"}',
    ])
    # page_number 0 is the document-level marker and must not reach source_pages.
    assert 0 not in out["source_pages"]


def test_history_accumulates_across_turns_but_evidence_does_not(store, scripted):
    index = FakeIndex()
    g = graph_mod.build_graph(store, index, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "multi"}, "recursion_limit": 30}

    scripted["replies"] = ['{"action":"search","query":"first"}', '{"action":"answer"}']
    scripted["calls"] = 0
    first = g.invoke({"doc_id": store.doc_id, "question": "Q1", "max_steps": 3, "steps": 0}, cfg)
    assert len(first["history"]) == 2   # user + assistant

    index.hits = [_Hit(2, "second page text", 0.7)]
    scripted["replies"] = ['{"action":"search","query":"second"}', '{"action":"answer"}']
    scripted["calls"] = 0
    second = g.invoke({"doc_id": store.doc_id, "question": "Q2", "max_steps": 3, "steps": 0}, cfg)

    assert len(second["history"]) == 4                # memory kept
    assert second["source_pages"] == [2]              # turn 1's page NOT carried over
    assert [t["args"]["query"] for t in second["trace"]] == ["second"]
