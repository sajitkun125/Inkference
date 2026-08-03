"""The agent graph.

    prepare -> plan -+-(tool)-> act -+-(steps/time left)-> plan
                     |               `-(budget spent)---> compose -> END
                     `-(answer)------------------------> compose -> END

Grading, query rewriting and groundedness verification are deliberately not here
yet — this shape is the loop that makes narrative questions answerable, and it is
worth proving on the real corpus before adding more LLM calls per turn.
"""
from __future__ import annotations

import logging
import threading

from langgraph.graph import END, START, StateGraph

from . import nodes
from .checkpoint import get_checkpointer
from .state import AgentState

logger = logging.getLogger("inkference.agent")

_graph = None
_graph_lock = threading.Lock()

def build_graph(store, index, checkpointer=None):
    """Compile the graph.

    The corpus store and index are bound into the nodes that need them via closures,
    so every node LangGraph sees is a plain (state) -> partial-state callable.
    (They are bound as `doc_store`/`rag_index`, never `store`: LangGraph injects its
    own BaseStore into a node parameter named `store` and would shadow ours.)
    """
    builder = StateGraph(AgentState)

    builder.add_node("prepare", lambda s: nodes.prepare(s, store, index))
    builder.add_node("plan", nodes.plan)
    builder.add_node("act", lambda s: nodes.act(s, store, index))
    builder.add_node("compose", nodes.compose)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "plan")
    builder.add_conditional_edges(
        "plan", nodes.route_after_plan, {"act": "act", "compose": "compose"}
    )
    builder.add_conditional_edges(
        "act", nodes.route_after_act, {"plan": "plan", "compose": "compose"}
    )
    builder.add_edge("compose", END)

    return builder.compile(checkpointer=checkpointer or get_checkpointer())

def get_graph(store, index):
    """Lazy singleton — compiling is cheap but the checkpointer connection is not."""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_graph(store, index)
                logger.info("agent: graph compiled")
    return _graph
