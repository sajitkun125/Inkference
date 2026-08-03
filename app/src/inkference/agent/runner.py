"""Entry point: run one agent turn and shape the result for the API."""
from __future__ import annotations

import logging
import uuid

from ..config import agent as agent_cfg
from .graph import get_graph
from .state import AgentAnswer

logger = logging.getLogger("inkference.agent")

# Hard stop on graph supersteps. max_steps bounds the tool loop; this bounds the
# graph itself, so a routing mistake can't spin forever.
_RECURSION_HEADROOM = 4


def run_agent(
    doc_id: int,
    question: str,
    store,
    index,
    thread_id: str | None = None,
    persona: str | None = None,
    max_steps: int | None = None,
) -> AgentAnswer:
    thread_id = thread_id or uuid.uuid4().hex
    steps_cap = max(1, min(int(max_steps or agent_cfg.max_steps), 8))
    graph = get_graph(store, index)

    config = {
        # Namespaced by document: a stale thread id from another corpus must not
        # resurrect the wrong conversation.
        "configurable": {"thread_id": f"{doc_id}:{thread_id}"},
        "recursion_limit": steps_cap * 2 + _RECURSION_HEADROOM,
    }
    state = {
        "doc_id": doc_id,
        "question": question,
        "persona": persona,
        "max_steps": steps_cap,
        # Reset per turn; history and evidence-free state carry over via the checkpointer.
        "steps": 0,
        "next_action": {},
        "stop_reason": "",
    }

    logger.info("agent: doc=%s thread=%s persona=%s q=%r",
                doc_id, thread_id, persona, question[:100])
    final = graph.invoke(state, config)
    logger.info("agent: doc=%s steps=%s stop=%s sources=%s",
                doc_id, final.get("steps"), final.get("stop_reason"),
                final.get("source_pages"))

    evidence = final.get("evidence") or []
    return AgentAnswer(
        question=question,
        answer=final.get("answer", ""),
        source_pages=final.get("source_pages", []),
        persona=persona,
        contexts=[
            {"page_number": e["page_number"], "score": e.get("score", 0.0),
             "text": e["text"], "via": e.get("via"), "cite": e.get("cite")}
            for e in evidence
        ],
        thread_id=thread_id,
        standalone_question=final.get("standalone_question"),
        steps=final.get("steps", 0),
        grounded=final.get("grounded", True),
        stop_reason=final.get("stop_reason") or "answer",
        trace=final.get("trace", []),
        books=final.get("books", {}),
    )
