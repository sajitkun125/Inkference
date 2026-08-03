"""Graph nodes. Each returns a partial AgentState; reducers in state.py merge them.

Every node checks the wall-clock deadline. That is what bounds a run even when a
provider is slow, retrying a 429, or the planner keeps asking for one more search.
"""
from __future__ import annotations

import logging
import time

from ..config import agent as agent_cfg
from ..config import rag as rag_cfg
from ..rag.llm import LLMUnavailable, complete, generate_answer
from . import corpus, prompts
from .protocol import REPAIR_PROMPT, TERMINAL, parse_action
from .state import AgentState

logger = logging.getLogger("inkference.agent")


def _out_of_time(state: AgentState) -> bool:
    return time.monotonic() >= state.get("deadline", 0.0)


def _steps_left(state: AgentState) -> int:
    return max(0, state.get("max_steps", agent_cfg.max_steps) - state.get("steps", 0))


def _signature(action: dict) -> tuple:
    """Identity of an action for repeat detection. Case/space-insensitive on strings
    so "Plymouth" and "plymouth " count as the same search."""
    args = tuple(
        (k, v.strip().lower() if isinstance(v, str) else v)
        for k, v in sorted(action.items())
        if k not in ("action", "why")
    )
    return (action.get("action"), args)


def _already_done(action: dict, trace: list[dict]) -> bool:
    done = {_signature({"action": t.get("action"), **(t.get("args") or {})}) for t in trace}
    return _signature(action) in done


# --------------------------------------------------------------------------- #
def prepare(state: AgentState, doc_store, rag_index) -> dict:
    """Set the time budget, make sure the index exists, load the book map.

    NOTE the parameter names: LangGraph injects its own BaseStore into any node
    parameter literally named `store`, which would silently shadow ours with None.
    """
    doc_id = state["doc_id"]
    if not rag_index.exists(doc_id):
        logger.info("agent: building RAG index for doc %s", doc_id)
        rag_index.build_from_store(doc_id, doc_store)

    books = corpus.book_map(doc_id, doc_store)
    return {
        "deadline": time.monotonic() + agent_cfg.time_budget_s,
        # Reset from any prior standalone_question left in the checkpoint; plan
        # recomputes it for this turn.
        "standalone_question": state["question"],
        "steps": 0,
        "books": {str(b): [lo, hi] for b, (lo, hi) in sorted(books.items())},
        # None = reset (see state.merge_evidence): the checkpointer carries state
        # across turns, and this turn must not cite last turn's pages.
        "evidence": None,
        "trace": None,
        # History is the one thing that must NOT reset. Record the user turn now so
        # it survives even if the run errors out.
        "history": [{"role": "user", "content": state["question"]}],
    }


# --------------------------------------------------------------------------- #
def plan(state: AgentState) -> dict:
    """Ask the model for the next action as JSON."""
    if _out_of_time(state):
        logger.info("agent: deadline hit before planning -> compose")
        return {"next_action": {"action": TERMINAL}, "stop_reason": "deadline"}
    if _steps_left(state) <= 0:
        return {"next_action": {"action": TERMINAL}, "stop_reason": "max_steps"}

    history = state.get("history") or []
    # Ask for the coreference-resolved question on the first PLAN STEP of every turn
    # — that is when it's needed most (a follow-up like "and the weather there?"),
    # and folding it in here avoids a separate rewrite round-trip.
    first_step = state.get("steps", 0) == 0

    question = state.get("standalone_question") or state["question"]
    system = prompts.plan_system(first_step=first_step)
    prompt = prompts.plan_prompt(
        question=question,
        history_block=prompts.render_history(history[:-1], agent_cfg.history_turns),
        evidence_block=prompts.render_evidence_digest(state.get("evidence") or []),
        actions_block=prompts.render_actions_taken(state.get("trace") or []),
        steps_left=_steps_left(state),
    )

    try:
        reply = complete(
            system, prompt, rag_cfg,
            max_tokens=400, temperature=0.0,
            timeout=agent_cfg.llm_timeout_s, label="agent plan",
        )
    except LLMUnavailable as exc:
        # No planner means no agent. Rather than compose from nothing, degrade to a
        # single deterministic retrieval — which is exactly what POST /ask does — so
        # the app still answers with no API key configured at all.
        logger.warning("agent: planner unavailable (%s)", exc)
        if not (state.get("evidence") or []):
            return {
                "next_action": {"action": "search", "query": question},
                "stop_reason": "llm_unavailable",
            }
        return {"next_action": {"action": TERMINAL}, "stop_reason": "llm_unavailable"}

    action, extras = parse_action(reply)
    if action is None:
        # One repair attempt, then give up and answer rather than loop on garbage.
        logger.info("agent: unparseable plan reply (%r), retrying once", reply[:300])
        try:
            reply = complete(
                system, prompt + "\n\n" + REPAIR_PROMPT, rag_cfg,
                max_tokens=400, temperature=0.0,
                timeout=agent_cfg.llm_timeout_s, label="agent plan (repair)",
            )
            action, extras = parse_action(reply)
        except LLMUnavailable:
            action, extras = None, None
        if action is None:
            logger.warning("agent: plan unparseable after repair -> answer")
            action = {"action": TERMINAL, "why": "planner output unparseable"}

    # Telling the planner "do not repeat an action" is not enough — it will happily
    # call overview() four times. An exact repeat can never add evidence, so treat it
    # as the signal that the planner is done.
    if action.get("action") != TERMINAL and _already_done(action, state.get("trace") or []):
        logger.info("agent: planner repeated %s -> compose", action)
        return {"next_action": {"action": TERMINAL, "why": "repeated action"},
                "stop_reason": "repeated_action"}

    out: dict = {"next_action": action}
    if extras and extras.get("standalone_question"):
        out["standalone_question"] = extras["standalone_question"]
        logger.info("agent: standalone_question=%r", out["standalone_question"])
    logger.info("agent: step %d action=%s", state.get("steps", 0) + 1, action)
    return out


# --------------------------------------------------------------------------- #
def act(state: AgentState, doc_store, rag_index) -> dict:
    """Run the chosen tool and fold the result into evidence + trace."""
    action = state.get("next_action") or {"action": TERMINAL}
    result = corpus.run_action(action, state["doc_id"], doc_store, rag_index, agent_cfg)

    step_no = state.get("steps", 0) + 1
    return {
        "steps": step_no,
        # Only the NEW passages: merge_evidence unions with what's already in state,
        # so returning a pre-budgeted full list would just resurrect what was evicted.
        # Budgeting therefore happens at read time, in compose.
        "evidence": result["passages"],
        "trace": [{
            "i": step_no,
            "action": action.get("action"),
            "args": {k: v for k, v in action.items() if k != "action"},
            "label": result["label"],
            "pages": result["pages"],
            "note": result["note"],
        }],
    }


# --------------------------------------------------------------------------- #
def compose(state: AgentState) -> dict:
    """Write the answer. Reuses the fast path's generator, so persona and the
    extractive fallback behave exactly as they do for POST /ask."""
    question = state.get("standalone_question") or state["question"]
    # Rank best-first, then apply the token budget — compose is the only place that
    # sends full passage text to a provider, so it is where the cap has to bite.
    evidence = sorted(
        state.get("evidence") or [], key=lambda e: e.get("score", 0.0), reverse=True
    )
    evidence = corpus.budget_evidence(evidence, agent_cfg)

    if not evidence:
        text = "No transcribed text is available to answer this question yet."
        return {"answer": text, "source_pages": [], "grounded": True,
                "history": [{"role": "assistant", "content": text}]}

    contexts = [(e["page_number"], e["text"]) for e in evidence]
    history = [
        (h.get("role", "user"), h.get("content", ""))
        for h in (state.get("history") or [])[:-1][-agent_cfg.history_turns:]
    ]

    text = generate_answer(
        question, contexts, rag_cfg,
        persona=state.get("persona"),
        history=history or None,
    )

    seen: set[int] = set()
    source_pages: list[int] = []
    for e in evidence:
        page = e["page_number"]
        if page > 0 and page not in seen:
            seen.add(page)
            source_pages.append(page)

    return {
        "answer": text,
        "source_pages": source_pages,
        "grounded": True,
        "stop_reason": state.get("stop_reason") or "answer",
        "history": [{"role": "assistant", "content": text}],
    }


# --------------------------------------------------------------------------- #
# routers
# --------------------------------------------------------------------------- #
def route_after_plan(state: AgentState) -> str:
    action = state.get("next_action") or {}
    if action.get("action") == TERMINAL:
        return "compose"
    return "act"


def route_after_act(state: AgentState) -> str:
    if _out_of_time(state):
        logger.info("agent: deadline hit after step %d -> compose", state.get("steps", 0))
        return "compose"
    if _steps_left(state) <= 0:
        return "compose"
    return "plan"
