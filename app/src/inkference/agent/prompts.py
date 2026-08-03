"""Prompts for the agent's decision nodes.

The ANSWER prompt is not here on purpose: composing the final answer reuses
rag.llm.generate_answer, so the persona (_SYSTEM_COOK) and the extractive fallback
come along unchanged.
"""
from __future__ import annotations

from .protocol import render_tool_catalog

PLAN_SYSTEM = """You are the research planner for Inkference, a searchable archive of a \
handwritten 18th-century ship's journal transcribed by OCR.

Your job is NOT to answer the question. Your job is to choose the single next action \
that gathers the evidence needed to answer it.

Available actions:
{catalog}

Rules:
- Reply with ONLY a JSON object. No prose, no markdown fence.
- The journal is a DIARY in page order. For "what happened next", "the days after", \
or any question about a sequence of events, use read_range around the relevant pages \
rather than another search.
- Use search to FIND where a topic is discussed; use read_page/read_range to actually \
READ what it says.
- The text is OCR of archaic handwriting, so spelling varies. If a search returns \
little, try different period wording rather than repeating the same query.
- Choose "answer" as soon as the evidence is enough. Do not gather more than you need.
- Do not repeat an action you have already performed."""

PLAN_FIRST_STEP_EXTRA = """
This is the first step for this question. Also include a "standalone_question" key: \
the user's question rewritten so it stands on its own, with every pronoun and every \
reference to an earlier turn replaced by what it actually refers to. This matters \
most for follow-ups — "and the weather there?" after a question about Plymouth must \
become "What was the weather like at Plymouth?". If the question already stands \
alone, repeat it unchanged.

Example: {{"standalone_question": "What was the weather like at Plymouth?", \
"action": "search", "query": "Plymouth weather", "k": 6}}"""


def plan_system(first_step: bool, actions: list[str] | None = None) -> str:
    base = PLAN_SYSTEM.format(catalog=render_tool_catalog(actions))
    return base + (PLAN_FIRST_STEP_EXTRA if first_step else "")


def render_history(history: list[dict], limit: int) -> str:
    """Recent turns, oldest first. Used for coreference, not as evidence."""
    if not history:
        return ""
    recent = history[-limit:]
    turns = "\n".join(
        f"{h.get('role', 'user').capitalize()}: {h.get('content', '')}" for h in recent
    )
    return f"Earlier in this conversation:\n{turns}\n\n"


def render_evidence_digest(evidence: list[dict], per_item: int = 120) -> str:
    """A DIGEST, not the passages themselves.

    The planner only needs to know what it already has in order to decide what to do
    next. Sending full text here would multiply the token cost of every step and is
    the main thing that makes naive agent loops expensive.
    """
    if not evidence:
        return "Evidence gathered so far: none.\n\n"
    lines = []
    for item in evidence:
        snippet = " ".join((item.get("text") or "").split())[:per_item]
        lines.append(f"- {item.get('cite', 'Page ?')}: {snippet}…")
    return "Evidence gathered so far:\n" + "\n".join(lines) + "\n\n"


def render_actions_taken(trace: list[dict]) -> str:
    if not trace:
        return ""
    done = "\n".join(
        f"- {t.get('action')} {t.get('args', {})} -> {t.get('note') or 'ok'}"
        for t in trace
    )
    return f"Actions already taken:\n{done}\n\n"


def plan_prompt(
    question: str,
    history_block: str,
    evidence_block: str,
    actions_block: str,
    steps_left: int,
) -> str:
    urgency = (
        "You have no steps left — you must reply with {\"action\":\"answer\"}."
        if steps_left <= 0
        else f"Steps remaining before you must answer: {steps_left}."
    )
    return (
        f"{history_block}"
        f"Question: {question}\n\n"
        f"{actions_block}"
        f"{evidence_block}"
        f"{urgency}\n\n"
        "Reply with the next action as a single JSON object."
    )
