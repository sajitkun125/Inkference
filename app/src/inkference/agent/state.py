"""Agent graph state, its reducers, and the response shape.

AgentAnswer.to_dict() is a strict SUPERSET of rag.answer.Answer.to_dict(), so the
frontend renders both the fast path and the agent with the same code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict


# --------------------------------------------------------------------------- #
# reducers
# --------------------------------------------------------------------------- #
def merge_evidence(left: list[dict], right: list[dict] | None) -> list[dict]:
    """Accumulate retrieved passages across steps, deduped by (page_number, text).

    The same page legitimately arrives twice — once from a search hit, once from a
    read_page — and we keep the higher score so the compose ordering stays sane.

    `right is None` RESETS. The checkpointer persists state across turns of a
    conversation, so without a reset the second question would inherit (and cite)
    the first question's evidence. `prepare` sends None at the start of every turn.
    """
    if right is None:
        return []
    if not right:
        return left
    merged: dict[tuple[int, str], dict] = {}
    for item in [*(left or []), *right]:
        key = (item.get("page_number", -1), item.get("text", ""))
        prev = merged.get(key)
        if prev is None or item.get("score", 0.0) > prev.get("score", 0.0):
            merged[key] = item
    return list(merged.values())


def append_turns(left: list[dict], right: list[dict]) -> list[dict]:
    """Conversation history. Plain append — the checkpointer persists it per thread.

    Deliberately has no reset: this is the memory the agent exists to provide.
    """
    return [*(left or []), *(right or [])]


def extend_trace(left: list[dict], right: list[dict] | None) -> list[dict]:
    """Per-step audit trail. Resets per turn on None, like merge_evidence."""
    if right is None:
        return []
    return [*(left or []), *right]


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    # -- inputs
    doc_id: int
    question: str                 # the raw user turn
    persona: str | None           # "cook" -> in-character Forster (rag/llm.py:_SYSTEM_COOK)
    max_steps: int

    # -- derived
    standalone_question: str      # coreference-resolved; what actually gets embedded
    history: Annotated[list[dict], append_turns]   # [{role, content}], survives via checkpointer
    evidence: Annotated[list[dict], merge_evidence]  # [{page_number, text, score, via}]
    trace: Annotated[list[dict], extend_trace]      # per-step audit, shown in the UI
    books: dict[str, Any]         # book map: {"1": [1, 132], ...}

    # -- control
    steps: int
    rewrites: int
    verify_attempts: int
    deadline: float               # time.monotonic() ceiling, checked in every node
    next_action: dict             # the action `plan` chose, consumed by `act`
    stop_reason: str              # why the loop ended: answer | max_steps | deadline | llm_unavailable

    # -- outputs
    answer: str
    source_pages: list[int]
    grounded: bool


@dataclass
class AgentAnswer:
    """Superset of rag.answer.Answer so the frontend can share one renderer."""

    question: str
    answer: str
    source_pages: list[int] = field(default_factory=list)
    persona: str | None = None
    contexts: list[dict] = field(default_factory=list)
    # agent-only
    thread_id: str | None = None
    standalone_question: str | None = None
    steps: int = 0
    grounded: bool = True
    stop_reason: str = "answer"
    trace: list[dict] = field(default_factory=list)
    books: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            # --- shared with Answer.to_dict()
            "question": self.question,
            "answer": self.answer,
            "source_pages": self.source_pages,
            "persona": self.persona,
            "in_character": bool(self.persona),
            "contexts": self.contexts,
            # --- agent extras
            "mode": "agent",
            "thread_id": self.thread_id,
            "standalone_question": self.standalone_question,
            "steps": self.steps,
            "grounded": self.grounded,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
            "books": self.books,
        }
