"""Orchestrate one RAG turn: retrieve -> generate -> answer + source pages."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RAGConfig
from ..config import rag as default_rag
from .index import RagIndex
from .llm import generate_answer


@dataclass
class Answer:
    question: str
    answer: str
    source_pages: list[int] = field(default_factory=list)
    persona: str | None = None  # e.g. "cook" -> shown with an IN CHARACTER tag
    # retrieved evidence, for transparency / debugging
    contexts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_pages": self.source_pages,
            "persona": self.persona,
            "in_character": bool(self.persona),
            "contexts": self.contexts,
        }


def answer_question(
    doc_id: int,
    question: str,
    index: RagIndex,
    cfg: RAGConfig = default_rag,
    top_k: int | None = None,
    persona: str | None = None,
) -> Answer:
    retrieved = index.query(doc_id, question, top_k=top_k)
    if not retrieved:
        return Answer(question, "No transcribed text is available for this document yet.",
                      persona=persona)

    contexts = [(r.page_number, r.text) for r in retrieved]
    text = generate_answer(question, contexts, cfg, persona=persona)

    # Distinct source pages in retrieval order -> the design's "Sources" chips.
    seen: set[int] = set()
    source_pages: list[int] = []
    for r in retrieved:
        if r.page_number not in seen:
            seen.add(r.page_number)
            source_pages.append(r.page_number)

    return Answer(
        question=question,
        answer=text,
        source_pages=source_pages,
        persona=persona,
        contexts=[{"page_number": r.page_number, "score": round(r.score, 4),
                   "text": r.text} for r in retrieved],
    )
