"""Orchestrate RAG turns: retrieve -> generate -> answer + source pages.

Two request shapes need different handling, since plain similarity retrieval
only ever surfaces the globally closest chunks:

- "summarize each X" (X = book / month / season): retrieval concentrates on
  1-2 groups and misses the rest. group_and_summarize() bypasses retrieval and
  samples pages evenly across every group instead (see corpus.sample_evenly),
  so each group gets its own grounded one-sentence summary.
- "what happened in August" / "how were the winters": a single question
  scoped to one metadata value across the whole corpus. filtered_answer()
  restricts candidate chunks to that value's pages first, then ranks by
  semantic similarity within that subset (index.query_filtered) — filtering
  and retrieval combined, not one instead of the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import RAGConfig
from .config import rag as default_rag
from .corpus import group_pages, load_pages, sample_evenly
from .index import RagIndex, Retrieved
from .llm import generate_answer, generate_group_summary


@dataclass
class Answer:
    question: str
    answer: str
    source_pages: list[str] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_pages": self.source_pages,
            "contexts": self.contexts,
        }


def _answer_from_retrieved(question: str, retrieved: list[Retrieved], cfg: RAGConfig) -> Answer:
    if not retrieved:
        return Answer(question, "No transcribed text is available to answer this question yet.")

    contexts = [(r.page_id, r.text) for r in retrieved]
    text = generate_answer(question, contexts, cfg)

    # Distinct source pages in retrieval order.
    seen: set[str] = set()
    source_pages: list[str] = []
    for r in retrieved:
        if r.page_id not in seen:
            seen.add(r.page_id)
            source_pages.append(r.page_id)

    return Answer(
        question=question,
        answer=text,
        source_pages=source_pages,
        contexts=[
            {"page_id": r.page_id, "score": round(r.score, 4), "text": r.text}
            for r in retrieved
        ],
    )


def answer_question(
    question: str,
    index: RagIndex,
    cfg: RAGConfig = default_rag,
    top_k: int | None = None,
) -> Answer:
    return _answer_from_retrieved(question, index.query(question, top_k=top_k), cfg)


def filtered_answer(
    question: str,
    allowed_page_ids: set[str],
    index: RagIndex,
    cfg: RAGConfig = default_rag,
    top_k: int | None = None,
) -> Answer:
    """Answer scoped to a metadata-filtered subset of pages (e.g. all pages
    mentioning August), ranked by semantic relevance within that subset."""
    retrieved = index.query_filtered(question, allowed_page_ids, top_k=top_k)
    return _answer_from_retrieved(question, retrieved, cfg)


@dataclass
class GroupSummary:
    label: str
    summary: str
    sampled_pages: list[str] = field(default_factory=list)


def group_and_summarize(
    facet: str, cfg: RAGConfig = default_rag, samples_per_group: int = 8
) -> list[GroupSummary]:
    """One grounded one-sentence summary per group of facet ("book" | "month" |
    "season"), sampled evenly across each group's pages (not retrieval-based —
    see module docstring)."""
    groups = group_pages(load_pages(), facet)
    summaries: list[GroupSummary] = []
    for label, pages in groups:
        sampled = sample_evenly(pages, samples_per_group)
        excerpts = [(p.page_id, p.text) for p in sampled]
        summary = generate_group_summary(label, excerpts, cfg)
        summaries.append(
            GroupSummary(label=label, summary=summary, sampled_pages=[p.page_id for p in sampled])
        )
    return summaries
