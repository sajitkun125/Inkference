"""Agent tools over the existing corpus. No retrieval logic is reimplemented here.

- search      -> RagIndex.query          (rag/index.py)
- read_page   -> DocumentStore.get_page  (store/store.py)
- read_range  -> read_page in a loop, clamped and book-aware
- overview    -> DocumentStore.get_document + the book map

This module also owns the TOKEN BUDGET. Every observation is truncated and the
accumulated evidence is capped, which is what keeps a multi-step run inside a free
Groq tier's per-minute token allowance.
"""
from __future__ import annotations

import logging
import re
import threading

from ..config import AgentConfig
from ..config import agent as default_agent
from ..store import DocumentStore

logger = logging.getLogger("inkference.agent")

# --------------------------------------------------------------------------- #
# book map
# --------------------------------------------------------------------------- #
# Seeded pages store a RELATIVE image key "book<N>/forster<N>/B<N>_P_<NNN>.jpg"
# (store/seed_all_books.py), so which of the six books a page belongs to is
# recoverable at runtime. We derive it from image_path rather than adding a field
# to rag.index.Chunk, because changing Chunk would invalidate the prebuilt FAISS
# index baked into the HF seed dataset and force a full re-embed + re-upload.
_BOOK_KEY_RE = re.compile(r"^book(\d+)/", re.IGNORECASE)

_book_cache: dict[int, dict[int, tuple[int, int]]] = {}
_book_lock = threading.Lock()


def book_map(doc_id: int, store: DocumentStore) -> dict[int, tuple[int, int]]:
    """{book_number: (first_page, last_page)} for a document, cached per process.

    Empty when the document wasn't seeded from the book corpus (e.g. a document of
    user-uploaded scans, whose image_path is absolute).
    """
    with _book_lock:
        cached = _book_cache.get(doc_id)
    if cached is not None:
        return cached

    spans: dict[int, tuple[int, int]] = {}
    for page_number, image_path in store.page_image_keys(doc_id):
        m = _BOOK_KEY_RE.match(image_path or "")
        if not m:
            continue  # uploaded page (absolute path) — no book
        book = int(m.group(1))
        lo, hi = spans.get(book, (page_number, page_number))
        spans[book] = (min(lo, page_number), max(hi, page_number))

    with _book_lock:
        _book_cache[doc_id] = spans
    logger.debug("agent: book map for doc %s -> %s", doc_id, spans)
    return spans


def book_for_page(doc_id: int, store: DocumentStore, page: int) -> int | None:
    for book, (lo, hi) in book_map(doc_id, store).items():
        if lo <= page <= hi:
            return book
    return None


def _cite(doc_id: int, store: DocumentStore, page: int) -> str:
    """Human label for a page: "Book 3, p. 47 (corpus p. 412)" when we know the book."""
    book = book_for_page(doc_id, store, page)
    if book is None:
        return f"Page {page}"
    lo, _hi = book_map(doc_id, store)[book]
    return f"Book {book}, p. {page - lo + 1} (corpus p. {page})"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"


def _page_text(page: dict) -> str:
    """Prefer post-corrected text, mirroring DocumentStore.iter_pages_text."""
    if page.get("corrected_text"):
        return page["corrected_text"]
    lines = page.get("lines") or []
    return "\n".join(ln.get("text", "") for ln in lines)


def budget_evidence(evidence: list[dict], cfg: AgentConfig = default_agent) -> list[dict]:
    """Cap total evidence characters, evicting the lowest-scoring items first.

    Kept in retrieval order for whatever survives, because compose passes contexts
    positionally and the prompt reads better when the best passage is first.
    """
    total = sum(len(e.get("text", "")) for e in evidence)
    if total <= cfg.evidence_chars:
        return evidence
    ranked = sorted(evidence, key=lambda e: e.get("score", 0.0), reverse=True)
    kept: list[dict] = []
    running = 0
    for item in ranked:
        size = len(item.get("text", ""))
        if running + size > cfg.evidence_chars:
            continue
        kept.append(item)
        running += size
    keep_ids = {id(k) for k in kept}
    dropped = len(evidence) - len(kept)
    if dropped:
        logger.debug("agent: evidence budget dropped %d passage(s)", dropped)
    return [e for e in evidence if id(e) in keep_ids]


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
def search(
    doc_id: int, store: DocumentStore, index, query: str,
    k: int | None = None, cfg: AgentConfig = default_agent,
) -> dict:
    """Semantic search. Applies a score floor in the TOOL layer only, so POST /ask
    keeps its current (unfiltered) recall."""
    k = max(1, min(int(k or cfg.search_k), 20))
    hits = index.query(doc_id, query, top_k=k)
    kept = [h for h in hits if h.score >= cfg.score_floor]
    # A floor that removes everything is worse than no floor — the model then has
    # nothing to reason about. Keep the single best hit so it can judge for itself.
    if not kept and hits:
        kept = hits[:1]

    passages = [
        {
            "page_number": h.page_number,
            "text": _truncate(h.text, cfg.page_chars),
            "score": round(float(h.score), 4),
            "via": "search",
            "cite": _cite(doc_id, store, h.page_number),
        }
        for h in kept
    ]
    return {
        "passages": passages,
        "label": f"Searching “{_truncate(query, 60)}”",
        "pages": [p["page_number"] for p in passages],
        "note": "" if passages else "no passages matched",
    }


def read_page(
    doc_id: int, store: DocumentStore, page: int, cfg: AgentConfig = default_agent,
) -> dict:
    row = store.get_page(doc_id, int(page))
    if not row:
        return {"passages": [], "label": f"Page {page} (not found)",
                "pages": [], "note": f"page {page} does not exist in this document"}
    text = _page_text(row)
    if not text.strip():
        return {"passages": [], "label": f"Page {page} (no transcription)",
                "pages": [], "note": f"page {page} has no transcribed text"}
    return {
        "passages": [{
            "page_number": int(page),
            "text": _truncate(text, cfg.page_chars),
            # Directly requested, so rank it above search hits (cosine maxes at 1.0).
            "score": 1.0,
            "via": "read_page",
            "cite": _cite(doc_id, store, int(page)),
        }],
        "label": f"Reading {_cite(doc_id, store, int(page))}",
        "pages": [int(page)],
        "note": "",
    }


def read_range(
    doc_id: int, store: DocumentStore, start: int, end: int,
    cfg: AgentConfig = default_agent,
) -> dict:
    """Consecutive pages in order — the tool that answers "what happened next".

    Clamped to cfg.max_span pages and never allowed to cross a book boundary, so a
    hallucinated range can't drag half the corpus into the prompt.
    """
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    span_end = min(end, start + cfg.max_span - 1)
    capped_by_span = span_end < end

    book = book_for_page(doc_id, store, start)
    capped_by_book = False
    if book is not None:
        _lo, hi = book_map(doc_id, store)[book]
        if span_end > hi:
            span_end = hi
            capped_by_book = True

    passages: list[dict] = []
    missing: list[int] = []
    for page in range(start, span_end + 1):
        one = read_page(doc_id, store, page, cfg)
        if one["passages"]:
            passages.extend(one["passages"])
        else:
            missing.append(page)

    # Notes say WHY the range was shortened, never which pages were read — the label
    # already carries those, so repeating them is noise in the UI trace and wasted
    # tokens for the planner. Book boundary is reported ahead of the span cap because
    # it is the more informative reason when both apply.
    notes = []
    if capped_by_book:
        notes.append(f"stopped at the end of Book {book}")
    elif capped_by_span:
        notes.append(f"clamped to {cfg.max_span} pages per read")
    if missing:
        notes.append(f"no text on page(s) {missing}")
    return {
        "passages": passages,
        "label": f"Reading pages {start}–{span_end}",
        "pages": [p["page_number"] for p in passages],
        "note": "; ".join(notes),
    }


def overview(doc_id: int, store: DocumentStore, cfg: AgentConfig = default_agent) -> dict:
    doc = store.get_document(doc_id)
    if not doc:
        return {"passages": [], "label": "Overview (document not found)",
                "pages": [], "note": "document not found"}
    books = book_map(doc_id, store)
    lines = [
        f"Title: {doc.get('title')}",
        f"Subtitle: {doc.get('subtitle') or '—'}",
        f"Pages: {doc.get('page_count')}",
    ]
    if books:
        lines.append("Books (corpus page ranges):")
        lines += [f"  Book {b}: pages {lo}–{hi}" for b, (lo, hi) in sorted(books.items())]
    else:
        lines.append("This document has no book structure (uploaded pages).")
    return {
        "passages": [{
            "page_number": 0,          # 0 = document-level, not a citable page
            "text": "\n".join(lines),
            "score": 0.0,
            "via": "overview",
            "cite": "Document overview",
        }],
        "label": "Reading the document overview",
        "pages": [],
        "note": "",
    }


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def run_action(action: dict, doc_id: int, store: DocumentStore, index,
               cfg: AgentConfig = default_agent) -> dict:
    """Execute a validated action. Tool errors become observations, never exceptions —
    the model gets to see what went wrong and try something else."""
    name = action.get("action")
    try:
        if name == "search":
            return search(doc_id, store, index, action["query"], action.get("k"), cfg)
        if name == "read_page":
            return read_page(doc_id, store, action["page"], cfg)
        if name == "read_range":
            return read_range(doc_id, store, action["start"], action["end"], cfg)
        if name == "overview":
            return overview(doc_id, store, cfg)
    except Exception as exc:  # a broken tool must not kill the turn
        logger.warning("agent: tool %r failed: %s", name, exc, exc_info=True)
        return {"passages": [], "label": f"{name} failed", "pages": [],
                "note": f"tool error: {exc}"}
    return {"passages": [], "label": f"unknown action {name!r}", "pages": [],
            "note": "unknown action"}
