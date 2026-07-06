"""CLI Q&A chat over the transcribed Captain Cook journal pages.

Usage (from the repo root, after `pip install -e ./app`):

    captain-rag              # or: python -m captain_rag.chat
    captain-rag --rebuild    # force a fresh index build
    captain-rag --debug      # also print retrieved snippets / sampled pages

On first run (or with --rebuild) the corpus under data/transcriptions/ is chunked,
embedded, and persisted to app/.rag_index/ so subsequent runs start instantly.

Two kinds of questions bypass plain similarity retrieval (see answer.py docstring
for why it fails them):

- "summarize each book/month/season" (e.g. "fasse jedes Buch zusammen", "jede
  Jahreszeit") -> one grounded one-sentence summary per group.
- a question scoped to one specific month/season (e.g. "wie waren die Winter?",
  "was gab's im August zu essen?") -> answered from just that value's pages,
  ranked by relevance within them.

Everything else falls through to normal vector retrieval.
"""
from __future__ import annotations

import argparse
import re
import sys

from .answer import answer_question, filtered_answer, group_and_summarize
from .config import TRANSCRIPTIONS_ROOT
from .config import rag as default_rag
from .corpus import load_pages, pages_by_month, pages_by_season, parse_month_name, parse_season_name
from .index import RagIndex

_EACH_GROUP_PATTERNS = {
    "book": re.compile(
        r"jedes buch|jedem buch|jedes der b[üu]cher|pro buch|alle b[üu]cher|"
        r"each book|every book|per book|all (of the )?books",
        re.IGNORECASE,
    ),
    "season": re.compile(
        r"jede jahreszeit|jeder jahreszeit|alle jahreszeiten|"
        r"each season|every season|per season|all seasons",
        re.IGNORECASE,
    ),
    "month": re.compile(
        r"jeden monat|jedem monat|alle monate|"
        r"each month|every month|per month|all months",
        re.IGNORECASE,
    ),
}


def _wants_group_summary(question: str) -> str | None:
    """Which facet ("book"/"month"/"season") the question asks to enumerate,
    if any — checked before the specific-value detectors below since "jede
    Jahreszeit" would otherwise also look like it names no particular season."""
    for facet, pattern in _EACH_GROUP_PATTERNS.items():
        if pattern.search(question):
            return facet
    return None


def _build_or_load(index: RagIndex, force_rebuild: bool) -> None:
    if force_rebuild or not index.exists():
        print(f"Indexing pages from {TRANSCRIPTIONS_ROOT} ...")
        n_chunks = index.build()
        print(f"Indexed {n_chunks} chunks.\n")
    else:
        print("Loaded existing index (use --rebuild to re-index).\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="force a fresh index build")
    parser.add_argument("--debug", action="store_true", help="print retrieved snippets")
    parser.add_argument("--top-k", type=int, default=None, help="override RAG_TOP_K")
    parser.add_argument(
        "--samples-per-group", type=int, default=8,
        help="pages sampled per group for 'summarize each book/month/season' style questions",
    )
    args = parser.parse_args()

    index = RagIndex()
    _build_or_load(index, args.rebuild)

    provider = default_rag.llm_provider if default_rag.llm_api_key else "extractive fallback (no API key)"
    print(f"Captain Cook journal Q&A — provider: {provider}. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        facet = _wants_group_summary(question)
        if facet:
            print(f"\nSampling {args.samples_per_group} pages per {facet} (not similarity retrieval) ...\n")
            for s in group_and_summarize(facet, samples_per_group=args.samples_per_group):
                print(f"{s.label}: {s.summary}")
                if args.debug:
                    print(f"  (sampled: {', '.join(s.sampled_pages)})")
            print()
            continue

        season = parse_season_name(question)
        month = parse_month_name(question)
        if season or month:
            pages = pages_by_season(load_pages()).get(season, []) if season else pages_by_month(load_pages()).get(month, [])
            allowed = {p.page_id for p in pages}
            label = season or f"month {month}"
            print(f"\nScoping to {len(allowed)} pages tagged '{label}' ...\n")
            result = filtered_answer(question, allowed, index, top_k=args.top_k)
        else:
            result = answer_question(question, index, top_k=args.top_k)

        print(f"\n{result.answer}")
        if result.source_pages:
            print(f"\nQuellen: {', '.join(result.source_pages)}")
        if args.debug:
            for ctx in result.contexts:
                snippet = ctx["text"].replace("\n", " ")[:120]
                print(f"  [{ctx['page_id']} score={ctx['score']}] {snippet}...")
        print()


if __name__ == "__main__":
    sys.exit(main())
