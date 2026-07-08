"""Backfill Qwen post-correction over already-recognized pages in the store.

Runs the corrector (Groq/API by config) over pages that already have TrOCR text,
without touching Kraken/TrOCR — so it's memory-light (good on small machines where
the live pipeline OOMs) and lets the Reader's Raw/Corrected toggle + green edits
appear on pages that were ingested before correction was enabled.

Usage (from repo root):
  python app/scripts/correct_corpus.py --slug model-check
  python app/scripts/correct_corpus.py --doc-id 2 --pages 14,15,16
"""
from __future__ import annotations

import argparse
import time

from inkference.config import htr as htr_cfg
from inkference.htr.align import align_line
from inkference.htr.correction import PostCorrector
from inkference.schemas import Line, PageResult, Word
from inkference.store import DocumentStore


def _page_to_result(store: DocumentStore, doc_id: int, page_number: int) -> tuple[int, PageResult] | None:
    page = store.get_page(doc_id, page_number)
    if not page or not page.get("lines"):
        return None
    lines = []
    for ln in page["lines"]:
        words = [Word(w["text"], w["confidence"], bool(w["needs_review"])) for w in ln["words"]]
        lines.append(Line(index=ln["idx"], bbox=tuple(ln["bbox"]), text=ln["text"],
                          confidence=ln["confidence"], words=words,
                          needs_review=bool(ln["needs_review"])))
    pr = PageResult(page_number, page["width"] or 0, page["height"] or 0,
                    lines=lines, scale=page["scale"] or 1.0)
    return page["id"], pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--doc-id", type=int)
    ap.add_argument("--pages", help="comma-separated page numbers (default: all completed)")
    ap.add_argument("--pace", type=float, default=4.0,
                    help="seconds to wait between pages (free-tier rate limits)")
    args = ap.parse_args()

    store = DocumentStore()
    if args.slug:
        doc = store.get_document_by_slug(args.slug)
        if not doc:
            raise SystemExit(f"no document with slug {args.slug!r}")
        doc_id = doc["id"]
    elif args.doc_id:
        doc_id = args.doc_id
    else:
        raise SystemExit("pass --slug or --doc-id")

    doc = store.get_document(doc_id)
    if not doc:
        raise SystemExit(f"document {doc_id} not found")
    page_numbers = (
        [int(p) for p in args.pages.split(",")] if args.pages
        else [p["page_number"] for p in doc["pages"] if p["status"] == "complete"]
    )

    corrector = PostCorrector()  # uses CorrectionConfig (Groq/API by env)
    thr = htr_cfg.low_confidence_threshold
    changed = 0
    for pn in page_numbers:
        got = _page_to_result(store, doc_id, pn)
        if not got:
            print(f"  page {pn}: no lines, skipped")
            continue
        page_id, pr = got
        try:
            res = corrector.correct_page(pr.lines)
        except Exception as exc:  # keep going on rate limits / transient errors
            print(f"  page {pn}: error ({str(exc)[:80]}) — skipped")
            time.sleep(args.pace)
            continue
        if res.lines is None:
            print(f"  page {pn}: unaligned ({len(res.page_text.splitlines())} vs {len(pr.lines)} lines) — skipped")
            time.sleep(args.pace)
            continue
        edits = 0
        for line, corrected in zip(pr.lines, res.lines):
            line.corrected_text = corrected
            line.corrected_words = align_line(line.words, corrected, thr)
            edits += sum(1 for w in line.corrected_words if w.qwen_replaced)
        store.save_page_result(page_id, pr)
        changed += 1
        print(f"  page {pn}: corrected ({edits} word edits)")
        time.sleep(args.pace)  # pace calls to respect free-tier rate limits

    print(f"\nDone. Corrected {changed}/{len(page_numbers)} pages in '{doc['slug']}'.")


if __name__ == "__main__":
    main()
