"""Preseed ALL 6 books as ONE corpus for Ask-the-Archive over the whole journal.

Default: RAW TrOCR only (no correction). With --with-correction: also fold in the
post-corrected text (align_page → green) for pages that have a corrected file, so the
SAME seeder + the SAME images dataset serve both corpora — just re-run with the flag.

Source (default ~/Downloads/AlexFiles):
  book<N>/forster<N>/B<N>_P_<NNN>.jpg               page scans
  moredata/confidence/B<N>_P<NNN>.txt               raw TrOCR words + per-word confidence
  moredata/corrected_transcriptions/B<N>_P<NNN>.txt corrected text (for --with-correction)

Page images are stored as RELATIVE keys ("book<N>/forster<N>/B<N>_P_<nnn>.jpg") and
resolved at serve time against INKFERENCE_IMAGES_ROOT — so one seeded DB is portable
across local (~/Downloads/AlexFiles) and the Space (a downloaded images dataset).

Usage (from repo root):
  INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_all_books \
    python -m inkference.store.seed_all_books --alex ~/Downloads/AlexFiles
  INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_all_books_corrected \
    python -m inkference.store.seed_all_books --alex ~/Downloads/AlexFiles --with-correction
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..htr.align import align_page
from ..schemas import Line, PageResult, Word
from ..store import DocumentStore
from .seed_book1 import parse_confidence, parse_corrected  # reuse the parsers

_STEM_RE = re.compile(r"^B(\d+)_P(\d+)$")


def _rel_key(stem: str) -> str | None:
    """confidence stem B<n>_P<nnn> -> relative image key book<n>/forster<n>/B<n>_P_<nnn>.jpg"""
    m = _STEM_RE.match(stem)
    if not m:
        return None
    b, p = m.group(1), m.group(2)
    return f"book{b}/forster{b}/B{b}_P_{p}.jpg"


def _sort_key(stem: str) -> tuple[int, int]:
    m = _STEM_RE.match(stem)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 9999)


def build_page(stem: str, alex: Path, page_number: int, with_correction: bool = False):
    from PIL import Image

    conf_path = alex / "moredata" / "confidence" / f"{stem}.txt"
    if not conf_path.exists():
        return None
    threshold, conf_lines = parse_confidence(conf_path)
    if not conf_lines:
        return None
    rel = _rel_key(stem)
    img = (alex / rel) if rel else None
    if img is None or not img.exists():
        return None
    with Image.open(img) as im:
        W, H = im.size

    rowh = H / max(1, len(conf_lines))
    raw_lines: list[Line] = []
    for i, words in enumerate(conf_lines):
        ws = [Word(w, c, c < threshold) for w, c in words]
        raw_lines.append(Line(index=i, bbox=(0, int(i * rowh), W, int((i + 1) * rowh)),
                              text=" ".join(w for w, _ in words),
                              confidence=(sum(c for _, c in words) / len(words) if words else 0.0),
                              words=ws))

    out = {"rel": rel, "W": W, "H": H, "raw_lines": raw_lines,
           "corrected_words": None, "corrected_text": None}
    if with_correction:
        corr_path = alex / "moredata" / "corrected_transcriptions" / f"{stem}.txt"
        if corr_path.exists():
            corrected_lines = parse_corrected(corr_path)
            flat_conf = [wc for line in conf_lines for wc in line]
            out["corrected_words"] = align_page(flat_conf, corrected_lines, threshold)
            out["corrected_text"] = "\n".join(corrected_lines)
    return out


def seed_all_books(
    alex_root: Path, store: DocumentStore | None = None, with_correction: bool = False,
    title: str = "Forster — Complete Journal (Books 1–6)",
) -> int:
    store = store or DocumentStore()
    slug = "all-books-with-post-correction" if with_correction else "all-books-without-post-correction"
    subtitle = "TrOCR + Qwen post-correction" if with_correction else "Raw TrOCR · no post-correction"
    if store.get_document_by_slug(slug):
        print(f"'{slug}' already seeded")
        return store.get_document_by_slug(slug)["id"]

    conf_dir = alex_root / "moredata" / "confidence"
    stems = sorted((p.stem for p in conf_dir.glob("B*_P*.txt")), key=_sort_key)
    if not stems:
        raise SystemExit(f"no confidence files under {conf_dir}")

    doc_id = store.create_document(title=title, slug=slug, subtitle=subtitle)
    n, n_corr = 0, 0
    for i, stem in enumerate(stems, start=1):
        page = build_page(stem, alex_root, i, with_correction)
        if page is None:
            continue
        page_id = store.add_page(doc_id, i, image_path=page["rel"])  # relative key
        if page["corrected_words"] is not None:
            store.save_full_page(page_id, page["W"], page["H"], page["raw_lines"],
                                 page["corrected_words"], page["corrected_text"])
            n_corr += 1
        else:
            store.save_page_result(
                page_id, PageResult(i, page["W"], page["H"], lines=page["raw_lines"], scale=1.0))
        n += 1
        if n % 50 == 0:
            print(f"  seeded {n}/{len(stems)} pages…", flush=True)

    from ..rag.index import RagIndex
    chunks = RagIndex().build_from_store(doc_id, store)
    extra = f" ({n_corr} with corrections)" if with_correction else ""
    print(f"\nseeded '{slug}' (id={doc_id}) with {n} pages{extra}; RAG index: {chunks} chunks")
    return doc_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alex", default=str(Path.home() / "Downloads" / "AlexFiles"))
    ap.add_argument("--with-correction", action="store_true",
                    help="also fold in post-corrected text where available")
    args = ap.parse_args()
    seed_all_books(Path(args.alex).expanduser(), with_correction=args.with_correction)


if __name__ == "__main__":
    main()