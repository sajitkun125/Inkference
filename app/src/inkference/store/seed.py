"""Seed the store from existing line-crop transcriptions.

We already have (line_crop.jpg, ground_truth.txt) pairs under transcriptions/.
Rather than re-OCR them, this loads them directly so the app ships with a populated
document (Reader + Ask work on first run, before any upload). Per page folder
(e.g. B1_P057) the crops are stitched vertically into a single page image for the
Reader's left pane, and each line's bbox is recorded against that stitched image.

Ground-truth lines get confidence 1.0 (manual transcription, not model output).

Usage:
  python -m inkference.store.seed                 # default Captain Cook corpus
  python -m inkference.store.seed --limit-pages 8 # smaller demo
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..config import TRANSCRIPTIONS_ROOT, StoreConfig
from ..config import store as default_store
from ..schemas import Line, PageResult, Word
from .store import DocumentStore

_MARGIN, _GAP = 40, 26
_LINE_RE = re.compile(r"_L(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def _is_junk(path: Path) -> bool:
    """Skip macOS archive artefacts (__MACOSX dirs, AppleDouble ._ files)."""
    return path.name.startswith("._") or "__MACOSX" in path.parts


def _find_page_dirs(root: Path) -> list[Path]:
    """Directories named like B<book>_P<page> that contain *_L##.jpg crops."""
    dirs: set[Path] = set()
    for img in root.rglob("*_L*.jpg"):
        if _LINE_RE.search(img.name) and not _is_junk(img):
            dirs.add(img.parent)
    return sorted(dirs, key=lambda p: p.name)


def _line_files(page_dir: Path) -> list[tuple[int, Path, Path]]:
    """Sorted (line_no, image, text) triples for a page; only pairs with text."""
    out = []
    for img in page_dir.glob("*_L*.*"):
        m = _LINE_RE.search(img.name)
        if not m or _is_junk(img):
            continue
        txt = img.with_suffix(".txt")
        if txt.exists():
            out.append((int(m.group(1)), img, txt))
    return sorted(out, key=lambda t: t[0])


def _build_page(page_dir: Path, page_number: int, assets_dir: Path) -> PageResult | None:
    from PIL import Image

    triples = _line_files(page_dir)
    if not triples:
        return None

    crops = [(Image.open(img).convert("RGB"), txt.read_text(encoding="utf-8").strip())
             for _, img, txt in triples]
    crops = [(im, t) for im, t in crops if t]  # drop empty GT
    if not crops:
        return None

    width = max(im.width for im, _ in crops) + 2 * _MARGIN
    height = sum(im.height for im, _ in crops) + _GAP * (len(crops) + 1)
    page_img = Image.new("RGB", (width, height), (250, 246, 237))  # paper tone

    lines: list[Line] = []
    y = _GAP
    for idx, (im, text) in enumerate(crops):
        page_img.paste(im, (_MARGIN, y))
        bbox = (_MARGIN, y, _MARGIN + im.width, y + im.height)
        words = [Word(text=w, confidence=1.0, needs_review=False) for w in text.split()]
        lines.append(Line(index=idx, bbox=bbox, text=text, confidence=1.0,
                          words=words, needs_review=False))
        y += im.height + _GAP

    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / f"{page_dir.name}.png"
    page_img.save(out_path)

    result = PageResult(page_number=page_number, image_width=width,
                        image_height=height, lines=lines, scale=1.0)
    result._image_path = str(out_path)  # type: ignore[attr-defined]
    return result


def seed_corpus(
    store: DocumentStore | None = None,
    transcriptions_root: Path | None = None,
    title: str = "Diary of Captain Cook Voyage",
    slug: str = "captain-cook-voyage",
    subtitle: str = "1772–1774",
    limit_pages: int | None = None,
) -> int:
    store = store or DocumentStore()
    root = transcriptions_root or TRANSCRIPTIONS_ROOT

    if store.get_document_by_slug(slug):
        print(f"document '{slug}' already seeded — skipping")
        return store.get_document_by_slug(slug)["id"]

    page_dirs = _find_page_dirs(root)
    if limit_pages:
        page_dirs = page_dirs[:limit_pages]
    if not page_dirs:
        raise SystemExit(f"no page folders with *_L##.jpg + .txt found under {root}")

    doc_id = store.create_document(title=title, slug=slug, subtitle=subtitle)
    assets = store.cfg.assets_dir / slug
    seeded = 0
    for i, pd in enumerate(page_dirs, start=1):
        result = _build_page(pd, page_number=i, assets_dir=assets)
        if result is None:
            continue
        page_id = store.add_page(doc_id, i, image_path=getattr(result, "_image_path", None))
        store.save_page_result(page_id, result)
        seeded += 1
        print(f"  seeded page {i}: {pd.name}  ({len(result.lines)} lines)")

    print(f"\nseeded document '{slug}' (id={doc_id}) with {seeded} pages")
    return doc_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-pages", type=int, default=None)
    ap.add_argument("--slug", default="captain-cook-voyage")
    args = ap.parse_args()
    seed_corpus(slug=args.slug, limit_pages=args.limit_pages)


if __name__ == "__main__":
    main()
