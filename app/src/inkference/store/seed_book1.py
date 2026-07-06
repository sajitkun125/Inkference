"""Preseed the store from Alex's real Book-1 data (36 pages).

Sources (default ~/Downloads/AlexFiles):
  book1/forster1/B1_P_<NNN>.jpg          page scans
  moredata/confidence/B1_P<NNN>.txt      raw words + per-word confidence ([L##] word(conf), <<low>>(conf))
  moredata/corrected_transcriptions/     verified corrected transcription (the 36 pages)

Raw and corrected have different line structures, so the corrected "green" view is
built with page-level word alignment (align_page, per post_correction_full_test_v2).

Each page stores: raw lines (confidence view) + corrected display lines (green) + a
page image, then the RAG index is (re)built over the corrected text.

Usage (from repo root, into a fresh data dir):
  INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_book1 \
    python app/scripts/seed_book1.py --alex ~/Downloads/AlexFiles
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..htr.align import align_page
from ..schemas import Line, PageResult, Word
from ..store import DocumentStore

_TOKEN_RE = re.compile(r"<<([^>]+)>>\(([\d.]+)\)|([^\s(]+)\(([\d.]+)\)")
_LINE_RE = re.compile(r"^\[L\d+\]\s?(.*)$")


def parse_confidence(path: Path) -> tuple[float, list[list[tuple[str, float]]]]:
    """Return (threshold, lines) where each line is a list of (word, confidence)."""
    threshold = 0.7
    lines: list[list[tuple[str, float]]] = []
    for row in path.read_text(encoding="utf-8").splitlines():
        if row.startswith("SCHWELLE:"):
            try:
                threshold = float(row.split(":", 1)[1])
            except ValueError:
                pass
            continue
        m = _LINE_RE.match(row)
        if not m:
            continue
        words: list[tuple[str, float]] = []
        for mt in _TOKEN_RE.finditer(m.group(1)):
            if mt.group(1):
                words.append((mt.group(1).strip(), float(mt.group(2))))
            else:
                words.append((mt.group(3).strip(), float(mt.group(4))))
        lines.append(words)
    return threshold, lines


def parse_corrected(path: Path) -> list[str]:
    out = []
    for row in path.read_text(encoding="utf-8").splitlines():
        row = row.rstrip()
        if row.startswith("===") or not row.strip():
            continue
        out.append(row.strip())
    return out


def _image_bytes(stem: str, alex: Path) -> bytes | None:
    """Return the page image bytes. Prefers a real .jpg (local dev), but falls back
    to a base64 sidecar (.jpg.b64) — used for the HF Space build, where binary files
    get stored as Git-LFS pointers and aren't materialized, whereas base64 text is."""
    # data stem B1_P012 -> image B1_P_012.jpg
    num = stem[len("B1_P"):]
    base = alex / "book1" / "forster1"
    jpg = base / f"B1_P_{num}.jpg"
    if jpg.exists():
        data = jpg.read_bytes()
        # Ignore Git-LFS pointer files (text stub, not the image) — fall through to .b64.
        if not data.startswith(b"version https://git-lfs"):
            return data
    b64 = base / f"B1_P_{num}.jpg.b64"
    if b64.exists():
        import base64
        return base64.b64decode(b64.read_text())
    return None


def build_page(stem: str, alex: Path, assets: Path, page_number: int):
    from PIL import Image

    conf_path = alex / "moredata" / "confidence" / f"{stem}.txt"
    corr_path = alex / "moredata" / "corrected_transcriptions" / f"{stem}.txt"
    if not conf_path.exists() or not corr_path.exists():
        return None
    threshold, conf_lines = parse_confidence(conf_path)
    corrected_lines = parse_corrected(corr_path)
    flat_conf = [wc for line in conf_lines for wc in line]

    img_data = _image_bytes(stem, alex)
    if img_data is None:
        return None
    from io import BytesIO
    with Image.open(BytesIO(img_data)) as im:
        W, H = im.size
    assets.mkdir(parents=True, exist_ok=True)
    dest = assets / f"{stem}.jpg"
    dest.write_bytes(img_data)  # store the decoded image for the Reader to serve

    # raw lines (Raw view) with synthetic vertical bands
    rowh = H / max(1, len(conf_lines))
    raw_lines: list[Line] = []
    for i, words in enumerate(conf_lines):
        ws = [Word(w, c, c < threshold) for w, c in words]
        raw_lines.append(Line(index=i, bbox=(0, int(i * rowh), W, int((i + 1) * rowh)),
                              text=" ".join(w for w, _ in words),
                              confidence=(sum(c for _, c in words) / len(words) if words else 0.0),
                              words=ws))

    # corrected display lines (green) via page-level alignment
    corrected_words = align_page(flat_conf, corrected_lines, threshold)

    return {
        "page_number": page_number, "width": W, "height": H, "image_path": str(dest),
        "raw_lines": raw_lines, "corrected_words": corrected_words,
        "corrected_text": "\n".join(corrected_lines),
    }


def seed_book1(alex_root: Path, store: DocumentStore | None = None,
               title: str = "Forster — Voyage Round the World (Book 1)",
               slug: str = "forster-book1",
               subtitle: str = "Captain Cook's 2nd voyage · 1772–1775") -> int:
    store = store or DocumentStore()
    if store.get_document_by_slug(slug):
        print(f"'{slug}' already seeded")
        return store.get_document_by_slug(slug)["id"]

    corr_dir = alex_root / "moredata" / "corrected_transcriptions"
    stems = sorted(p.stem for p in corr_dir.glob("B1_P*.txt"))
    if not stems:
        raise SystemExit(f"no corrected_transcriptions found under {corr_dir}")

    doc_id = store.create_document(title=title, slug=slug, subtitle=subtitle)
    assets = store.cfg.assets_dir / slug
    n = 0
    for i, stem in enumerate(stems, start=1):
        page = build_page(stem, alex_root, assets, page_number=i)
        if page is None:
            print(f"  {stem}: missing data/image, skipped")
            continue
        page_id = store.add_page(doc_id, i, image_path=page["image_path"])
        store.save_full_page(page_id, page["width"], page["height"],
                             page["raw_lines"], page["corrected_words"], page["corrected_text"])
        greens = sum(1 for ln in page["corrected_words"] for w in ln if w.qwen_replaced)
        print(f"  page {i:2d} ({stem}): {len(page['raw_lines'])} raw lines, {greens} corrections")
        n += 1

    # build RAG index over the corrected text
    from ..rag.index import RagIndex
    chunks = RagIndex().build_from_store(doc_id, store)
    print(f"\nseeded '{slug}' (id={doc_id}) with {n} pages; RAG index: {chunks} chunks")
    return doc_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alex", default=str(Path.home() / "Downloads" / "AlexFiles"))
    args = ap.parse_args()
    seed_book1(Path(args.alex).expanduser())


if __name__ == "__main__":
    main()
