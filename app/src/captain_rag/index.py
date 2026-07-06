"""Chunk journal pages, embed, store in FAISS. One global index over the whole
corpus (unlike the per-document index in the inkferenceApp example, since this
corpus is a single set of journal books, not multiple uploaded documents).

sentence-transformers / faiss imports are lazy so importing this module stays cheap.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import INDEX_DIR, RAGConfig
from .config import rag as default_rag
from .corpus import Page, load_pages

_CHUNK_CHARS = 600
_CHUNK_OVERLAP_LINES = 1


@dataclass
class Chunk:
    page_id: str
    text: str


@dataclass
class Retrieved:
    page_id: str
    text: str
    score: float


def chunk_page(page_id: str, text: str) -> list[Chunk]:
    """Split a page into <=_CHUNK_CHARS chunks on line boundaries."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    for ln in lines:
        if size + len(ln) > _CHUNK_CHARS and buf:
            chunks.append(Chunk(page_id, "\n".join(buf)))
            buf = buf[-_CHUNK_OVERLAP_LINES:] if _CHUNK_OVERLAP_LINES else []
            size = sum(len(x) for x in buf)
        buf.append(ln)
        size += len(ln)
    if buf:
        chunks.append(Chunk(page_id, "\n".join(buf)))
    return chunks


class RagIndex:
    def __init__(self, cfg: RAGConfig = default_rag, index_dir: Path = INDEX_DIR) -> None:
        self.cfg = cfg
        self.index_dir = index_dir
        self._embedder = None
        self._index = None
        self._chunks: list[Chunk] = []

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.cfg.embed_model_id)
        return self._embedder

    def _embed(self, texts: list[str]):
        import numpy as np

        emb = self._ensure_embedder().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return emb.astype(np.float32)

    def _paths(self) -> tuple[Path, Path]:
        return self.index_dir / "index.faiss", self.index_dir / "chunks.json"

    def build(self, pages: list[Page] | None = None) -> int:
        """Build (or rebuild) the index from the transcriptions corpus."""
        import faiss

        pages = pages if pages is not None else load_pages()
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(chunk_page(page.page_id, page.text))
        if not chunks:
            self._index, self._chunks = None, []
            return 0

        vectors = self._embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])  # cosine via normalized vectors
        index.add(vectors)

        self._index, self._chunks = index, chunks
        self._persist()
        return len(chunks)

    def _persist(self) -> None:
        import faiss

        idx_path, meta_path = self._paths()
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(idx_path))
        meta_path.write_text(json.dumps([asdict(c) for c in self._chunks]), encoding="utf-8")

    def exists(self) -> bool:
        idx_path, meta_path = self._paths()
        return idx_path.exists() and meta_path.exists()

    def _ensure_loaded(self) -> bool:
        if self._index is not None:
            return True
        import faiss

        idx_path, meta_path = self._paths()
        if not idx_path.exists() or not meta_path.exists():
            return False
        self._index = faiss.read_index(str(idx_path))
        self._chunks = [Chunk(**c) for c in json.loads(meta_path.read_text("utf-8"))]
        return True

    def query(self, question: str, top_k: int | None = None) -> list[Retrieved]:
        k = top_k or self.cfg.top_k
        if not self._ensure_loaded() or self._index is None:
            return []
        qv = self._embed([question])
        k = min(k, len(self._chunks))
        scores, idxs = self._index.search(qv, k)
        out: list[Retrieved] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            c = self._chunks[int(i)]
            out.append(Retrieved(page_id=c.page_id, text=c.text, score=float(score)))
        return out

    def query_filtered(
        self, question: str, allowed_page_ids: set[str], top_k: int | None = None
    ) -> list[Retrieved]:
        """Like query(), but only rank chunks whose page belongs to
        allowed_page_ids — combines metadata filtering (e.g. "only August
        pages") with semantic ranking within that subset, instead of choosing
        between filter-then-sample and pure similarity search."""
        k = top_k or self.cfg.top_k
        if not self._ensure_loaded() or self._index is None:
            return []
        mask = [i for i, c in enumerate(self._chunks) if c.page_id in allowed_page_ids]
        if not mask:
            return []

        import numpy as np

        vectors = self._index.reconstruct_n(0, self._index.ntotal)
        qv = self._embed([question])[0]
        sub_scores = vectors[mask] @ qv
        k = min(k, len(mask))
        top = np.argsort(-sub_scores)[:k]
        out: list[Retrieved] = []
        for pos in top:
            i = mask[int(pos)]
            c = self._chunks[i]
            out.append(Retrieved(page_id=c.page_id, text=c.text, score=float(sub_scores[pos])))
        return out
