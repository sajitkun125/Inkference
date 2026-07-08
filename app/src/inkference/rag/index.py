"""Per-document retrieval index: chunk page text, embed, store in FAISS.

Chunks carry their source `page_number` so answers can cite pages (the design's
"Sources: Page 24" chips). Index + chunk metadata persist under
StoreConfig.index_dir/doc_<id>.{faiss,json} so a free Space can rebuild lazily.

sentence-transformers / faiss imports are lazy.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..config import RAGConfig, StoreConfig
from ..config import rag as default_rag
from ..config import store as default_store

if TYPE_CHECKING:  # pragma: no cover
    from ..store import DocumentStore

# Chunk target size in characters. Diary pages are short; one chunk per page is
# common, but long pages split on line boundaries so retrieval stays focused.
_CHUNK_CHARS = 600
_CHUNK_OVERLAP_LINES = 1


@dataclass
class Chunk:
    page_number: int
    text: str


@dataclass
class Retrieved:
    page_number: int
    text: str
    score: float


def chunk_page(page_number: int, text: str) -> list[Chunk]:
    """Split a page into <=_CHUNK_CHARS chunks on line boundaries."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    for ln in lines:
        if size + len(ln) > _CHUNK_CHARS and buf:
            chunks.append(Chunk(page_number, "\n".join(buf)))
            buf = buf[-_CHUNK_OVERLAP_LINES:] if _CHUNK_OVERLAP_LINES else []
            size = sum(len(x) for x in buf)
        buf.append(ln)
        size += len(ln)
    if buf:
        chunks.append(Chunk(page_number, "\n".join(buf)))
    return chunks


class RagIndex:
    def __init__(
        self,
        cfg: RAGConfig = default_rag,
        store_cfg: StoreConfig = default_store,
    ) -> None:
        self.cfg = cfg
        self.store_cfg = store_cfg
        self._embedder = None
        # in-memory state for the currently loaded document
        self._doc_id: int | None = None
        self._index = None
        self._chunks: list[Chunk] = []

    # -- embedder ----------------------------------------------------------- #
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

    # -- paths -------------------------------------------------------------- #
    def _paths(self, doc_id: int) -> tuple[Path, Path]:
        base = self.store_cfg.index_dir
        return base / f"doc_{doc_id}.faiss", base / f"doc_{doc_id}.json"

    # -- build / persist ---------------------------------------------------- #
    def build(self, doc_id: int, pages: Iterable[tuple[int, str]]) -> int:
        """Build (or rebuild) the index for a document from (page_no, text) pairs."""
        import faiss

        chunks: list[Chunk] = []
        for page_number, text in pages:
            chunks.extend(chunk_page(page_number, text))
        if not chunks:
            self._doc_id, self._index, self._chunks = doc_id, None, []
            return 0

        vectors = self._embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])  # cosine via normalized vectors
        index.add(vectors)

        self._doc_id, self._index, self._chunks = doc_id, index, chunks
        self._persist(doc_id)
        return len(chunks)

    def build_from_store(self, doc_id: int, store: "DocumentStore") -> int:
        # RAG_USE_CORRECTED (default True) -> index post-corrected text, else raw TrOCR.
        return self.build(
            doc_id,
            store.iter_pages_text(doc_id, prefer_corrected=self.cfg.use_corrected_text),
        )

    def add_pages(self, doc_id: int, store: "DocumentStore", page_numbers: list[int]) -> int:
        """Incrementally embed only `page_numbers` and append to the persisted index.

        Rebuilding the whole index on every upload re-embeds the entire corpus (slow:
        ~1s per 30 chunks). Appending just the new pages keeps ingestion fast regardless
        of corpus size. Falls back to a full build if no index exists yet.
        """
        if not page_numbers or not self._ensure_loaded(doc_id) or self._index is None:
            return self.build_from_store(doc_id, store)
        wanted = set(page_numbers)
        new_chunks: list[Chunk] = []
        for page_number, text in store.iter_pages_text(
            doc_id, prefer_corrected=self.cfg.use_corrected_text
        ):
            if page_number in wanted:
                new_chunks.extend(chunk_page(page_number, text))
        if not new_chunks:
            return len(self._chunks)
        vectors = self._embed([c.text for c in new_chunks])
        self._index.add(vectors)
        self._chunks.extend(new_chunks)
        self._persist(doc_id)
        return len(self._chunks)

    def _persist(self, doc_id: int) -> None:
        import faiss

        idx_path, meta_path = self._paths(doc_id)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(idx_path))
        meta_path.write_text(
            json.dumps([asdict(c) for c in self._chunks]), encoding="utf-8"
        )

    # -- load / query ------------------------------------------------------- #
    def _ensure_loaded(self, doc_id: int) -> bool:
        if self._doc_id == doc_id and self._index is not None:
            return True
        import faiss

        idx_path, meta_path = self._paths(doc_id)
        if not idx_path.exists() or not meta_path.exists():
            return False
        self._index = faiss.read_index(str(idx_path))
        self._chunks = [Chunk(**c) for c in json.loads(meta_path.read_text("utf-8"))]
        self._doc_id = doc_id
        return True

    def query(self, doc_id: int, question: str, top_k: int | None = None) -> list[Retrieved]:
        k = top_k or self.cfg.top_k
        if not self._ensure_loaded(doc_id) or self._index is None:
            return []
        qv = self._embed([question])
        k = min(k, len(self._chunks))
        scores, idxs = self._index.search(qv, k)
        out: list[Retrieved] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            c = self._chunks[int(i)]
            out.append(Retrieved(page_number=c.page_number, text=c.text, score=float(score)))
        return out

    def exists(self, doc_id: int) -> bool:
        idx_path, meta_path = self._paths(doc_id)
        return idx_path.exists() and meta_path.exists()
