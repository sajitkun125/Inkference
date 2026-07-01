"""SQLite-backed store for documents, pages, lines, words, and ingest jobs.

Uses only the stdlib `sqlite3` (no extra dependency). WAL mode + a fresh
connection per operation keeps it safe to use from FastAPI background threads.
Page scans and line crops live on the filesystem under StoreConfig.assets_dir;
the DB holds paths and structured transcription data.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from ..config import StoreConfig
from ..config import store as default_store
from ..schemas import JobStatus, Line, PageResult, Word

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    slug        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    subtitle    TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number    INTEGER NOT NULL,
    image_path     TEXT,
    width          INTEGER,
    height         INTEGER,
    scale          REAL DEFAULT 1.0,
    status         TEXT NOT NULL DEFAULT 'queued',
    avg_confidence REAL,
    low_conf_words INTEGER,
    UNIQUE(document_id, page_number)
);
CREATE TABLE IF NOT EXISTS lines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id      INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
    text         TEXT,
    confidence   REAL,
    needs_review INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS words (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id      INTEGER NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    text         TEXT,
    confidence   REAL,
    needs_review INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'queued',
    stage       TEXT,
    progress    REAL DEFAULT 0.0,
    message     TEXT,
    total_pages INTEGER DEFAULT 0,
    done_pages  INTEGER DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_lines_page ON lines(page_id);
CREATE INDEX IF NOT EXISTS idx_words_line ON words(line_id);
"""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class DocumentStore:
    def __init__(self, cfg: StoreConfig = default_store) -> None:
        self.cfg = cfg
        self.cfg.assets_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- connection --------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cfg.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- documents ---------------------------------------------------------- #
    def create_document(self, title: str, slug: str, subtitle: str | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO documents(slug,title,subtitle,created_at) VALUES(?,?,?,?)",
                (slug, title, subtitle, _now()),
            )
            return int(cur.lastrowid)

    def get_document_by_slug(self, slug: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE slug=?", (slug,)).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT d.*,
                          COUNT(p.id) AS page_count,
                          SUM(CASE WHEN p.status='complete' THEN 1 ELSE 0 END) AS pages_done,
                          AVG(p.avg_confidence) AS avg_confidence
                   FROM documents d LEFT JOIN pages p ON p.document_id=d.id
                   GROUP BY d.id ORDER BY d.created_at DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return None
            doc = dict(row)
            pages = conn.execute(
                """SELECT page_number,status,avg_confidence,low_conf_words,width,height
                   FROM pages WHERE document_id=? ORDER BY page_number""",
                (doc_id,),
            ).fetchall()
            doc["pages"] = [dict(p) for p in pages]
            doc["page_count"] = len(doc["pages"])
            return doc

    # -- pages -------------------------------------------------------------- #
    def add_page(
        self, doc_id: int, page_number: int, image_path: str | Path | None = None
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO pages(document_id,page_number,image_path,status)
                   VALUES(?,?,?, 'queued')
                   ON CONFLICT(document_id,page_number)
                   DO UPDATE SET image_path=excluded.image_path, status='queued'""",
                (doc_id, page_number, str(image_path) if image_path else None),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM pages WHERE document_id=? AND page_number=?",
                (doc_id, page_number),
            ).fetchone()
            return int(row["id"])

    def set_page_status(self, page_id: int, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE pages SET status=? WHERE id=?", (status, page_id))

    def save_page_result(self, page_id: int, result: PageResult) -> None:
        """Persist a completed PageResult: page metadata + lines + words."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE pages SET width=?,height=?,scale=?,status='complete',
                          avg_confidence=?,low_conf_words=? WHERE id=?""",
                (
                    result.image_width,
                    result.image_height,
                    result.scale,
                    result.avg_confidence,
                    result.low_confidence_word_count,
                    page_id,
                ),
            )
            # Replace any prior lines/words for idempotent re-ingest.
            conn.execute("DELETE FROM lines WHERE page_id=?", (page_id,))
            for line in result.lines:
                x0, y0, x1, y1 = line.bbox
                cur = conn.execute(
                    """INSERT INTO lines(page_id,idx,x0,y0,x1,y1,text,confidence,needs_review)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (page_id, line.index, x0, y0, x1, y1, line.text,
                     line.confidence, int(line.needs_review)),
                )
                line_id = cur.lastrowid
                conn.executemany(
                    """INSERT INTO words(line_id,idx,text,confidence,needs_review)
                       VALUES(?,?,?,?,?)""",
                    [(line_id, i, w.text, w.confidence, int(w.needs_review))
                     for i, w in enumerate(line.words)],
                )

    def get_page(self, doc_id: int, page_number: int) -> dict[str, Any] | None:
        """Full page payload for the Reader: metadata + lines (with words)."""
        with self._connect() as conn:
            page = conn.execute(
                "SELECT * FROM pages WHERE document_id=? AND page_number=?",
                (doc_id, page_number),
            ).fetchone()
            if not page:
                return None
            page = dict(page)
            lines = conn.execute(
                "SELECT * FROM lines WHERE page_id=? ORDER BY idx", (page["id"],)
            ).fetchall()
            out_lines = []
            for ln in lines:
                ln = dict(ln)
                words = conn.execute(
                    "SELECT text,confidence,needs_review FROM words WHERE line_id=? ORDER BY idx",
                    (ln["id"],),
                ).fetchall()
                ln["words"] = [dict(w) for w in words]
                ln["bbox"] = [ln.pop("x0"), ln.pop("y0"), ln.pop("x1"), ln.pop("y1")]
                out_lines.append(ln)
            page["lines"] = out_lines
            return page

    def get_page_image_path(self, doc_id: int, page_number: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_path FROM pages WHERE document_id=? AND page_number=?",
                (doc_id, page_number),
            ).fetchone()
            return row["image_path"] if row and row["image_path"] else None

    def iter_pages_text(self, doc_id: int) -> Iterator[tuple[int, str]]:
        """(page_number, full_text) for completed pages — feeds RAG indexing."""
        with self._connect() as conn:
            pages = conn.execute(
                "SELECT id,page_number FROM pages WHERE document_id=? AND status='complete' ORDER BY page_number",
                (doc_id,),
            ).fetchall()
            for p in pages:
                rows = conn.execute(
                    "SELECT text FROM lines WHERE page_id=? ORDER BY idx", (p["id"],)
                ).fetchall()
                text = "\n".join(r["text"] for r in rows if r["text"])
                yield int(p["page_number"]), text

    # -- jobs --------------------------------------------------------------- #
    def create_job(self, doc_id: int, total_pages: int) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO jobs(document_id,status,total_pages,created_at,updated_at)
                   VALUES(?, 'queued', ?, ?, ?)""",
                (doc_id, total_pages, _now(), _now()),
            )
            return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        # JobStatus/Stage enums -> their string value
        cleaned = {k: (v.value if hasattr(v, "value") else v) for k, v in fields.items()}
        cols = ", ".join(f"{k}=?" for k in cleaned)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*cleaned.values(), job_id))

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None
