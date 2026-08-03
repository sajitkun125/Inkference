"""Shared fixtures. Everything here is offline — no model loads, no provider calls."""
from __future__ import annotations

import pytest

from inkference.config import AgentConfig, StoreConfig
from inkference.store import DocumentStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A tiny two-book DocumentStore, shaped like the seeded corpus.

    Pages 1-3 are "book1", 4-5 are "book2", and page 6 has an ABSOLUTE image_path —
    standing in for a user-uploaded scan, which has no book and must not crash the
    book-map derivation.
    """
    cfg = StoreConfig(
        db_path=tmp_path / "t.db",
        assets_dir=tmp_path / "assets",
        index_dir=tmp_path / "index",
    )
    st = DocumentStore(cfg)
    doc_id = st.create_document(title="Test Journal", slug="test", subtitle="fixture")

    keys = {
        1: "book1/forster1/B1_P_001.jpg",
        2: "book1/forster1/B1_P_002.jpg",
        3: "book1/forster1/B1_P_003.jpg",
        4: "book2/forster2/B2_P_001.jpg",
        5: "book2/forster2/B2_P_002.jpg",
        6: str(tmp_path / "assets" / "uploaded.png"),
    }
    for page_number, key in keys.items():
        page_id = st.add_page(doc_id, page_number, image_path=key)
        # Page 5 deliberately has no text, to exercise the empty-page branches.
        text = "" if page_number == 5 else f"Text of page {page_number}. Plymouth harbour."
        _write_page_text(st, page_id, text)
    st.doc_id = doc_id
    return st


def _write_page_text(st: DocumentStore, page_id: int, text: str) -> None:
    """Insert one line of text directly — cheaper and more explicit than driving the
    whole HTR pipeline just to populate a fixture."""
    with st._connect() as conn:
        conn.execute(
            "UPDATE pages SET status='complete', corrected_text=? WHERE id=?",
            (text or None, page_id),
        )
        if text:
            conn.execute(
                "INSERT INTO lines (page_id, idx, x0, y0, x1, y1, text, confidence, "
                "needs_review) VALUES (?,0,0,0,10,10,?,0.9,0)",
                (page_id, text),
            )


@pytest.fixture
def agent_cfg(tmp_path):
    return AgentConfig(
        max_steps=3,
        max_span=2,          # small, so clamping is easy to assert
        page_chars=200,
        evidence_chars=400,
        score_floor=0.25,
        checkpoint_path=tmp_path / "ckpt.db",
    )
