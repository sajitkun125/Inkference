"""Shared fixtures.

Offline in the sense that matters: no model loads and no provider calls. The auth
fixtures do start a local PostgreSQL in Docker, because the accounts store runs on
Postgres in every environment and a substitute engine would test different
behaviour than the one that ships. Tests needing it skip cleanly where Docker is
not available (see `postgres_url`).
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from inkference.config import AgentConfig, AuthConfig, DatabaseConfig, StoreConfig
from inkference.store import DocumentStore

@pytest.fixture(scope="session")
def scrypt_n() -> int:
    """Well below the production 2**14, so the suite is not dominated by KDF cost.

    A fixture rather than a module constant because conftest is not importable —
    pytest loads it as a plugin, not as part of a package.
    """
    return 2**8


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """A throwaway PostgreSQL, one per test session."""
    try:
        # testcontainers 4.13 moved the modules under .community and deprecated the
        # old paths; keep both so the suite runs on either side of that split.
        try:
            from testcontainers.community.postgres import PostgresContainer
        except ImportError:
            from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - depends on the local install
        pytest.skip("testcontainers is not installed (pip install 'testcontainers[postgres]')")

    try:
        # Same major as deploy/docker-compose.dev.yml and Azure Flexible Server.
        with PostgresContainer("postgres:16-alpine", driver="psycopg") as container:
            yield container.get_connection_url()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Docker is not available for the accounts database: {exc}")


@pytest.fixture(scope="session")
def migrated_database(postgres_url):
    """The container, migrated to head. Session-scoped: Alembic runs once, not per test."""
    from inkference.auth.db import dispose_engine, get_engine
    from inkference.auth.migrate import upgrade_to_head

    cfg = DatabaseConfig(url=postgres_url, pool_size=2, max_overflow=2,
                         migrate_on_startup=True)
    dispose_engine()          # the engine is a module singleton; drop any earlier one
    get_engine(cfg)
    upgrade_to_head(cfg)
    yield cfg
    dispose_engine()


@pytest.fixture
def db_cfg(migrated_database):
    """Migrated and empty. TRUNCATE rather than a fresh container per test: it costs
    milliseconds instead of seconds, and RESTART IDENTITY keeps ids predictable."""
    from inkference.auth.db import get_engine

    with get_engine(migrated_database).begin() as conn:
        conn.execute(
            text("TRUNCATE users, oauth_identities, sessions RESTART IDENTITY CASCADE")
        )
    return migrated_database


@pytest.fixture
def auth_store(db_cfg, scrypt_n):
    from inkference.auth import AuthStore

    return AuthStore(AuthConfig(scrypt_n=scrypt_n), db_cfg)


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
