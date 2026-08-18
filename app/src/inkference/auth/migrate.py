"""Run Alembic migrations from inside the application.

Container Apps has no natural "run this once before the new revision goes live"
step, so the app migrates itself at startup. That is safe here because of the
advisory lock below; for a larger schema it should still move to a release job
(DB_MIGRATE_ON_STARTUP=false), since a long migration would hold the whole rollout
open behind a lock.
"""
from __future__ import annotations

import logging

from sqlalchemy import text

from ..config import APP_ROOT, DatabaseConfig
from ..config import database as default_database
from .db import get_engine

logger = logging.getLogger("inkference.auth.migrate")

# PostgreSQL advisory lock keys are plain bigints with no meaning of their own. Any
# constant works, as long as every replica picks the SAME one and nothing else in this
# database uses it.
_MIGRATION_LOCK_KEY = 8_411_512_099


def upgrade_to_head(cfg: DatabaseConfig = default_database) -> None:
    """Bring the accounts database up to the newest revision.

    Wrapped in a transaction-scoped advisory lock so that N replicas starting at
    once do not all run the same DDL. The first through the door migrates; the rest
    block, then find the database already at head and do nothing. Without it, two
    replicas racing `CREATE TABLE users` means one of them crash-loops.
    """
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(str(APP_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(APP_ROOT / "migrations"))

    engine = get_engine(cfg)
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _MIGRATION_LOCK_KEY})
        alembic_cfg.attributes["connection"] = conn
        logger.info("running database migrations (%s)", cfg.safe_url)
        command.upgrade(alembic_cfg, "head")
    logger.info("database is at head")


def current_revision(cfg: DatabaseConfig = default_database) -> str | None:
    """The revision the database is actually on. Surfaced by /api/health so a
    half-deployed revision is visible without shelling into the container."""
    with get_engine(cfg).connect() as conn:
        try:
            row = conn.execute(text("SELECT version_num FROM alembic_version")).first()
        except Exception:
            # Table absent = never migrated. A missing table is an answer, not an error.
            return None
    return row[0] if row else None
