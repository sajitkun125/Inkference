"""Engine, pool, and session plumbing for the accounts database.

The engine is a process-wide singleton built on first use, not at import: the
seeders and HTR scripts import `inkference.config` without ever touching accounts,
and they must not open a Postgres connection (or fail because none is reachable)
just by being imported.

Sync SQLAlchemy on purpose. Every FastAPI auth route is a plain `def`, which
Starlette already runs in a worker thread, so a sync engine adds no blocking the
event loop would have noticed — and it keeps the whole auth path free of the
async/sync colouring that async SQLAlchemy would spread through it.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import event, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession, sessionmaker

from ..config import DatabaseConfig
from ..config import database as default_database

logger = logging.getLogger("inkference.auth.db")

_engine: Engine | None = None
_session_factory: sessionmaker[OrmSession] | None = None
_lock = Lock()


def get_engine(cfg: DatabaseConfig = default_database) -> Engine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is None:                      # re-check: another thread may have won
            _engine = _build_engine(cfg)
            _session_factory = sessionmaker(
                bind=_engine, expire_on_commit=False, future=True
            )
    return _engine


def _build_engine(cfg: DatabaseConfig) -> Engine:
    url = make_url(cfg.normalized_url)
    if not url.drivername.startswith("postgresql"):
        # Fail loudly at startup rather than three layers down inside a query. This
        # app has one supported database; a sqlite:// URL in an env file is a mistake
        # worth stopping for, not something to quietly limp along on.
        raise RuntimeError(
            f"DATABASE_URL must be a PostgreSQL URL, got driver {url.drivername!r}. "
            "See app/deploy/docker-compose.dev.yml for a local server."
        )
    logger.info("accounts database: %s (pool=%d+%d)", cfg.safe_url, cfg.pool_size,
                cfg.max_overflow)
    engine = create_engine_with_pool(cfg, url)
    _install_statement_timeout(engine, cfg)
    return engine


def create_engine_with_pool(cfg: DatabaseConfig, url) -> Engine:
    from sqlalchemy import create_engine

    return create_engine(
        url,
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        pool_recycle=cfg.pool_recycle_seconds,
        pool_pre_ping=cfg.pool_pre_ping,
        echo=cfg.echo,
        future=True,
        # Named %(name)s placeholders and one round trip per execute. Auth queries are
        # single-row lookups, so there is nothing here for a prepared-statement cache
        # to amortise, and disabling it avoids the stale-plan class of failure when a
        # migration changes a table under a long-lived pooled connection.
        connect_args={"prepare_threshold": None},
    )


def _install_statement_timeout(engine: Engine, cfg: DatabaseConfig) -> None:
    """Cap every statement on every pooled connection.

    Set per-connection rather than per-session: a connection is reused across
    hundreds of requests, so paying for this once at checkout is far cheaper than
    a SET on each. Auth queries are indexed point lookups — anything still running
    after ten seconds is a lock or a fault, and failing fast beats piling up
    workers behind it.
    """
    if cfg.statement_timeout_ms <= 0:
        return

    @event.listens_for(engine, "connect")
    def _set_timeout(dbapi_conn, _record) -> None:  # pragma: no cover - driver callback
        with dbapi_conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(cfg.statement_timeout_ms)}")


def get_session_factory(cfg: DatabaseConfig = default_database) -> sessionmaker[OrmSession]:
    get_engine(cfg)
    assert _session_factory is not None  # get_engine builds it under the same lock
    return _session_factory


@contextmanager
def session_scope(cfg: DatabaseConfig = default_database) -> Iterator[OrmSession]:
    """Transactional scope: commit on clean exit, roll back on any exception.

    Every write in AuthStore goes through this, so a half-applied sign-in (identity
    row written, user row not) cannot be left behind by an error mid-way.
    """
    session = get_session_factory(cfg)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(cfg: DatabaseConfig = default_database) -> bool:
    """One cheap round trip, for the readiness probe. Never raises: a probe that
    throws is indistinguishable from a probe that hangs."""
    try:
        with get_engine(cfg).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except (SQLAlchemyError, RuntimeError, OSError) as exc:
        logger.warning("accounts database unreachable: %s", exc)
        return False


def dispose_engine() -> None:
    """Drop the pool and forget the singleton. For tests, which build a fresh engine
    per database, and for a clean shutdown."""
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None
