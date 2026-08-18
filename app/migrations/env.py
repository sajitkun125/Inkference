"""Alembic environment.

The connection URL comes from `inkference.config.database`, never from alembic.ini:
the deployed URL is an Azure secret injected as DATABASE_URL, and a second copy in
a checked-in ini file is exactly how the two drift apart.
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# `alembic` is run from app/, where src/ is not yet importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inkference.auth.models import Base  # noqa: E402
from inkference.config import database as db_cfg  # noqa: E402

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False is not optional here. The app migrates itself
    # during startup, and fileConfig's default silences every logger already
    # configured — which at that moment is uvicorn's, plus our own. The symptom is a
    # server that runs but stops logging anything, including tracebacks.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

config.set_main_option("sqlalchemy.url", db_cfg.normalized_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it — `alembic upgrade head --sql`.

    This is the path to use when a DBA has to review the change, or when the
    application's own database role is not allowed to run DDL in production.
    """
    context.configure(
        url=db_cfg.normalized_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)
    if connectable is not None:
        # Reuse a connection the caller already opened (the startup migration hook
        # and the test fixtures both do this) rather than building a second pool.
        _run(connectable)
        return

    engine = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with engine.connect() as connection:
        _run(connection)


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Catch a column whose type drifted from the model, not just added/dropped ones.
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
