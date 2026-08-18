"""accounts, federated identities, and sessions

Initial schema for the PostgreSQL accounts database.

This replaces the single-file sqlite3 auth.db the app used while it was a
single-process demo. There is deliberately no data migration from that file: it
was per-developer local state that never held production accounts, and writing a
cross-engine backfill for it would be more code than re-registering.

Revision ID: 0001_accounts
Revises:
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_accounts"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        # 320 = the RFC 5321 maximum (64 local + @ + 255 domain). Always stored
        # lowercase by AuthStore.normalize_email, so a plain unique index gives
        # case-insensitive uniqueness without the citext extension.
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        # Nullable: an account that has only ever signed in through Google or Entra
        # has no password, and NULL says that far more honestly than a sentinel
        # string that some future code path might try to verify against.
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "oauth_identities",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_oauth_identities"),
        # Deleting a user must take their identities with them, in one statement,
        # inside the database — an application-side cascade would leave orphans
        # behind whenever a delete happened any other way.
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name="fk_oauth_identities_user_id_users", ondelete="CASCADE",
        ),
        # `subject` is unique only within its provider, so the provider has to be
        # part of the key. This is the constraint that makes the identity lookup a
        # single index probe and makes double-linking impossible under a race.
        sa.UniqueConstraint(
            "provider", "subject", name="uq_oauth_identity_provider_subject"
        ),
    )
    op.create_index("ix_oauth_identities_user_id", "oauth_identities", ["user_id"])

    op.create_table(
        "sessions",
        # SHA-256 hex of the cookie. The cookie itself is never stored, so a dump of
        # this table yields nothing replayable.
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "auth_method", sa.String(length=40),
            server_default=sa.text("'password'"), nullable=False,
        ),
        sa.PrimaryKeyConstraint("token_hash", name="pk_sessions"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name="fk_sessions_user_id_users", ondelete="CASCADE",
        ),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    # purge_expired sweeps on this column; without the index that DELETE degrades to
    # a sequential scan once the table carries any history.
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_sessions_expires_at", table_name="sessions")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
    op.drop_index("ix_oauth_identities_user_id", table_name="oauth_identities")
    op.drop_table("oauth_identities")
    op.drop_table("users")
