"""Account tables: users, their federated identities, and their sessions.

Three tables rather than two because a user can hold more than one way in. Someone
who signed up with a password and later clicks "Continue with Google" is one person,
one library, one row in `users` — with a row in `oauth_identities` per provider. A
`google_sub` column on `users` would have forced a second column for Microsoft, and
a third for whatever comes next.

Nothing here stores a credential in a usable form: `password_hash` is scrypt output,
and sessions are keyed by the SHA-256 of the cookie, never the cookie itself.
"""
from __future__ import annotations

import datetime as _dt

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # Always stored lowercase (see AuthStore.normalize_email). Case folding happens in
    # Python rather than via CITEXT so the schema needs no Postgres extension — one
    # less thing for a managed server to have to permit.
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(String(200))
    # NULL for an account that has only ever signed in through a provider. Nullable is
    # the honest representation: the old sentinel-string trick made "no password" and
    # "unusable password" indistinguishable in the data.
    password_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    last_login_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Soft lock, for an operator disabling an account without deleting the library
    # behind it. Checked on every session resolution, so it takes effect immediately.
    disabled_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True))

    identities: Mapped[list["OAuthIdentity"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def is_active(self) -> bool:
        return self.disabled_at is None

    def to_public(self) -> dict:
        """The only shape that ever reaches the API. Explicit allow-list, so adding a
        column here can never leak it through /api/auth/me by accident."""
        return {"id": self.id, "email": self.email, "name": self.name}


class OAuthIdentity(Base):
    """One (provider, subject) pair the user can sign in with."""

    __tablename__ = "oauth_identities"
    __table_args__ = (
        # The real primary key of a federated identity. Google's `sub` is unique only
        # within Google, so the provider has to be part of the constraint.
        UniqueConstraint("provider", "subject", name="uq_oauth_identity_provider_subject"),
        Index("ix_oauth_identities_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    # The provider's immutable user id (`sub`). Deliberately NOT the email: an address
    # can be renamed or, in a workplace tenant, reassigned to a different human. `sub`
    # is what keeps a renamed account attached to the same library.
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    # Snapshot of the address as the provider last reported it — for support and audit,
    # never for lookup. users.email is the authoritative one.
    email: Mapped[str | None] = mapped_column(String(320))
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_login_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="identities")


class Session(Base):
    """A signed-in browser. The row is the session; deleting it is the logout."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_user_id", "user_id"),
        # purge_expired sweeps on this column, and it keeps that delete off a seq scan
        # once the table has any history in it.
        Index("ix_sessions_expires_at", "expires_at"),
    )

    # SHA-256 hex of the cookie value. The cookie itself is never written down, so a
    # dump of this table yields nothing that can be replayed as a session.
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Which provider minted this session ("password", "google", "microsoft"). Useful
    # when a provider has to be revoked and every session it issued cut with it.
    auth_method: Mapped[str] = mapped_column(String(40), nullable=False, server_default="password")

    user: Mapped[User] = relationship(back_populates="sessions")
