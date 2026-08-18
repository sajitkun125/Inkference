"""Accounts and sessions, on PostgreSQL via SQLAlchemy.

Two things this file is careful about:

* **Only hashes are stored.** Passwords go through scrypt with a per-user salt, and
  session cookies are stored as a SHA-256 digest — a dump of these tables yields
  neither a usable password nor a usable session cookie.
* **Separate database from the corpus.** deploy_all_books.sh publishes inkference.db
  to a PUBLIC HF dataset. Accounts have never lived there and must not start.

Password hashing stays stdlib `hashlib.scrypt`: it is memory-hard, it is in the
standard library, and the cost parameters are stored inside each hash string, so
they can be raised later without invalidating existing accounts.
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import logging
import secrets
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession

from ..config import AuthConfig, DatabaseConfig
from ..config import auth as default_auth
from ..config import database as default_database
from .db import session_scope
from .models import OAuthIdentity, Session, User

logger = logging.getLogger("inkference.auth")

PASSWORD_METHOD = "password"


class EmailTaken(Exception):
    """Signup with an address that already has an account."""


class AccountDisabled(Exception):
    """Credentials were right, but an operator has switched the account off."""


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


class AuthStore:
    """All account reads and writes. Holds no connection of its own — every method
    opens a scoped transaction, so instances are safe to share across threads."""

    def __init__(
        self,
        cfg: AuthConfig = default_auth,
        db_cfg: DatabaseConfig = default_database,
    ) -> None:
        self.cfg = cfg
        self.db_cfg = db_cfg

    def _scope(self):
        return session_scope(self.db_cfg)

    # -- passwords ---------------------------------------------------------- #
    def hash_password(self, password: str, *, salt: bytes | None = None) -> str:
        """scrypt hash, self-describing so cost parameters can change later without
        invalidating existing accounts (the stored string carries the n/r/p it used)."""
        salt = salt or secrets.token_bytes(16)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=self.cfg.scrypt_n,
            r=self.cfg.scrypt_r,
            p=self.cfg.scrypt_p,
            dklen=32,
            maxmem=self.cfg.scrypt_n * self.cfg.scrypt_r * 256,
        )
        b64 = lambda raw: base64.b64encode(raw).decode()  # noqa: E731
        return (
            f"scrypt${self.cfg.scrypt_n}${self.cfg.scrypt_r}${self.cfg.scrypt_p}"
            f"${b64(salt)}${b64(derived)}"
        )

    def verify_password(self, password: str, stored: str | None) -> bool:
        if not stored:
            # A provider-only account: no password exists, so no password matches.
            return False
        try:
            scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
            if scheme != "scrypt":
                return False
            n, r, p = int(n), int(r), int(p)
            salt, expected = base64.b64decode(salt_b64), base64.b64decode(hash_b64)
        except (ValueError, TypeError):
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
            dklen=len(expected), maxmem=n * r * 256,
        )
        return hmac.compare_digest(derived, expected)

    # -- users -------------------------------------------------------------- #
    def create_user(self, email: str, password: str, name: str | None = None) -> dict[str, Any]:
        email = normalize_email(email)
        record = self.hash_password(password)
        try:
            with self._scope() as db:
                user = User(
                    email=email,
                    name=(name or "").strip() or None,
                    password_hash=record,
                )
                db.add(user)
                db.flush()          # assigns the id inside the transaction
                return user.to_public()
        except IntegrityError as exc:
            # Relies on the unique index rather than a SELECT-then-INSERT, which would
            # race two concurrent signups for the same address into two accounts.
            raise EmailTaken(email) from exc

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._scope() as db:
            user = db.get(User, user_id)
            return user.to_public() if user else None

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._scope() as db:
            user = self._find_by_email(db, normalize_email(email))
            return user.to_public() if user else None

    @staticmethod
    def _find_by_email(db: OrmSession, email: str) -> User | None:
        return db.scalar(select(User).where(User.email == email))

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        """Return the user on a correct password, else None.

        Raises AccountDisabled when the password was right but the account is off —
        the caller turns that into a distinct message, since telling someone their
        password works but their account is suspended reveals nothing an attacker
        with the right password doesn't already have.
        """
        with self._scope() as db:
            user = self._find_by_email(db, normalize_email(email))
            if user is None:
                # Hash anyway so a missing account takes as long as a wrong password —
                # otherwise response time reveals which addresses are registered.
                self.hash_password(password)
                return None
            if not self.verify_password(password, user.password_hash):
                return None
            if not user.is_active:
                raise AccountDisabled(user.email)
            user.last_login_at = _now()
            return user.to_public()

    def set_password(self, user_id: int, password: str) -> bool:
        """Set or replace a password — also the way a provider-only account gains one."""
        record = self.hash_password(password)
        with self._scope() as db:
            result = db.execute(
                update(User).where(User.id == user_id).values(password_hash=record)
            )
            return result.rowcount > 0

    # -- federated identity (OpenID Connect) -------------------------------- #
    def upsert_oauth_user(
        self,
        provider: str,
        subject: str,
        email: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a provider identity to a local account, creating or linking as needed.

        `subject` is the provider's stable, immutable user id and the real key. An
        address can be renamed, or in a workplace tenant reassigned to a different
        person; `subject` cannot. Keying on it is what keeps a user who renames their
        mailbox attached to the same library.

        THE CALLER MUST HAVE VERIFIED THE EMAIL. The linking branch below adopts an
        existing password account with a matching address, so accepting an unverified
        claim would be an account-takeover primitive: anyone able to make a provider
        emit `ada@example.com` would inherit Ada's library.
        """
        if not provider or not subject:
            raise ValueError("provider and subject are required")
        email = normalize_email(email)
        name = (name or "").strip() or None

        with self._scope() as db:
            identity = db.scalar(
                select(OAuthIdentity).where(
                    OAuthIdentity.provider == provider,
                    OAuthIdentity.subject == subject,
                )
            )

            if identity is not None:
                user = identity.user
                if not user.is_active:
                    raise AccountDisabled(user.email)
                identity.last_login_at = _now()
                identity.email = email or identity.email
                # The provider is authoritative for both fields once linked, but only
                # move the address if it is actually free — a rename onto an address
                # another account already holds must not fail the sign-in.
                if email and user.email != email and self._find_by_email(db, email) is None:
                    user.email = email
                if name and not user.name:
                    user.name = name
                user.last_login_at = _now()
                return user.to_public()

            # First sign-in for this identity. Adopt the password account with the same
            # (verified) address if there is one, so a user who signed up with a
            # password and later clicks the provider button lands in the same library
            # rather than a silent duplicate.
            user = self._find_by_email(db, email) if email else None
            if user is None:
                user = User(email=email, name=name, password_hash=None)
                db.add(user)
                db.flush()
                logger.info("account created via %s id=%s", provider, user.id)
            elif not user.is_active:
                raise AccountDisabled(user.email)
            else:
                logger.info("linked %s identity to existing account id=%s", provider, user.id)
                if name and not user.name:
                    user.name = name

            db.add(
                OAuthIdentity(
                    user_id=user.id,
                    provider=provider,
                    subject=subject,
                    email=email or None,
                    last_login_at=_now(),
                )
            )
            user.last_login_at = _now()
            return user.to_public()

    def list_identities(self, user_id: int) -> list[str]:
        """Which providers this account can sign in with. Drives "you signed up with
        Google — use that button" on a failed password attempt."""
        with self._scope() as db:
            return list(
                db.scalars(
                    select(OAuthIdentity.provider).where(OAuthIdentity.user_id == user_id)
                )
            )

    # -- sessions ----------------------------------------------------------- #
    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, user_id: int, auth_method: str = PASSWORD_METHOD) -> str:
        """Mint a session and return the RAW token — the only moment it exists in the
        clear. The caller puts it in a cookie; the database keeps just its digest."""
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._scope() as db:
            db.add(
                Session(
                    token_hash=self._token_hash(token),
                    user_id=user_id,
                    created_at=now,
                    expires_at=now + _dt.timedelta(days=self.cfg.session_ttl_days),
                    auth_method=auth_method,
                )
            )
        return token

    def get_session_user(self, token: str | None) -> dict[str, Any] | None:
        """Resolve a raw cookie value to its user, or None if unknown/expired/disabled."""
        if not token:
            return None
        with self._scope() as db:
            row = db.execute(
                select(Session, User)
                .join(User, User.id == Session.user_id)
                .where(Session.token_hash == self._token_hash(token))
            ).first()
            if row is None:
                return None
            session, user = row
            if session.expires_at <= _now():
                db.delete(session)
                return None
            if not user.is_active:
                # Disabling an account cuts its live sessions on the next request,
                # rather than leaving them valid until they expire.
                db.delete(session)
                return None
            return user.to_public()

    def delete_session(self, token: str | None) -> bool:
        if not token:
            return False
        with self._scope() as db:
            result = db.execute(
                delete(Session).where(Session.token_hash == self._token_hash(token))
            )
            return result.rowcount > 0

    def delete_user_sessions(self, user_id: int) -> int:
        """Sign a user out everywhere. The right response to a password change."""
        with self._scope() as db:
            return db.execute(delete(Session).where(Session.user_id == user_id)).rowcount

    def purge_expired(self) -> int:
        with self._scope() as db:
            return db.execute(delete(Session).where(Session.expires_at <= _now())).rowcount
