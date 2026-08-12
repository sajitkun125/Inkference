"""Accounts and sessions, backed by their own SQLite file.

Stdlib only (`sqlite3`, `hashlib`, `secrets`) so the Space image gains no dependency.

Two deliberate choices:

* **Separate database.** `AuthConfig.db_path` is NOT inkference.db, because
  deploy_all_books.sh uploads that file to a public HF dataset. Password hashes and
  session tokens must never ship with the corpus.
* **Only hashes are stored.** Passwords go through scrypt with a per-user salt, and
  session cookies are stored as a SHA-256 digest — a leak of this file yields neither
  a usable password nor a usable session cookie.
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import secrets
import sqlite3
from typing import Any

from ..config import AuthConfig
from ..config import auth as default_auth

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    name          TEXT,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


class EmailTaken(Exception):
    """Signup with an address that already has an account."""


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(dt: _dt.datetime) -> str:
    return dt.isoformat(timespec="seconds")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


class AuthStore:
    def __init__(self, cfg: AuthConfig = default_auth) -> None:
        self.cfg = cfg
        self.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cfg.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

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

    def verify_password(self, password: str, stored: str) -> bool:
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
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users(email,name,password_hash,created_at) VALUES(?,?,?,?)",
                    (email, (name or "").strip() or None, record, _iso(_now())),
                )
                user_id = int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise EmailTaken(email) from exc
        return {"id": user_id, "email": email, "name": (name or "").strip() or None}

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id,email,name,created_at FROM users WHERE id=?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        """Return the user on a correct password, else None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id,email,name,password_hash FROM users WHERE email=?",
                (normalize_email(email),),
            ).fetchone()
        if row is None:
            # Hash anyway so a missing account takes as long as a wrong password —
            # otherwise response time reveals which addresses are registered.
            self.hash_password(password)
            return None
        if not self.verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "email": row["email"], "name": row["name"]}

    # -- sessions ----------------------------------------------------------- #
    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, user_id: int) -> str:
        """Mint a session and return the RAW token — the only time it exists in the
        clear. The caller puts it in a cookie; the DB keeps just its digest."""
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(token_hash,user_id,created_at,expires_at) VALUES(?,?,?,?)",
                (
                    self._token_hash(token),
                    user_id,
                    _iso(now),
                    _iso(now + _dt.timedelta(days=self.cfg.session_ttl_days)),
                ),
            )
        return token

    def get_session_user(self, token: str | None) -> dict[str, Any] | None:
        """Resolve a raw cookie value to its user, or None if unknown/expired."""
        if not token:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT u.id, u.email, u.name, s.expires_at FROM sessions s "
                "JOIN users u ON u.id = s.user_id WHERE s.token_hash=?",
                (self._token_hash(token),),
            ).fetchone()
        if row is None:
            return None
        if _dt.datetime.fromisoformat(row["expires_at"]) <= _now():
            self.delete_session(token)
            return None
        return {"id": row["id"], "email": row["email"], "name": row["name"]}

    def delete_session(self, token: str | None) -> bool:
        if not token:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE token_hash=?", (self._token_hash(token),)
            )
        return cur.rowcount > 0

    def purge_expired(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (_iso(_now()),))
        return cur.rowcount
