"""Accounts, sessions, and federated sign-in.

Backed by PostgreSQL (DatabaseConfig), deliberately never by inkference.db — that
one is uploaded to a public HF dataset at deploy time.

Submodules are imported lazily by the API layer; importing this package pulls in
SQLAlchemy but opens no connection, so the seeders and HTR scripts stay unaffected.
"""
from .models import Base, OAuthIdentity, Session, User
from .store import AccountDisabled, AuthStore, EmailTaken, normalize_email

__all__ = [
    "AccountDisabled",
    "AuthStore",
    "Base",
    "EmailTaken",
    "OAuthIdentity",
    "Session",
    "User",
    "normalize_email",
]
