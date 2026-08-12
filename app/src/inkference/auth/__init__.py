"""Accounts and sessions for the sign-in page.

Stored in their own SQLite file (AuthConfig.db_path), never in inkference.db —
that one is uploaded to a public HF dataset at deploy time.
"""
from .store import AuthStore, EmailTaken, normalize_email

__all__ = ["AuthStore", "EmailTaken", "normalize_email"]
