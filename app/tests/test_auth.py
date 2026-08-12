"""Accounts, sessions, and the API gate. Offline — no model loads, no provider calls."""
from __future__ import annotations

import datetime as _dt
import sqlite3

import pytest
from fastapi.testclient import TestClient

from inkference.auth import AuthStore, EmailTaken
from inkference.config import AuthConfig

PASSWORD = "correct-horse-battery"


@pytest.fixture
def auth_store(tmp_path):
    # scrypt_n well below the production 2**14 so the suite isn't dominated by KDF cost.
    return AuthStore(AuthConfig(db_path=tmp_path / "auth.db", scrypt_n=2**8))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with auth pointed at a throwaway DB and the gate switched on."""
    from inkference.api import main, services

    cfg = AuthConfig(db_path=tmp_path / "auth.db", scrypt_n=2**8, required=True)
    monkeypatch.setattr(main, "auth_cfg", cfg)
    monkeypatch.setattr(services, "_auth", AuthStore(cfg))
    return TestClient(main.app)


# -- passwords -------------------------------------------------------------- #
def test_password_round_trip(auth_store):
    record = auth_store.hash_password(PASSWORD)
    assert auth_store.verify_password(PASSWORD, record)
    assert not auth_store.verify_password("wrong", record)


def test_password_is_not_recoverable_from_the_record(auth_store):
    """The stored string must not contain the password, and must be salted so two
    accounts with the same password do not share a hash."""
    record = auth_store.hash_password(PASSWORD)
    assert PASSWORD not in record
    assert record.startswith("scrypt$")
    assert record != auth_store.hash_password(PASSWORD)


def test_verify_rejects_a_malformed_record(auth_store):
    for junk in ("", "nonsense", "scrypt$bad", "md5$1$2$3$4$5"):
        assert not auth_store.verify_password(PASSWORD, junk)


# -- users ------------------------------------------------------------------ #
def test_create_and_authenticate(auth_store):
    user = auth_store.create_user("Ada@Example.COM ", PASSWORD, name="Ada")
    assert user["email"] == "ada@example.com"  # normalized
    assert auth_store.authenticate("ada@example.com", PASSWORD)["id"] == user["id"]
    assert auth_store.authenticate("ADA@example.com", PASSWORD) is not None
    assert auth_store.authenticate("ada@example.com", "wrong") is None
    assert auth_store.authenticate("nobody@example.com", PASSWORD) is None


def test_duplicate_email_rejected(auth_store):
    auth_store.create_user("ada@example.com", PASSWORD)
    with pytest.raises(EmailTaken):
        auth_store.create_user("ADA@example.com", PASSWORD)


# -- sessions --------------------------------------------------------------- #
def test_session_lifecycle(auth_store):
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    assert auth_store.get_session_user(token)["id"] == user["id"]
    assert auth_store.delete_session(token) is True
    assert auth_store.get_session_user(token) is None


def test_unknown_and_empty_tokens_resolve_to_nobody(auth_store):
    assert auth_store.get_session_user("not-a-real-token") is None
    assert auth_store.get_session_user(None) is None
    assert auth_store.get_session_user("") is None


def test_raw_token_is_not_stored(auth_store):
    """A dump of auth.db must not yield a usable cookie."""
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    with sqlite3.connect(auth_store.cfg.db_path) as conn:
        stored = [r[0] for r in conn.execute("SELECT token_hash FROM sessions")]
    assert token not in stored


def test_expired_session_is_refused(auth_store):
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    past = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=1)).isoformat()
    with sqlite3.connect(auth_store.cfg.db_path) as conn:
        conn.execute("UPDATE sessions SET expires_at=?", (past,))
    assert auth_store.get_session_user(token) is None


def test_deleting_a_user_drops_their_sessions(auth_store):
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    with auth_store._connect() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (user["id"],))
    assert auth_store.get_session_user(token) is None


# -- the API gate ----------------------------------------------------------- #
def test_protected_route_requires_a_session(client):
    assert client.get("/api/documents").status_code == 401


def test_health_and_me_stay_public(client):
    assert client.get("/api/health").status_code == 200
    body = client.get("/api/auth/me").json()
    assert body["user"] is None and body["auth_required"] is True


def test_signup_then_reach_a_protected_route(client):
    r = client.post(
        "/api/auth/signup",
        json={"email": "ada@example.com", "password": PASSWORD, "name": "Ada"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["user"]["email"] == "ada@example.com"
    assert "password" not in r.text.lower()
    # The cookie is now on the client, so the gated route opens.
    assert client.get("/api/documents").status_code == 200
    assert client.get("/api/auth/me").json()["user"]["email"] == "ada@example.com"


def test_logout_closes_the_session(client):
    client.post("/api/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    assert client.get("/api/documents").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").json()["user"] is None
    assert client.get("/api/documents").status_code == 401


def test_login_wrong_password_is_refused_and_reveals_nothing(client):
    client.post("/api/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    client.post("/api/auth/logout")
    bad_pw = client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": "wrong-one"}
    )
    missing = client.post(
        "/api/auth/login", json={"email": "ghost@example.com", "password": PASSWORD}
    )
    assert bad_pw.status_code == missing.status_code == 401
    # Identical wording, so the response cannot be used to enumerate accounts.
    assert bad_pw.json()["detail"] == missing.json()["detail"]


def test_login_restores_access(client):
    client.post("/api/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    client.post("/api/auth/logout")
    assert client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    ).status_code == 200
    assert client.get("/api/documents").status_code == 200


def test_signup_validates_email_and_password_length(client):
    assert client.post(
        "/api/auth/signup", json={"email": "not-an-email", "password": PASSWORD}
    ).status_code == 422
    assert client.post(
        "/api/auth/signup", json={"email": "ada@example.com", "password": "short"}
    ).status_code == 422


def test_duplicate_signup_is_a_conflict(client):
    client.post("/api/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    r = client.post("/api/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    assert r.status_code == 409


def test_session_cookie_is_httponly(client):
    r = client.post("/api/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    assert "httponly" in r.headers["set-cookie"].lower()


def test_gate_can_be_switched_off_for_a_public_demo(client, monkeypatch):
    """INKFERENCE_AUTH_REQUIRED=false must open the API without removing accounts."""
    from inkference.api import main

    monkeypatch.setattr(main.auth_cfg, "required", False)
    assert client.get("/api/documents").status_code == 200
    assert client.get("/api/auth/me").json()["auth_required"] is False


# -- deployment safety ------------------------------------------------------ #
def test_credentials_never_live_in_the_published_corpus_db():
    """deploy_all_books.sh uploads inkference.db to a PUBLIC HF dataset. If accounts
    were stored there, every password hash would ship with the corpus."""
    from inkference.config import auth as auth_cfg, store as store_cfg

    assert auth_cfg.db_path != store_cfg.db_path
    assert auth_cfg.db_path.name != store_cfg.db_path.name
