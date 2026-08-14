"""Accounts, sessions, federated identity linking, and the API gate.

Runs against a real PostgreSQL started by the `postgres_url` fixture — the store
depends on behaviour a substitute engine gets subtly wrong (unique violations
surfacing as IntegrityError, ON DELETE CASCADE, timestamptz round-tripping), and
the migrations are the part most worth exercising.
"""
from __future__ import annotations

import datetime as _dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from inkference.auth import AccountDisabled, AuthStore, EmailTaken
from inkference.auth.db import get_engine
from inkference.config import AuthConfig, OIDCConfig, OIDCProvider

PASSWORD = "correct-horse-battery"


@pytest.fixture
def client(db_cfg, scrypt_n, monkeypatch):
    """TestClient with auth pointed at the throwaway database and the gate on."""
    from inkference.api import main, services

    cfg = AuthConfig(scrypt_n=scrypt_n, required=True, secret_key="test-secret")
    monkeypatch.setattr(main, "auth_cfg", cfg)
    monkeypatch.setattr(main, "db_cfg", db_cfg)
    monkeypatch.setattr(services, "_auth", AuthStore(cfg, db_cfg))
    # The fixture already migrated; re-running it inside the lifespan would only
    # re-take the advisory lock for nothing.
    monkeypatch.setattr(services, "init_database", lambda: None)
    with TestClient(main.app) as c:
        yield c


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
    for junk in ("", "nonsense", "scrypt$bad", "md5$1$2$3$4$5", None):
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


def test_disabled_account_cannot_sign_in(auth_store, db_cfg):
    user = auth_store.create_user("ada@example.com", PASSWORD)
    with get_engine(db_cfg).begin() as conn:
        conn.execute(text("UPDATE users SET disabled_at = now() WHERE id = :i"),
                     {"i": user["id"]})
    with pytest.raises(AccountDisabled):
        auth_store.authenticate("ada@example.com", PASSWORD)


def test_disabling_an_account_kills_its_live_sessions(auth_store, db_cfg):
    """Immediately, not when the cookie expires — otherwise disabling an account
    leaves it usable for up to the session TTL."""
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    assert auth_store.get_session_user(token) is not None
    with get_engine(db_cfg).begin() as conn:
        conn.execute(text("UPDATE users SET disabled_at = now() WHERE id = :i"),
                     {"i": user["id"]})
    assert auth_store.get_session_user(token) is None


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


def test_raw_token_is_not_stored(auth_store, db_cfg):
    """A dump of the sessions table must not yield a usable cookie."""
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    with get_engine(db_cfg).connect() as conn:
        stored = [r[0] for r in conn.execute(text("SELECT token_hash FROM sessions"))]
    assert token not in stored
    assert len(stored) == 1 and len(stored[0]) == 64


def test_expired_session_is_refused(auth_store, db_cfg):
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    with get_engine(db_cfg).begin() as conn:
        conn.execute(text("UPDATE sessions SET expires_at = now() - interval '1 second'"))
    assert auth_store.get_session_user(token) is None


def test_purge_expired_removes_only_expired_sessions(auth_store, db_cfg):
    user = auth_store.create_user("ada@example.com", PASSWORD)
    live = auth_store.create_session(user["id"])
    stale = auth_store.create_session(user["id"])
    with get_engine(db_cfg).begin() as conn:
        conn.execute(
            text("UPDATE sessions SET expires_at = now() - interval '1 day' "
                 "WHERE token_hash = :h"),
            {"h": AuthStore._token_hash(stale)},
        )
    assert auth_store.purge_expired() == 1
    assert auth_store.get_session_user(live) is not None


def test_deleting_a_user_drops_their_sessions(auth_store, db_cfg):
    """ON DELETE CASCADE in the database, not in the application — a delete issued
    any other way must not leave orphaned sessions behind."""
    user = auth_store.create_user("ada@example.com", PASSWORD)
    token = auth_store.create_session(user["id"])
    with get_engine(db_cfg).begin() as conn:
        conn.execute(text("DELETE FROM users WHERE id = :i"), {"i": user["id"]})
    assert auth_store.get_session_user(token) is None


def test_sign_out_everywhere(auth_store):
    user = auth_store.create_user("ada@example.com", PASSWORD)
    tokens = [auth_store.create_session(user["id"]) for _ in range(3)]
    assert auth_store.delete_user_sessions(user["id"]) == 3
    assert all(auth_store.get_session_user(t) is None for t in tokens)


# -- federated identity ----------------------------------------------------- #
def test_first_google_sign_in_creates_an_account(auth_store):
    user = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com", "Ada")
    assert user["email"] == "ada@example.com"
    assert user["name"] == "Ada"
    assert auth_store.list_identities(user["id"]) == ["google"]


def test_second_sign_in_returns_the_same_account(auth_store):
    first = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com", "Ada")
    second = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com", "Ada")
    assert first["id"] == second["id"]


def test_google_links_to_an_existing_password_account(auth_store):
    """The whole point of keying on a verified address: someone who signed up with a
    password and later clicks the Google button lands in the same library."""
    original = auth_store.create_user("ada@example.com", PASSWORD, name="Ada")
    linked = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com", "Ada L")
    assert linked["id"] == original["id"]
    assert auth_store.list_identities(original["id"]) == ["google"]
    # The password still works — linking adds a way in, it does not replace one.
    assert auth_store.authenticate("ada@example.com", PASSWORD)["id"] == original["id"]


def test_two_providers_resolve_to_one_account(auth_store):
    """Google and Entra for the same verified address are one person, not two."""
    google = auth_store.upsert_oauth_user("google", "g-1", "ada@example.com", "Ada")
    entra = auth_store.upsert_oauth_user("microsoft", "m-1", "ada@example.com", "Ada")
    assert google["id"] == entra["id"]
    assert sorted(auth_store.list_identities(google["id"])) == ["google", "microsoft"]


def test_same_subject_from_different_providers_stays_separate(auth_store):
    """`sub` is unique only within its provider, so the pair is the key. Two
    providers that happen to issue the same subject string are different people."""
    a = auth_store.upsert_oauth_user("google", "collide", "one@example.com")
    b = auth_store.upsert_oauth_user("microsoft", "collide", "two@example.com")
    assert a["id"] != b["id"]


def test_identity_follows_the_subject_when_the_address_changes(auth_store):
    """A user who renames their mailbox keeps their library. This is why the store
    keys on `sub` and not on the email."""
    original = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com", "Ada")
    renamed = auth_store.upsert_oauth_user("google", "sub-1", "ada.lovelace@example.com")
    assert renamed["id"] == original["id"]
    assert renamed["email"] == "ada.lovelace@example.com"


def test_a_rename_onto_a_taken_address_does_not_fail_the_sign_in(auth_store):
    """Someone else already holds the new address. Keeping the old one is wrong-ish,
    but locking the user out of their own library over it is worse."""
    auth_store.create_user("taken@example.com", PASSWORD)
    original = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com")
    again = auth_store.upsert_oauth_user("google", "sub-1", "taken@example.com")
    assert again["id"] == original["id"]
    assert again["email"] == "ada@example.com"


def test_provider_only_account_has_no_password(auth_store, db_cfg):
    """No password exists, so no password may open it — including the empty one."""
    user = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com")
    with get_engine(db_cfg).connect() as conn:
        stored = conn.execute(
            text("SELECT password_hash FROM users WHERE id = :i"), {"i": user["id"]}
        ).scalar()
    assert stored is None
    assert auth_store.authenticate("ada@example.com", "") is None
    assert auth_store.authenticate("ada@example.com", PASSWORD) is None


def test_provider_account_can_be_given_a_password_later(auth_store):
    user = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com")
    assert auth_store.set_password(user["id"], PASSWORD) is True
    assert auth_store.authenticate("ada@example.com", PASSWORD)["id"] == user["id"]


def test_disabled_account_cannot_sign_in_through_a_provider(auth_store, db_cfg):
    user = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com")
    with get_engine(db_cfg).begin() as conn:
        conn.execute(text("UPDATE users SET disabled_at = now() WHERE id = :i"),
                     {"i": user["id"]})
    with pytest.raises(AccountDisabled):
        auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com")


def test_oauth_sessions_record_which_provider_issued_them(auth_store, db_cfg):
    """So that revoking a provider can cut exactly the sessions it minted."""
    user = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com")
    auth_store.create_session(user["id"], auth_method="google")
    with get_engine(db_cfg).connect() as conn:
        assert conn.execute(text("SELECT auth_method FROM sessions")).scalar() == "google"


def test_upsert_requires_provider_and_subject(auth_store):
    with pytest.raises(ValueError):
        auth_store.upsert_oauth_user("google", "", "ada@example.com")


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


def test_ready_reports_the_schema_revision(client):
    body = client.get("/api/ready").json()
    assert body["status"] == "ok"
    assert body["schema_revision"] == "0001_accounts"
    # The probe response is not a place to publish a database password.
    assert ":***@" in body["database_url"]


# -- federated sign-in routes ----------------------------------------------- #
def test_providers_are_reported_as_unconfigured_by_default(client):
    providers = client.get("/api/auth/providers").json()["providers"]
    assert {p["key"] for p in providers} == {"google", "microsoft"}
    assert all(p["enabled"] is False for p in providers)


def test_start_on_an_unconfigured_provider_is_refused(client):
    r = client.get("/api/auth/oidc/google/start", follow_redirects=False)
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]


def test_start_on_an_unknown_provider_is_404(client):
    assert client.get("/api/auth/oidc/facebook/start").status_code == 404


def test_start_redirects_to_the_provider_when_configured(client, monkeypatch):
    from inkference.api import main
    from inkference.auth import oidc

    provider = OIDCProvider(
        key="google", label="Google",
        discovery_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id="cid", client_secret="secret",
    )
    monkeypatch.setattr(main, "oidc_cfg", OIDCConfig(providers={"google": provider}))
    monkeypatch.setattr(
        oidc, "discover",
        lambda p: oidc.Discovery(
            issuer="https://accounts.google.com",
            authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
        ),
    )

    r = client.get("/api/auth/oidc/google/start", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "code_challenge_method=S256" in r.headers["location"]
    # The state cookie must not be readable by page scripts, and must be scoped to
    # the OAuth routes rather than sent with every request to the app.
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "path=/api/auth/oidc" in cookie


def test_callback_without_a_state_cookie_redirects_with_an_error(client, monkeypatch):
    from inkference.api import main

    provider = OIDCProvider(key="google", label="Google", discovery_url="x",
                            client_id="cid", client_secret="secret")
    monkeypatch.setattr(main, "oidc_cfg", OIDCConfig(providers={"google": provider}))

    r = client.get("/api/auth/oidc/google/callback?code=abc&state=xyz",
                   follow_redirects=False)
    # A browser arrived here by top-level navigation, so an error has to land
    # somewhere that can render it — never as a JSON body.
    assert r.status_code == 303
    assert "auth_error=" in r.headers["location"]
    assert client.get("/api/auth/me").json()["user"] is None


def test_cancelled_consent_returns_a_friendly_message(client, monkeypatch):
    from inkference.api import main

    provider = OIDCProvider(key="google", label="Google", discovery_url="x",
                            client_id="cid", client_secret="secret")
    monkeypatch.setattr(main, "oidc_cfg", OIDCConfig(providers={"google": provider}))

    r = client.get("/api/auth/oidc/google/callback?error=access_denied",
                   follow_redirects=False)
    assert r.status_code == 303
    assert "cancelled" in r.headers["location"].lower()


# -- redirect URI derivation ------------------------------------------------ #
# Getting this wrong is the single most common OAuth deployment failure: behind
# Container Apps' ingress the app sees plain http on an internal address, and a
# redirect URI built from that is rejected by every provider.
def _request(headers: dict, base: str = "http://10.0.0.4:8000/"):
    from starlette.datastructures import Headers

    class _Req:
        def __init__(self):
            self.headers = Headers(headers)
            self.base_url = base

    return _Req()


def test_public_base_url_wins_over_everything(monkeypatch):
    from inkference.api import main

    monkeypatch.setattr(main.auth_cfg, "public_base_url", "https://inkference.example.org")
    request = _request({"x-forwarded-proto": "http", "x-forwarded-host": "attacker.test"})
    assert main._redirect_uri(request, "google") == (
        "https://inkference.example.org/api/auth/oidc/google/callback"
    )


def test_forwarded_headers_are_used_when_trusted(monkeypatch):
    from inkference.api import main

    monkeypatch.setattr(main.auth_cfg, "public_base_url", "")
    monkeypatch.setattr(main.auth_cfg, "trust_proxy_headers", True)
    request = _request({"x-forwarded-proto": "https", "x-forwarded-host": "app.example.org"})
    assert main._redirect_uri(request, "microsoft") == (
        "https://app.example.org/api/auth/oidc/microsoft/callback"
    )


def test_forwarded_headers_are_ignored_when_not_trusted(monkeypatch):
    """A client can forge these headers. On a directly-exposed app, honouring them
    would let a caller point the OAuth redirect at a host they control."""
    from inkference.api import main

    monkeypatch.setattr(main.auth_cfg, "public_base_url", "")
    monkeypatch.setattr(main.auth_cfg, "trust_proxy_headers", False)
    request = _request({"x-forwarded-proto": "https", "x-forwarded-host": "attacker.test"})
    assert main._redirect_uri(request, "google") == (
        "http://10.0.0.4:8000/api/auth/oidc/google/callback"
    )


def test_only_the_first_forwarded_entry_is_used(monkeypatch):
    """Chained proxies produce a comma-separated list; the first is nearest the client."""
    from inkference.api import main

    monkeypatch.setattr(main.auth_cfg, "public_base_url", "")
    monkeypatch.setattr(main.auth_cfg, "trust_proxy_headers", True)
    request = _request({
        "x-forwarded-proto": "https, http",
        "x-forwarded-host": "app.example.org, internal.local",
    })
    assert main._redirect_uri(request, "google").startswith("https://app.example.org/")


# -- deployment safety ------------------------------------------------------ #
def test_credentials_never_live_in_the_published_corpus_db():
    """deploy_all_books.sh uploads inkference.db to a PUBLIC HF dataset. Accounts
    live in PostgreSQL, so there is no file for them to be published in — this test
    fails the moment someone points the accounts store back at a SQLite path."""
    from inkference.config import database as database_cfg

    assert database_cfg.normalized_url.startswith("postgresql+psycopg://")


def test_database_url_from_a_managed_service_is_rewritten_to_our_driver():
    """Azure and Heroku hand out postgres:// and postgresql://, both of which
    SQLAlchemy maps to psycopg2 — a driver this project does not install."""
    from inkference.config import DatabaseConfig

    for given in ("postgres://u:p@h:5432/d", "postgresql://u:p@h:5432/d"):
        assert DatabaseConfig(url=given).normalized_url == "postgresql+psycopg://u:p@h:5432/d"


def test_safe_url_blanks_the_password():
    from inkference.config import DatabaseConfig

    safe = DatabaseConfig(url="postgresql://user:hunter2@host:5432/db").safe_url
    assert "hunter2" not in safe
    assert safe == "postgresql+psycopg://user:***@host:5432/db"


def test_a_non_postgres_url_is_rejected_at_startup():
    """A sqlite:// URL in an env file is a mistake worth stopping for, not something
    to limp along on until the first write races another replica."""
    from inkference.auth.db import _build_engine
    from inkference.config import DatabaseConfig

    with pytest.raises(RuntimeError, match="must be a PostgreSQL URL"):
        _build_engine(DatabaseConfig(url="sqlite:///tmp/auth.db"))


# -- migrations ------------------------------------------------------------- #
def test_schema_matches_the_models(migrated_database):
    """Every table and column the ORM expects must exist after `upgrade head`. This
    is what catches a model change that shipped without a migration."""
    from inkference.auth.models import Base
    from sqlalchemy import inspect

    inspector = inspect(get_engine(migrated_database))
    for table in Base.metadata.sorted_tables:
        assert inspector.has_table(table.name), f"missing table {table.name}"
        actual = {c["name"] for c in inspector.get_columns(table.name)}
        expected = {c.name for c in table.columns}
        assert expected <= actual, f"{table.name} is missing {expected - actual}"


def test_migrations_are_reversible(postgres_url):
    """A migration that cannot be undone is a deploy that cannot be rolled back."""
    from alembic import command
    from alembic.config import Config

    from inkference.auth.db import create_engine_with_pool
    from inkference.config import APP_ROOT, DatabaseConfig
    from sqlalchemy import inspect, make_url

    cfg = DatabaseConfig(url=postgres_url, pool_size=1, max_overflow=0)
    # A private engine: this test drops every table, and must not do that through
    # the pool the session-scoped fixtures are still holding.
    engine = create_engine_with_pool(cfg, make_url(cfg.normalized_url))
    alembic_cfg = Config(str(APP_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(APP_ROOT / "migrations"))
    try:
        with engine.begin() as conn:
            alembic_cfg.attributes["connection"] = conn
            command.downgrade(alembic_cfg, "base")
            assert not inspect(conn).has_table("users")
            command.upgrade(alembic_cfg, "head")
            assert inspect(conn).has_table("users")
    finally:
        engine.dispose()


def test_identity_is_unique_per_provider_and_subject(auth_store, db_cfg):
    """The constraint, not the application, is what makes double-linking impossible
    when two sign-ins for the same identity race."""
    from sqlalchemy.exc import IntegrityError

    user = auth_store.upsert_oauth_user("google", "sub-1", "ada@example.com")
    with pytest.raises(IntegrityError):
        with get_engine(db_cfg).begin() as conn:
            conn.execute(
                text("INSERT INTO oauth_identities (user_id, provider, subject) "
                     "VALUES (:u, 'google', 'sub-1')"),
                {"u": user["id"]},
            )
