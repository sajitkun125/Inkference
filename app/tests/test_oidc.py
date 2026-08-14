"""The OpenID Connect flow, exercised without a network or a database.

Discovery and the JWKS are stubbed and id_tokens are signed with a throwaway RSA
key, so these run offline and fast. What they actually pin down is the set of
checks that stand between a callback URL and a session: signature, issuer,
audience, expiry, nonce, state, and the email-verification rule.

Each negative test names the attack it forecloses — if one of these ever starts
failing, the fix is not to relax the assertion.
"""
from __future__ import annotations

import datetime as _dt
import json

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from inkference.auth import oidc
from inkference.config import AuthConfig, OIDCProvider

CLIENT_ID = "test-client-id.apps.googleusercontent.com"
ISSUER = "https://accounts.google.com"
SUBJECT = "1234567890"
REDIRECT = "https://inkference.example.org/api/auth/oidc/google/callback"


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def cfg():
    return AuthConfig(secret_key="test-signing-secret", oauth_state_ttl_seconds=600)


@pytest.fixture
def provider():
    return OIDCProvider(
        key="google",
        label="Google",
        discovery_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=CLIENT_ID,
        client_secret="test-client-secret",
    )


@pytest.fixture(autouse=True)
def stub_discovery(monkeypatch, rsa_key):
    """No network: fixed endpoints, and a JWKS that holds only our test key."""
    monkeypatch.setattr(
        oidc, "discover",
        lambda p: oidc.Discovery(
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
        ),
    )

    class _Key:
        key = rsa_key.public_key()

    class _Client:
        def get_signing_key_from_jwt(self, _token):
            return _Key()

    monkeypatch.setattr(oidc, "_jwk_client", lambda _uri: _Client())


def make_id_token(rsa_key, **overrides) -> str:
    now = _dt.datetime.now(_dt.timezone.utc)
    claims = {
        "iss": ISSUER,
        "aud": CLIENT_ID,
        "sub": SUBJECT,
        "iat": int(now.timestamp()),
        "exp": int((now + _dt.timedelta(minutes=10)).timestamp()),
        "email": "ada@example.com",
        "email_verified": True,
        "name": "Ada Lovelace",
        "nonce": "test-nonce",
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": "test"})


_DEFAULT = object()   # distinct from None, which several tests pass deliberately


def run_callback(provider, cfg, rsa_key, monkeypatch, *, make_token=None,
                 state=_DEFAULT, sealed=_DEFAULT, **claims):
    """Drive begin() -> complete() with the token endpoint stubbed out.

    The nonce is minted inside begin(), so the id_token has to be built here rather
    than by the caller — otherwise every test would trip the nonce check first and
    never reach the assertion it cares about. Pass `nonce=` explicitly to test that
    check itself, or `make_token=` to control the signature.
    """
    _url, cookie = oidc.begin(provider, REDIRECT, cfg)
    payload = oidc._unseal(cookie, cfg)
    claims.setdefault("nonce", payload["n"])
    id_token = make_token(payload["n"]) if make_token else make_id_token(rsa_key, **claims)

    monkeypatch.setattr(oidc, "_redeem_code", lambda *a, **k: {"id_token": id_token})
    return oidc.complete(
        provider,
        code="auth-code",
        state=payload["s"] if state is _DEFAULT else state,
        sealed_state=cookie if sealed is _DEFAULT else sealed,
        redirect_uri=REDIRECT,
        cfg=cfg,
    )


# -- discovery --------------------------------------------------------------- #
# These stub `requests`, not `discover`, so the real caching path runs. Every other
# test here replaces `discover` wholesale, which once let a NameError inside it ship
# to a running server — the unit tests were green and /start returned 500.
@pytest.fixture
def unstubbed_discovery(monkeypatch):
    monkeypatch.undo()          # drop the module-wide `discover` stub
    oidc.reset_caches()
    yield
    oidc.reset_caches()


def _fake_requests(monkeypatch, doc, counter):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            counter.append(1)
            return doc

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())


def test_discovery_reads_the_endpoints(provider, unstubbed_discovery, monkeypatch):
    calls = []
    _fake_requests(monkeypatch, {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    }, calls)

    discovery = oidc.discover(provider)
    assert discovery.issuer == ISSUER
    assert discovery.token_endpoint == "https://oauth2.googleapis.com/token"


def test_discovery_is_cached(provider, unstubbed_discovery, monkeypatch):
    """One fetch per provider per TTL, not one per sign-in."""
    calls = []
    _fake_requests(monkeypatch, {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    }, calls)

    oidc.discover(provider)
    oidc.discover(provider)
    assert len(calls) == 1


def test_incomplete_discovery_document_is_refused(provider, unstubbed_discovery, monkeypatch):
    _fake_requests(monkeypatch, {"issuer": ISSUER}, [])
    with pytest.raises(oidc.OIDCError, match="missing"):
        oidc.discover(provider)


# -- the authorize request --------------------------------------------------- #
def test_begin_requests_pkce_and_a_nonce(provider, cfg):
    url, cookie = oidc.begin(provider, REDIRECT, cfg)
    assert "code_challenge=" in url and "code_challenge_method=S256" in url
    assert "nonce=" in url and "state=" in url
    assert f"client_id={CLIENT_ID}" in url.replace("%2E", ".")

    payload = oidc._unseal(cookie, cfg)
    # The verifier must live only in the cookie. In the URL it would be visible to
    # the provider and to anything logging the redirect, defeating PKCE entirely.
    assert payload["v"] not in url


def test_begin_mints_fresh_secrets_each_time(provider, cfg):
    first = oidc._unseal(oidc.begin(provider, REDIRECT, cfg)[1], cfg)
    second = oidc._unseal(oidc.begin(provider, REDIRECT, cfg)[1], cfg)
    assert first["s"] != second["s"]
    assert first["n"] != second["n"]
    assert first["v"] != second["v"]


# -- the happy path ---------------------------------------------------------- #
def test_valid_callback_yields_an_identity(provider, cfg, rsa_key, monkeypatch):
    identity, _return_to = run_callback(provider, cfg, rsa_key, monkeypatch)
    assert identity.provider == "google"
    assert identity.subject == SUBJECT
    assert identity.email == "ada@example.com"
    assert identity.name == "Ada Lovelace"


def test_email_is_normalized_to_lowercase(provider, cfg, rsa_key, monkeypatch):
    identity, _ = run_callback(provider, cfg, rsa_key, monkeypatch,
                               email="Ada@Example.COM")
    assert identity.email == "ada@example.com"


# -- state: login CSRF ------------------------------------------------------- #
def test_state_mismatch_is_refused(provider, cfg, rsa_key, monkeypatch):
    """Without this, an attacker can hand a victim a callback URL for the attacker's
    own sign-in and land the victim in the attacker's account."""
    with pytest.raises(oidc.OIDCError, match="state mismatch"):
        run_callback(provider, cfg, rsa_key, monkeypatch, state="not-the-state")


def test_missing_state_cookie_is_refused(provider, cfg, rsa_key, monkeypatch):
    """A callback with no cookie behind it — a bare URL someone was handed, or a
    flow that began in a different browser."""
    with pytest.raises(oidc.OIDCError, match="no sign-in is in progress"):
        run_callback(provider, cfg, rsa_key, monkeypatch, sealed=None)


def test_forged_state_cookie_is_refused(provider, cfg, rsa_key, monkeypatch):
    """A cookie minted by anyone but us must not verify — otherwise the attacker
    supplies both halves of the state check and it proves nothing."""
    forged = oidc._seal(
        {"p": "google", "s": "x", "n": "y", "v": "z", "r": "",
         "e": int(_dt.datetime.now(_dt.timezone.utc).timestamp()) + 600},
        AuthConfig(secret_key="a-different-secret"),
    )
    with pytest.raises(oidc.OIDCError, match="signature does not verify"):
        run_callback(provider, cfg, rsa_key, monkeypatch, sealed=forged, state="x")


def test_tampered_state_cookie_payload_is_refused(provider, cfg, rsa_key, monkeypatch):
    _url, cookie = oidc.begin(provider, REDIRECT, cfg)
    body, sig = cookie.split(".", 1)
    payload = json.loads(oidc._b64url_decode(body))
    payload["s"] = "attacker-chosen-state"
    tampered = f"{oidc._b64url(json.dumps(payload).encode())}.{sig}"
    with pytest.raises(oidc.OIDCError, match="signature does not verify"):
        run_callback(provider, cfg, rsa_key, monkeypatch, sealed=tampered,
                     state="attacker-chosen-state")


def test_expired_state_is_refused(provider, rsa_key, monkeypatch):
    expired = AuthConfig(secret_key="test-signing-secret", oauth_state_ttl_seconds=-1)
    with pytest.raises(oidc.OIDCError, match="took too long"):
        run_callback(provider, expired, rsa_key, monkeypatch)


def test_state_from_another_provider_is_refused(cfg, rsa_key, monkeypatch):
    """Two providers, two sets of credentials: a flow begun against Microsoft must
    not be completable against Google."""
    microsoft = OIDCProvider(
        key="microsoft", label="Microsoft",
        discovery_url="https://login.microsoftonline.com/x/v2.0/.well-known/openid-configuration",
        client_id=CLIENT_ID, client_secret="s",
    )
    google = OIDCProvider(
        key="google", label="Google",
        discovery_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=CLIENT_ID, client_secret="s",
    )
    _url, cookie = oidc.begin(microsoft, REDIRECT, cfg)
    payload = oidc._unseal(cookie, cfg)
    monkeypatch.setattr(oidc, "_redeem_code", lambda *a, **k: {"id_token": "unused"})
    with pytest.raises(oidc.OIDCError, match="different provider"):
        oidc.complete(google, code="c", state=payload["s"], sealed_state=cookie,
                      redirect_uri=REDIRECT, cfg=cfg)


# -- the id_token ------------------------------------------------------------ #
def test_nonce_mismatch_is_refused(provider, cfg, rsa_key, monkeypatch):
    """Binds the token to this authorization request. Without it, an id_token
    captured from a different sign-in could be replayed into this callback."""
    with pytest.raises(oidc.OIDCError, match="nonce mismatch"):
        run_callback(provider, cfg, rsa_key, monkeypatch, nonce="some-other-nonce")


def test_token_for_another_audience_is_refused(provider, cfg, rsa_key, monkeypatch):
    """`aud` pins the token to THIS OAuth client. Any Google app can mint a valid,
    correctly-signed id_token; only ours may be accepted here."""
    with pytest.raises(oidc.OIDCError, match="id_token rejected"):
        run_callback(provider, cfg, rsa_key, monkeypatch,
                     aud="some-other-app.apps.googleusercontent.com")


def test_token_from_another_issuer_is_refused(provider, cfg, rsa_key, monkeypatch):
    with pytest.raises(oidc.OIDCError, match="id_token rejected"):
        run_callback(provider, cfg, rsa_key, monkeypatch, iss="https://evil.example.com")


def test_expired_token_is_refused(provider, cfg, rsa_key, monkeypatch):
    past = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=2)
    with pytest.raises(oidc.OIDCError, match="id_token rejected"):
        run_callback(provider, cfg, rsa_key, monkeypatch,
                     exp=int(past.timestamp()), iat=int(past.timestamp()))


def test_token_signed_by_the_wrong_key_is_refused(provider, cfg, rsa_key, monkeypatch):
    """The whole point of verifying against the JWKS rather than trusting TLS."""
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def sign_with_attacker_key(nonce):
        now = _dt.datetime.now(_dt.timezone.utc)
        return jwt.encode(
            {"iss": ISSUER, "aud": CLIENT_ID, "sub": SUBJECT,
             "iat": int(now.timestamp()),
             "exp": int((now + _dt.timedelta(minutes=10)).timestamp()),
             "email": "ada@example.com", "email_verified": True, "nonce": nonce},
            attacker_key, algorithm="RS256", headers={"kid": "test"},
        )

    with pytest.raises(oidc.OIDCError, match="id_token rejected"):
        run_callback(provider, cfg, rsa_key, monkeypatch, make_token=sign_with_attacker_key)


def test_unsigned_token_is_refused(provider, cfg, rsa_key, monkeypatch):
    """alg=none is the oldest JWT attack there is."""
    def unsigned(nonce):
        return jwt.encode({"iss": ISSUER, "aud": CLIENT_ID, "sub": SUBJECT,
                           "email": "ada@example.com", "email_verified": True,
                           "nonce": nonce},
                          key="", algorithm="none")

    with pytest.raises(oidc.OIDCError, match="id_token rejected"):
        run_callback(provider, cfg, rsa_key, monkeypatch, make_token=unsigned)


def test_token_missing_required_claims_is_refused(provider, cfg, rsa_key, monkeypatch):
    """An absent claim must fail like a wrong one — PyJWT skips checks for claims
    a token simply omits unless `require` names them."""
    with pytest.raises(oidc.OIDCError, match="id_token rejected"):
        run_callback(provider, cfg, rsa_key, monkeypatch, exp=None)


# -- the email-verification rule --------------------------------------------- #
def test_unverified_google_address_is_refused(provider, cfg, rsa_key, monkeypatch):
    """AuthStore.upsert_oauth_user adopts an existing password account with a
    matching address. Accepting an unverified claim would therefore be an
    account-takeover primitive, not merely untidy."""
    with pytest.raises(oidc.EmailNotVerified):
        run_callback(provider, cfg, rsa_key, monkeypatch, email_verified=False)


def test_token_without_an_address_is_refused(provider, cfg, rsa_key, monkeypatch):
    with pytest.raises(oidc.OIDCError, match="no email address"):
        run_callback(provider, cfg, rsa_key, monkeypatch, email=None, email_verified=None)


def test_entra_work_account_is_trusted_without_email_verified(cfg, rsa_key, monkeypatch):
    """Entra omits email_verified. A work/school address sits in a domain the tenant
    administers, so the tenant vouches for it."""
    tenant = "8f4c1e22-0000-4000-9000-abcdef123456"
    token = make_id_token(
        rsa_key, email=None, email_verified=None,
        preferred_username="ada@contoso.com", tid=tenant,
    )
    claims = jwt.decode(token, options={"verify_signature": False})
    provider = OIDCProvider(key="microsoft", label="Microsoft", discovery_url="x",
                            client_id=CLIENT_ID, client_secret="s")
    assert oidc._verified_email(provider, claims) == "ada@contoso.com"


def test_entra_personal_account_is_not_trusted(rsa_key):
    """A personal Microsoft account's address is self-asserted — its holder types it
    in — so it is exactly the case the verification rule exists to catch."""
    token = make_id_token(rsa_key, email=None, email_verified=None,
                          preferred_username="someone@outlook.com",
                          tid="9188040d-6c67-4c5b-b112-36a304b66dad")
    claims = jwt.decode(token, options={"verify_signature": False})
    provider = OIDCProvider(key="microsoft", label="Microsoft", discovery_url="x",
                            client_id=CLIENT_ID, client_secret="s")
    with pytest.raises(oidc.EmailNotVerified):
        oidc._verified_email(provider, claims)


def test_non_email_preferred_username_is_not_used_as_an_address(rsa_key):
    token = make_id_token(rsa_key, email=None, email_verified=None,
                          preferred_username="ada", tid="8f4c1e22-0000-4000-9000-abc")
    claims = jwt.decode(token, options={"verify_signature": False})
    provider = OIDCProvider(key="microsoft", label="Microsoft", discovery_url="x",
                            client_id=CLIENT_ID, client_secret="s")
    with pytest.raises(oidc.OIDCError, match="no email address"):
        oidc._verified_email(provider, claims)


# -- Entra's templated issuer ------------------------------------------------ #
def test_multi_tenant_issuer_resolves_from_the_token_tenant(rsa_key):
    tenant = "8f4c1e22-0000-4000-9000-abcdef123456"
    token = make_id_token(rsa_key, tid=tenant)
    resolved = oidc._expected_issuer(
        "https://login.microsoftonline.com/{tenantid}/v2.0", token
    )
    assert resolved == f"https://login.microsoftonline.com/{tenant}/v2.0"


def test_multi_tenant_token_without_a_tenant_is_refused(rsa_key):
    token = make_id_token(rsa_key)          # no tid claim
    with pytest.raises(oidc.OIDCError, match="no tenant id"):
        oidc._expected_issuer("https://login.microsoftonline.com/{tenantid}/v2.0", token)


def test_single_tenant_issuer_is_left_alone(rsa_key):
    issuer = "https://login.microsoftonline.com/8f4c1e22/v2.0"
    assert oidc._expected_issuer(issuer, make_id_token(rsa_key)) == issuer
