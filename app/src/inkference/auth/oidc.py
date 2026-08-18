"""OpenID Connect authorization-code flow, written once for every provider.

Google and Microsoft Entra ID both publish a discovery document and a JWKS, so
neither needs bespoke code — `config.OIDCProvider` supplies the two URLs and the
credentials, and everything below is shared.

The flow, and what protects each step:

1. `begin()` mints `state`, a PKCE `code_verifier`, and a `nonce`, seals all three
   into a signed cookie, and returns the provider URL to redirect the browser to.
2. The provider authenticates the user and sends them back with a `code`.
3. `complete()` re-opens the cookie, checks `state` matches (login CSRF), redeems
   the code with the `code_verifier` (proves the redeemer started the flow), and
   verifies the returned `id_token`'s signature, issuer, audience, expiry, and
   `nonce` (proves the token was minted for this request, by the right provider).

The `id_token` signature is verified against the provider's published JWKS rather
than trusted because it arrived over TLS. The shortcut OpenID Connect Core §3.1.3.7
permits would be sound today — but it is sound only while the token comes straight
off the token endpoint, and that is exactly the kind of precondition a later
refactor breaks silently. Verifying costs one cached HTTP fetch.
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from ..config import AuthConfig, OIDCProvider
from ..config import auth as default_auth

logger = logging.getLogger("inkference.auth.oidc")

_HTTP_TIMEOUT = 15
# Discovery documents change on the order of years; an hour keeps a provider's key
# rotation from taking more than that to be noticed, at one request per hour.
_DISCOVERY_TTL_SECONDS = 3600
# Entra's "organizations"/"common" discovery returns this literal in place of the
# tenant, because the real issuer is only known once a token names its tenant.
_TENANT_PLACEHOLDER = "{tenantid}"
# The fixed tenant id Microsoft uses for personal accounts (outlook.com, hotmail).
_MSA_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad"


class OIDCError(Exception):
    """Any failure in the flow. Logged in full; never shown verbatim to the user,
    since provider errors carry configuration detail."""


class EmailNotVerified(OIDCError):
    """The provider would not confirm the address. Refused rather than trusted —
    see `_verified_email` for why this one is not negotiable."""


@dataclass(frozen=True)
class Identity:
    """A verified end user, as the provider describes them."""

    provider: str
    subject: str
    email: str
    name: str | None


@dataclass(frozen=True)
class Discovery:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str


# --------------------------------------------------------------------------- #
# discovery + signing keys, cached per provider
# --------------------------------------------------------------------------- #
_discovery_cache: dict[str, tuple[float, Discovery]] = {}
_jwk_clients: dict[str, Any] = {}
_cache_lock = threading.Lock()


def discover(provider: OIDCProvider) -> Discovery:
    """Fetch and cache the provider's OpenID configuration."""
    with _cache_lock:
        hit = _discovery_cache.get(provider.discovery_url)
        if hit and time.monotonic() - hit[0] < _DISCOVERY_TTL_SECONDS:
            return hit[1]

    import requests

    try:
        resp = requests.get(provider.discovery_url, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        doc = resp.json()
    except Exception as exc:  # requests raises a family; all of them mean the same here
        raise OIDCError(f"{provider.key}: discovery failed: {exc}") from exc

    try:
        discovery = Discovery(
            issuer=doc["issuer"],
            authorization_endpoint=doc["authorization_endpoint"],
            token_endpoint=doc["token_endpoint"],
            jwks_uri=doc["jwks_uri"],
        )
    except KeyError as exc:
        raise OIDCError(f"{provider.key}: discovery document missing {exc}") from exc

    with _cache_lock:
        _discovery_cache[provider.discovery_url] = (time.monotonic(), discovery)
    return discovery


def _jwk_client(jwks_uri: str):
    """One PyJWKClient per provider — it caches keys internally and refetches only
    when a token names a `kid` it has not seen, which is what makes provider key
    rotation a non-event here."""
    with _cache_lock:
        client = _jwk_clients.get(jwks_uri)
        if client is None:
            from jwt import PyJWKClient

            client = PyJWKClient(jwks_uri, cache_keys=True, max_cached_keys=16)
            _jwk_clients[jwks_uri] = client
        return client


def reset_caches() -> None:
    """Drop discovery and key caches. For tests, and for an operator who has just
    rotated a client and does not want to wait out the TTL."""
    with _cache_lock:
        _discovery_cache.clear()
        _jwk_clients.clear()


# --------------------------------------------------------------------------- #
# step 1: begin
# --------------------------------------------------------------------------- #
def begin(
    provider: OIDCProvider,
    redirect_uri: str,
    cfg: AuthConfig = default_auth,
    return_to: str = "",
) -> tuple[str, str]:
    """Start a sign-in. Returns (authorize_url, sealed_state_cookie_value)."""
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())

    params = {
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": provider.scopes,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        **dict(provider.extra_auth_params),
    }
    url = discover(provider).authorization_endpoint + "?" + urlencode(params)
    cookie = _seal(
        {
            "p": provider.key,
            "s": state,
            "n": nonce,
            "v": verifier,
            "r": return_to,
            "e": int(_utcnow().timestamp()) + cfg.oauth_state_ttl_seconds,
        },
        cfg,
    )
    return url, cookie


# --------------------------------------------------------------------------- #
# step 3: complete
# --------------------------------------------------------------------------- #
def complete(
    provider: OIDCProvider,
    *,
    code: str,
    state: str,
    sealed_state: str | None,
    redirect_uri: str,
    cfg: AuthConfig = default_auth,
) -> tuple[Identity, str]:
    """Turn a callback into a verified Identity. Returns (identity, return_to)."""
    payload = _unseal(sealed_state, cfg)
    if payload.get("p") != provider.key:
        raise OIDCError("state cookie belongs to a different provider")
    # compare_digest, not ==: state is a secret being compared against attacker-supplied
    # input, and a short-circuiting compare leaks its prefix through timing.
    if not state or not hmac.compare_digest(str(payload.get("s", "")), state):
        raise OIDCError("state mismatch — the sign-in did not start here")

    tokens = _redeem_code(provider, code, redirect_uri, str(payload["v"]))
    id_token = tokens.get("id_token")
    if not id_token:
        raise OIDCError("token response carried no id_token")

    claims = _verify_id_token(provider, id_token, expected_nonce=str(payload["n"]))
    identity = _identity_from_claims(provider, claims)
    return identity, str(payload.get("r") or "")


def _redeem_code(
    provider: OIDCProvider, code: str, redirect_uri: str, verifier: str
) -> dict[str, Any]:
    import requests

    try:
        resp = requests.post(
            discover(provider).token_endpoint,
            data={
                "code": code,
                "client_id": provider.client_id,
                "client_secret": provider.client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": verifier,
            },
            headers={"Accept": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )
    except Exception as exc:
        raise OIDCError(f"{provider.key}: token endpoint unreachable: {exc}") from exc

    if resp.status_code != 200:
        # The body is where the actionable part lives ("redirect_uri_mismatch",
        # "invalid_client", "AADSTS…"); a status code alone never fixed a deployment.
        raise OIDCError(
            f"{provider.key}: token exchange failed [{resp.status_code}]: {resp.text[:400]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise OIDCError(f"{provider.key}: token response was not JSON") from exc


def _verify_id_token(
    provider: OIDCProvider, id_token: str, expected_nonce: str
) -> dict[str, Any]:
    import jwt

    discovery = discover(provider)
    expected_issuer = _expected_issuer(discovery.issuer, id_token)

    try:
        signing_key = _jwk_client(discovery.jwks_uri).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            key=signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=provider.client_id,
            issuer=expected_issuer,
            # Absent claims are as dangerous as wrong ones: without `require`, PyJWT
            # simply skips the check for anything the token omits.
            options={"require": ["exp", "iat", "aud", "iss", "sub"]},
            leeway=60,          # tolerate small clock drift between us and the provider
        )
    except jwt.PyJWTError as exc:
        raise OIDCError(f"{provider.key}: id_token rejected: {exc}") from exc

    if not claims.get("sub"):
        raise OIDCError(f"{provider.key}: id_token carried no subject")
    nonce = claims.get("nonce")
    if not nonce or not hmac.compare_digest(str(nonce), expected_nonce):
        # Binds this token to the authorization request we started. Without it, an
        # id_token captured from another sign-in could be replayed into this callback.
        raise OIDCError(f"{provider.key}: nonce mismatch")
    return claims


def _expected_issuer(discovered_issuer: str, id_token: str) -> str:
    """Resolve Entra's templated issuer.

    Multi-tenant Entra discovery advertises `https://login.microsoftonline.com/
    {tenantid}/v2.0`, because the true issuer depends on which tenant authenticated
    the user. Substituting the token's own `tid` sounds circular, but it is not: the
    signature is still checked against the tenant-independent JWKS, so a forged `tid`
    yields a token that no published key signs.
    """
    if _TENANT_PLACEHOLDER not in discovered_issuer:
        return discovered_issuer
    import jwt

    try:
        unverified = jwt.decode(id_token, options={"verify_signature": False})
    except jwt.PyJWTError as exc:
        raise OIDCError(f"id_token is not a readable JWT: {exc}") from exc
    tid = str(unverified.get("tid") or "").strip()
    if not tid:
        raise OIDCError("multi-tenant id_token carried no tenant id")
    return discovered_issuer.replace(_TENANT_PLACEHOLDER, tid)


def _identity_from_claims(provider: OIDCProvider, claims: dict[str, Any]) -> Identity:
    email = _verified_email(provider, claims)
    name = (claims.get("name") or "").strip() or None
    return Identity(
        provider=provider.key,
        subject=str(claims["sub"]),
        email=email,
        name=name,
    )


def _verified_email(provider: OIDCProvider, claims: dict[str, Any]) -> str:
    """Extract the address, and refuse it unless the provider vouches for it.

    This is the load-bearing check of the whole file. AuthStore.upsert_oauth_user
    adopts an existing password account with a matching address, so an unverified
    claim would let anyone who can persuade a provider to emit `ada@example.com`
    walk into Ada's library. Refusing outright — rather than creating a parallel
    account — also denies the squatting move where an unverified sign-in reserves an
    address before its real owner registers.
    """
    email = ""
    for claim in ("email", "preferred_username", "upn"):
        candidate = (claims.get(claim) or "").strip()
        # Entra puts a UPN in preferred_username, which is usually but not always an
        # address; anything without an @ is a username, not something to key on.
        if candidate and "@" in candidate:
            email = candidate
            break
    if not email:
        raise OIDCError(f"{provider.key}: id_token carried no email address")

    verified = claims.get("email_verified")
    if verified is True or (isinstance(verified, str) and verified.lower() == "true"):
        return email.lower()
    if verified is not None:
        raise EmailNotVerified(email)

    # Entra omits email_verified entirely. For a work/school account the address sits
    # in a domain the tenant administers, so the tenant is the authority and the claim
    # is as good as verified. A personal Microsoft account is self-asserted — its
    # holder types their own address — so it is not.
    tid = str(claims.get("tid") or "").strip()
    if tid and tid != _MSA_TENANT:
        return email.lower()
    raise EmailNotVerified(email)


# --------------------------------------------------------------------------- #
# sealed state cookie
# --------------------------------------------------------------------------- #
# The cookie carries the PKCE verifier and the nonce, so it must be tamper-proof:
# an attacker who could mint one could start a flow the server would then accept as
# its own (login CSRF — the victim ends up signed into the attacker's account).
# HMAC over the payload is what makes it unforgeable. It stays a cookie rather than
# a server-side row so that any replica can finish a sign-in another one started.
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


_ephemeral_secret: str | None = None


def _signing_secret(cfg: AuthConfig) -> bytes:
    global _ephemeral_secret
    if cfg.secret_key:
        return cfg.secret_key.encode("utf-8")
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_urlsafe(48)
        logger.warning(
            "INKFERENCE_SECRET_KEY is unset — using a per-process key. Federated "
            "sign-in will fail whenever the callback lands on a different replica "
            "than the one that started it, and every restart invalidates flows in "
            "progress. Set it before scaling past one container."
        )
    return _ephemeral_secret.encode("utf-8")


def _seal(payload: dict[str, Any], cfg: AuthConfig) -> str:
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_signing_secret(cfg), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url(sig)}"


def _unseal(cookie: str | None, cfg: AuthConfig) -> dict[str, Any]:
    if not cookie:
        raise OIDCError("no sign-in is in progress (state cookie missing or expired)")
    try:
        body, sig = cookie.split(".", 1)
    except ValueError as exc:
        raise OIDCError("malformed state cookie") from exc

    expected = hmac.new(_signing_secret(cfg), body.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url(expected), sig):
        raise OIDCError("state cookie signature does not verify")

    try:
        payload = json.loads(_b64url_decode(body))
    except (ValueError, TypeError) as exc:
        raise OIDCError("state cookie payload is not JSON") from exc
    if not isinstance(payload, dict) or not {"p", "s", "n", "v", "e"} <= payload.keys():
        raise OIDCError("state cookie payload is incomplete")
    # Signed, so the expiry cannot have been edited — bound the window in which a
    # leaked cookie is worth anything.
    if _utcnow().timestamp() > float(payload["e"]):
        raise OIDCError("sign-in took too long — please try again")
    return payload
