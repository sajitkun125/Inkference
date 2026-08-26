"""Inkference FastAPI app.

Every /api route below except /api/health and /api/auth/* requires a signed-in
session (see require_user). Set INKFERENCE_AUTH_REQUIRED=false to open the API up
for a public demo without removing accounts.

Endpoints:
  GET  /api/health
  GET  /api/ready                        readiness probe (checks PostgreSQL)
  POST /api/auth/signup                  create an account -> session cookie
  POST /api/auth/login                   session cookie
  POST /api/auth/logout                  clear the session
  GET  /api/auth/me                      current user (null when signed out)
  GET  /api/auth/providers               which federated sign-ins are configured
  GET  /api/auth/oidc/{provider}/start   redirect to Google / Microsoft Entra ID
  GET  /api/auth/oidc/{provider}/callback   provider returns here -> session cookie
  GET  /api/documents
  POST /api/documents
  GET  /api/documents/{id}
  POST /api/documents/{id}/pages         upload scans -> background ingest job
  GET  /api/documents/{id}/pages/{n}     transcription (lines + words + confidence)
  GET  /api/documents/{id}/pages/{n}/image
  POST /api/documents/{id}/ask           RAG answer + source pages (fast path)
  POST /api/documents/{id}/agent         LangGraph research agent (multi-step + memory)
  DEL  /api/documents/{id}/agent/threads/{thread_id}   forget one conversation
  GET  /api/jobs/{id}                    ingestion progress
"""
from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel ,EmailStr, Field
# EmailUndeliverableError is not imported: it subclasses EmailNotValidError, so
# catching the base covers both syntax and deliverability failures.
from email_validator import caching_resolver, validate_email, EmailNotValidError

from ..auth import AccountDisabled, EmailTaken
from ..config import (
    FRONTEND_DIR,
    IMAGES_BASE_URL,
    IMAGES_ROOT,
    agent as agent_cfg,
    auth as auth_cfg,
    correction as correction_cfg,
    database as db_cfg,
    htr as htr_cfg,
    oidc as oidc_cfg,
    rag as rag_cfg,
)
from ..rag.answer import answer_question
from . import services


def _setup_logging() -> None:
    """Route all `inkference.*` loggers to the console at INKFERENCE_LOG_LEVEL
    (default INFO; set DEBUG for verbose). Independent of uvicorn's own loggers."""
    level = os.getenv("INKFERENCE_LOG_LEVEL", "INFO").upper()
    lg = logging.getLogger("inkference")
    lg.setLevel(level)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
        lg.propagate = False


_setup_logging()
logger = logging.getLogger("inkference.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Migrate the accounts database before the first request is served.

    Failure here is intentionally fatal: an app that starts without its accounts
    database serves a sign-in page that 500s on submit. Crashing instead means the
    Container Apps revision never goes healthy and the previous one keeps serving.
    """
    services.init_database()
    yield


app = FastAPI(title="Inkference", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_frontend(request, call_next):
    """Tell browsers to revalidate the static frontend so edits (js/css/html)
    always load fresh instead of serving a stale cached bundle."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #
class CreateDocument(BaseModel):
    title: str
    slug: str | None = None
    subtitle: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    persona: str | None = None  # "cook" -> answer in character as Captain Cook


class AgentAskRequest(BaseModel):
    question: str
    # Conversation id, minted by the client. Omit for a one-off (unremembered) turn.
    thread_id: str | None = None
    persona: str | None = None
    max_steps: int | None = None


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "document"


# --------------------------------------------------------------------------- #
# accounts + sessions
# --------------------------------------------------------------------------- #
class SignupRequest(BaseModel):
    email: EmailStr | None = Field(default=None)
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr | None = Field(default=None)
    password: str


#_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def current_user(session: str | None = Cookie(default=None, alias=auth_cfg.cookie_name)):
    """Resolve the session cookie to a user, or None. Never raises — routes that
    must have a user depend on require_user instead."""
    return services.get_auth_store().get_session_user(session)


def require_user(user=Depends(current_user)):
    """Gate a route behind a signed-in account.

    With INKFERENCE_AUTH_REQUIRED=false this waves everything through, so a public
    demo Space can stay open without removing the accounts system.
    """
    if not auth_cfg.required:
        return user
    if user is None:
        raise HTTPException(401, "sign in to continue")
    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        auth_cfg.cookie_name,
        token,
        max_age=auth_cfg.session_ttl_days * 24 * 3600,
        httponly=True,          # keeps the token away from any XSS on the page
        samesite="lax",
        secure=auth_cfg.cookie_secure,
        path="/",
    )


_email_resolver = None
_email_resolver_lock = threading.Lock()


def _dns_resolver():
    """One shared, caching, short-timeout DNS resolver for address validation.

    Built once and reused, for two reasons. A per-domain LRU cache (honouring TTLs)
    turns the deliverability check into one lookup per *new* domain rather than one
    per signup — most signups share a handful of domains. And passing our own
    resolver stops email_validator from reaching for dnspython's global default and
    overwriting its timeout, which it warns changes behaviour for every other user
    of dnspython in the process.

    Unreachable nameservers need no handling here: email_validator already treats
    Timeout and NoNameservers as "unknown", not as undeliverable, so a resolver
    outage cannot reject a legitimate address.
    """
    global _email_resolver
    if _email_resolver is None:
        with _email_resolver_lock:
            if _email_resolver is None:
                _email_resolver = caching_resolver(
                    timeout=auth_cfg.email_dns_timeout_seconds
                )
    return _email_resolver


@app.post("/api/auth/signup")
def signup(body: SignupRequest, response: Response) -> dict:
    check_dns = auth_cfg.email_check_deliverability
    try:
        validated = validate_email(
            body.email or "",
            check_deliverability=check_dns,
            dns_resolver=_dns_resolver() if check_dns else None,
        )
    except EmailNotValidError as exc:
        # EmailUndeliverableError subclasses this, so one clause covers both syntax
        # and deliverability. The library's own wording names the actual problem
        # ("does not exist", "does not accept email"), which is far more actionable
        # than a generic rejection.
        raise HTTPException(422, str(exc))

    if len(body.password or "") < auth_cfg.min_password_length:
        raise HTTPException(
            422, f"password must be at least {auth_cfg.min_password_length} characters"
        )
    store = services.get_auth_store()
    try:
        # The normalized form: domain lowercased and IDNs punycoded, so two
        # spellings of one address cannot become two accounts.
        user = store.create_user(validated.normalized, body.password, name=body.name)
    except EmailTaken:
        # Deliberately explicit: signup cannot hide that an address is taken, since
        # the account simply cannot be created twice.
        raise HTTPException(409, "an account with that email already exists")
    _set_session_cookie(response, store.create_session(user["id"]))
    logger.info("account created id=%s", user["id"])
    return {"user": user}


@app.post("/api/auth/login")
def login(body: LoginRequest, response: Response) -> dict:
    store = services.get_auth_store()
    try:
        user = store.authenticate(body.email, body.password)
    except AccountDisabled:
        raise HTTPException(403, "this account has been disabled")
    if user is None:
        # One message for both "no such account" and "wrong password" so the
        # response cannot be used to enumerate registered addresses.
        raise HTTPException(401, "email or password is incorrect")
    _set_session_cookie(response, store.create_session(user["id"]))
    return {"user": user}


@app.post("/api/auth/logout")
def logout(
    response: Response,
    session: str | None = Cookie(default=None, alias=auth_cfg.cookie_name),
) -> dict:
    services.get_auth_store().delete_session(session)
    response.delete_cookie(auth_cfg.cookie_name, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user=Depends(current_user)) -> dict:
    """Who am I? Drives the frontend gate, so it stays 200 when signed out."""
    return {
        "user": user,
        "auth_required": auth_cfg.required,
        "providers": _provider_list(),
    }


@app.get("/api/auth/providers")
def auth_providers() -> dict:
    """Which federated sign-ins this deployment can actually perform.

    Public: it reveals only which buttons work, which the sign-in page has to render
    before any session exists. Client ids are omitted — they are not secret, but
    nothing on this page needs them.
    """
    return {"providers": _provider_list()}


def _provider_list() -> list[dict]:
    return [
        {"key": p.key, "label": p.label, "enabled": p.enabled}
        for p in oidc_cfg.providers.values()
    ]


# --------------------------------------------------------------------------- #
# federated sign-in (OpenID Connect: Google, Microsoft Entra ID)
# --------------------------------------------------------------------------- #
# Both providers use the same authorization-code flow, so these two routes serve
# either one — the {provider} segment picks the config, and inkference.auth.oidc
# does the protocol work. See that module for what guards each step.
def _external_base_url(request: Request) -> str:
    """The origin a browser reached us on.

    Container Apps' ingress terminates TLS and forwards plain http to the container,
    so `request.url` reads http://<internal-ip>:8000 — useless as an OAuth redirect
    target. Order of preference: the configured public URL (always right), then the
    proxy's forwarded headers, then the raw request (correct only for direct local
    runs). Forwarded headers are honoured only when trust_proxy_headers is on,
    because a client can forge them.
    """
    if auth_cfg.public_base_url:
        return auth_cfg.public_base_url

    if auth_cfg.trust_proxy_headers:
        # Both headers are comma-separated lists when several proxies are chained;
        # the first entry is the one nearest the client.
        proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
        if proto and host:
            return f"{proto}://{host}"

    return str(request.base_url).rstrip("/")


def _redirect_uri(request: Request, provider_key: str) -> str:
    """Must match an authorized redirect URI registered with the provider, byte for
    byte — a trailing slash difference is enough for Google to refuse the exchange."""
    return f"{_external_base_url(request)}/api/auth/oidc/{provider_key}/callback"


def _require_provider(provider_key: str):
    provider = oidc_cfg.get(provider_key)
    if provider is None:
        raise HTTPException(404, "unknown sign-in provider")
    if not provider.enabled:
        raise HTTPException(
            503, f"{provider.label} sign-in is not configured on this deployment"
        )
    return provider


def _set_oauth_state_cookie(response: Response, value: str | None) -> None:
    """The in-flight sign-in, or None to clear it.

    SameSite=Lax rather than Strict: the callback arrives as a cross-site top-level
    navigation from the provider, and Strict would withhold the cookie exactly then,
    breaking every sign-in. Lax still sends it on that navigation while withholding
    it from cross-site subresource requests, which is the protection that matters.
    """
    if value is None:
        response.delete_cookie(_OAUTH_STATE_COOKIE, path="/api/auth/oidc")
        return
    response.set_cookie(
        _OAUTH_STATE_COOKIE,
        value,
        max_age=auth_cfg.oauth_state_ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=auth_cfg.cookie_secure,
        # Scoped to the OAuth routes: it is useless anywhere else, and a cookie that
        # is not sent is a cookie that cannot leak.
        path="/api/auth/oidc",
    )


_OAUTH_STATE_COOKIE = "inkference_oauth"


@app.get("/api/auth/oidc/{provider_key}/start")
def oidc_start(provider_key: str, request: Request):
    """Begin a federated sign-in: redirect the browser to the provider."""
    from ..auth import oidc

    provider = _require_provider(provider_key)
    try:
        url, sealed = oidc.begin(provider, _redirect_uri(request, provider.key), auth_cfg)
    except oidc.OIDCError as exc:
        # Discovery is the only thing that can fail this early, and it means the
        # provider is unreachable rather than that the user did anything wrong.
        logger.error("oidc start failed for %s: %s", provider.key, exc)
        raise HTTPException(502, f"could not reach {provider.label}")

    response = RedirectResponse(url, status_code=307)
    _set_oauth_state_cookie(response, sealed)
    return response


@app.get("/api/auth/oidc/{provider_key}/callback")
def oidc_callback(
    provider_key: str,
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    oauth_state: str | None = Cookie(default=None, alias=_OAUTH_STATE_COOKIE),
):
    """Finish a federated sign-in and land the browser back in the app.

    Always answers with a redirect, never JSON: the user's browser is here as a
    top-level navigation, so an error has to arrive somewhere that can render it.
    Failures go to #signin with a short, non-specific message — the detail goes to
    the log, since provider errors describe our configuration.
    """
    from ..auth import oidc

    provider = _require_provider(provider_key)

    if error:
        # The user pressed "Cancel" on the consent screen, or the provider refused.
        logger.info("oidc %s returned error=%s (%s)", provider.key, error, error_description)
        return _oidc_failure(
            "Sign-in was cancelled."
            if error == "access_denied"
            else f"{provider.label} could not sign you in."
        )
    if not code:
        return _oidc_failure("That sign-in link was incomplete. Please try again.")

    try:
        identity, return_to = oidc.complete(
            provider,
            code=code,
            state=state or "",
            sealed_state=oauth_state,
            redirect_uri=_redirect_uri(request, provider.key),
            cfg=auth_cfg,
        )
    except oidc.EmailNotVerified as exc:
        logger.warning("oidc %s: unverified address %s", provider.key, exc)
        return _oidc_failure(
            f"{provider.label} has not verified that email address, so it cannot be "
            "used to sign in."
        )
    except oidc.OIDCError as exc:
        logger.error("oidc %s callback failed: %s", provider.key, exc)
        return _oidc_failure("Sign-in could not be completed. Please try again.")

    store = services.get_auth_store()
    try:
        user = store.upsert_oauth_user(
            identity.provider, identity.subject, identity.email, identity.name
        )
    except AccountDisabled:
        return _oidc_failure("This account has been disabled.")

    logger.info("signed in via %s: user id=%s", provider.key, user["id"])
    target = return_to if return_to.startswith("/") else "/#library"
    response = RedirectResponse(target, status_code=303)
    _set_session_cookie(response, store.create_session(user["id"], auth_method=provider.key))
    _set_oauth_state_cookie(response, None)   # single use — it has done its job
    return response


def _oidc_failure(message: str) -> RedirectResponse:
    """Back to the sign-in page with a message the frontend renders in the error box."""
    response = RedirectResponse(f"/?auth_error={quote(message)}#signin", status_code=303)
    _set_oauth_state_cookie(response, None)
    return response


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    """Liveness. Answers from process state only — no database, no network — so a
    Postgres blip cannot get healthy containers restarted. Readiness is /api/ready."""
    return {"status": "ok", "trocr_model": htr_cfg.trocr_model_id,
            "llm_provider": rag_cfg.llm_provider,
            "llm_configured": bool(rag_cfg.llm_api_key),
            "correction_enabled": correction_cfg.enabled,
            "correction_backend": correction_cfg.backend,
            "correction_model": (correction_cfg.api_model if correction_cfg.backend == "api"
                                 else correction_cfg.model_id),
            # Reported because it decides whether correction works at all, and because a
            # server started before an .env edit keeps serving the old value — this is
            # the quickest way to tell a stale process from a real fault.
            "correction_reasoning_effort": correction_cfg.api_reasoning_effort or None,
            "agent_enabled": agent_cfg.enabled,
            "agent_max_steps": agent_cfg.max_steps,
            "auth_required": auth_cfg.required,
            "auth_providers": [p["key"] for p in _provider_list() if p["enabled"]]}


@app.get("/api/ready")
def ready(response: Response) -> dict:
    """Readiness. Point the Container Apps readiness probe here.

    Reports 503 while Postgres is unreachable, which takes this replica out of the
    ingress rotation instead of letting it answer every sign-in with a 500.
    """
    from ..auth.db import check_connection
    from ..auth.migrate import current_revision

    db_ok = check_connection(db_cfg)
    if not db_ok:
        response.status_code = 503
        return {"status": "degraded", "database": "unreachable"}
    return {
        "status": "ok",
        "database": "ok",
        "database_url": db_cfg.safe_url,   # password blanked by safe_url
        "schema_revision": current_revision(db_cfg),
    }


@app.get("/api/stats")
def corpus_stats() -> dict:
    """Corpus totals for the signed-out sign-in panel.

    Public on purpose — it is the only thing that page can show before a session
    exists, and it exposes counts only, never page text. Titles stay behind the gate.
    """
    docs = services.get_store().list_documents()
    pages = sum(d.get("page_count") or 0 for d in docs)
    scored = [(d.get("avg_confidence"), d.get("page_count") or 0) for d in docs
              if d.get("avg_confidence") is not None]
    weighted = sum(c * n for c, n in scored)
    total_scored = sum(n for _, n in scored)
    return {
        "documents": len(docs),
        "pages": pages,
        "avg_confidence": (weighted / total_scored) if total_scored else None,
    }


@app.get("/api/documents", dependencies=[Depends(require_user)])
def list_documents() -> list[dict]:
    return services.get_store().list_documents()


@app.post("/api/documents", dependencies=[Depends(require_user)])
def create_document(body: CreateDocument) -> dict:
    store = services.get_store()
    if body.slug:
        # An explicit slug is a request for that exact name, so a clash is a real
        # conflict the caller has to resolve.
        if store.get_document_by_slug(body.slug):
            raise HTTPException(409, f"document with slug '{body.slug}' already exists")
        slug = body.slug
    else:
        # Derived from the title, where a clash is ordinary: two books called
        # "Untitled manuscript" is a thing users do, and refusing the second one
        # would block "Add a book" for no reason the user could act on.
        slug = _unique_slug(store, _slugify(body.title))
    doc_id = store.create_document(title=body.title, slug=slug, subtitle=body.subtitle)
    return {"id": doc_id, "slug": slug}


def _unique_slug(store, base: str) -> str:
    """base, base-2, base-3, … — the first that is free."""
    if not store.get_document_by_slug(base):
        return base
    for n in range(2, 1000):
        candidate = f"{base}-{n}"
        if not store.get_document_by_slug(candidate):
            return candidate
    # A thousand books sharing one title is not a naming problem any more.
    raise HTTPException(409, f"too many documents named '{base}'")


@app.get("/api/documents/{doc_id}", dependencies=[Depends(require_user)])
def get_document(doc_id: int) -> dict:
    doc = services.get_store().get_document(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    return doc


# --------------------------------------------------------------------------- #
# pages: upload + ingest
# --------------------------------------------------------------------------- #
@app.post("/api/documents/{doc_id}/pages", dependencies=[Depends(require_user)])
async def upload_pages(doc_id: int, files: list[UploadFile] = File(...)) -> dict:
    store = services.get_store()
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")

    assets = store.cfg.assets_dir / doc["slug"]
    assets.mkdir(parents=True, exist_ok=True)
    start = doc["page_count"] + 1

    specs: list[tuple[int, int, str]] = []
    for offset, upload in enumerate(files):
        page_number = start + offset
        ext = Path(upload.filename or "").suffix.lower() or ".png"
        dest = assets / f"page_{page_number:04d}{ext}"
        dest.write_bytes(await upload.read())
        page_id = store.add_page(doc_id, page_number, image_path=str(dest))
        specs.append((page_id, page_number, str(dest)))

    job_id = store.create_job(doc_id, total_pages=len(specs))
    services.submit_ingest(doc_id, specs, job_id)
    return {"job_id": job_id, "pages": [s[1] for s in specs]}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_user)])
def get_job(job_id: int) -> dict:
    job = services.get_store().get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# --------------------------------------------------------------------------- #
# pages: read
# --------------------------------------------------------------------------- #
@app.get("/api/documents/{doc_id}/pages/{page_number}", dependencies=[Depends(require_user)])
def get_page(doc_id: int, page_number: int) -> dict:
    page = services.get_store().get_page(doc_id, page_number)
    if not page:
        raise HTTPException(404, "page not found")
    page.pop("image_path", None)  # internal path; image served via dedicated route
    return page


@app.get(
    "/api/documents/{doc_id}/pages/{page_number}/image",
    dependencies=[Depends(require_user)],
)
def get_page_image(doc_id: int, page_number: int):
    path = services.get_store().get_page_image_path(doc_id, page_number)
    if not path:
        raise HTTPException(404, "page image not found")
    p = Path(path)
    # Page NUMBERS are reused across uploads/reseeds, so /pages/{n}/image can map to
    # different bytes over time. Forbid browser caching so a reused number never serves
    # a stale scan (the "correct transcript but wrong old page image" bug).
    no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    # Remote images (deployment): redirect relative keys to a CDN/dataset base URL
    # so large image sets don't need to be baked into the app.
    if IMAGES_BASE_URL and not p.is_absolute():
        return RedirectResponse(f"{IMAGES_BASE_URL.rstrip('/')}/{path}", headers=no_cache)
    # Local: absolute path (self-contained seeds) or relative key under IMAGES_ROOT.
    for candidate in (p, (IMAGES_ROOT / path) if IMAGES_ROOT else None):
        if candidate and candidate.exists():
            return FileResponse(candidate, headers=no_cache)
    raise HTTPException(404, "page image not found")


# --------------------------------------------------------------------------- #
# ask the archive (RAG)
# --------------------------------------------------------------------------- #
@app.post("/api/documents/{doc_id}/ask", dependencies=[Depends(require_user)])
def ask(doc_id: int, body: AskRequest) -> dict:
    store = services.get_store()
    if not store.get_document(doc_id):
        raise HTTPException(404, "document not found")
    index = services.get_index()
    if not index.exists(doc_id):
        logger.info("building RAG index for doc %s", doc_id)
        index.build_from_store(doc_id, store)
    logger.info("ask doc=%s persona=%s q=%r", doc_id, body.persona, body.question[:100])
    ans = answer_question(doc_id, body.question, index, top_k=body.top_k, persona=body.persona)
    logger.info("ask doc=%s -> sources=%s", doc_id, ans.source_pages)
    return ans.to_dict()


# --------------------------------------------------------------------------- #
# ask the archive (LangGraph agent)
# --------------------------------------------------------------------------- #
# /ask above stays the fast path (~1 LLM call). The agent adds a tool-using loop
# for the questions /ask cannot serve: narrative "what happened next" questions
# that need pages read in order, and follow-ups that need conversation memory.
@app.post("/api/documents/{doc_id}/agent", dependencies=[Depends(require_user)])
def ask_agent(doc_id: int, body: AgentAskRequest) -> dict:
    if not agent_cfg.enabled:
        raise HTTPException(503, "agent is disabled (AGENT_ENABLED=false)")
    if not services.get_store().get_document(doc_id):
        raise HTTPException(404, "document not found")
    ans = services.run_agent(
        doc_id, body.question,
        thread_id=body.thread_id, persona=body.persona, max_steps=body.max_steps,
    )
    return ans.to_dict()


@app.delete(
    "/api/documents/{doc_id}/agent/threads/{thread_id}",
    dependencies=[Depends(require_user)],
)
def delete_agent_thread(doc_id: int, thread_id: str) -> dict:
    """Forget one conversation ("New conversation" in the UI).

    Doc-scoped because runner.run_agent namespaces thread ids as "{doc_id}:{id}".
    """
    from ..agent.checkpoint import delete_thread

    return {"deleted": delete_thread(f"{doc_id}:{thread_id}"), "thread_id": thread_id}


# --------------------------------------------------------------------------- #
# static frontend (mounted last so /api/* wins)
# --------------------------------------------------------------------------- #
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
