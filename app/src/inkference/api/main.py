"""Inkference FastAPI app.

Every /api route below except /api/health and /api/auth/* requires a signed-in
session (see require_user). Set INKFERENCE_AUTH_REQUIRED=false to open the API up
for a public demo without removing accounts.

Endpoints:
  GET  /api/health
  POST /api/auth/signup                  create an account -> session cookie
  POST /api/auth/login                   session cookie
  POST /api/auth/logout                  clear the session
  GET  /api/auth/me                      current user (null when signed out)
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
import unicodedata
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..auth import EmailTaken
from ..config import (
    FRONTEND_DIR,
    IMAGES_BASE_URL,
    IMAGES_ROOT,
    agent as agent_cfg,
    auth as auth_cfg,
    correction as correction_cfg,
    htr as htr_cfg,
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

app = FastAPI(title="Inkference", version="0.1.0")

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
    email: str
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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


@app.post("/api/auth/signup")
def signup(body: SignupRequest, response: Response) -> dict:
    email = (body.email or "").strip()
    if not _EMAIL_RE.match(email):
        raise HTTPException(422, "enter a valid email address")
    if len(body.password or "") < auth_cfg.min_password_length:
        raise HTTPException(
            422, f"password must be at least {auth_cfg.min_password_length} characters"
        )
    store = services.get_auth_store()
    try:
        user = store.create_user(email, body.password, name=body.name)
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
    user = store.authenticate(body.email, body.password)
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
    return {"user": user, "auth_required": auth_cfg.required}


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "trocr_model": htr_cfg.trocr_model_id,
            "llm_provider": rag_cfg.llm_provider,
            "llm_configured": bool(rag_cfg.llm_api_key),
            "correction_enabled": correction_cfg.enabled,
            "correction_backend": correction_cfg.backend,
            "correction_model": (correction_cfg.api_model if correction_cfg.backend == "api"
                                 else correction_cfg.model_id),
            "agent_enabled": agent_cfg.enabled,
            "agent_max_steps": agent_cfg.max_steps,
            "auth_required": auth_cfg.required}


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
    slug = body.slug or _slugify(body.title)
    if store.get_document_by_slug(slug):
        raise HTTPException(409, f"document with slug '{slug}' already exists")
    doc_id = store.create_document(title=body.title, slug=slug, subtitle=body.subtitle)
    return {"id": doc_id, "slug": slug}


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
