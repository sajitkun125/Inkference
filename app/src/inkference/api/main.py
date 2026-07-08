"""Inkference FastAPI app.

Endpoints (see projectNotes/inkference_platform_plan.md):
  GET  /api/health
  GET  /api/documents
  POST /api/documents
  GET  /api/documents/{id}
  POST /api/documents/{id}/pages         upload scans -> background ingest job
  GET  /api/documents/{id}/pages/{n}     transcription (lines + words + confidence)
  GET  /api/documents/{id}/pages/{n}/image
  POST /api/documents/{id}/ask           RAG answer + source pages
  GET  /api/jobs/{id}                    ingestion progress
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import (
    FRONTEND_DIR,
    IMAGES_BASE_URL,
    IMAGES_ROOT,
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


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "document"


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
                                 else correction_cfg.model_id)}


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return services.get_store().list_documents()


@app.post("/api/documents")
def create_document(body: CreateDocument) -> dict:
    store = services.get_store()
    slug = body.slug or _slugify(body.title)
    if store.get_document_by_slug(slug):
        raise HTTPException(409, f"document with slug '{slug}' already exists")
    doc_id = store.create_document(title=body.title, slug=slug, subtitle=body.subtitle)
    return {"id": doc_id, "slug": slug}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int) -> dict:
    doc = services.get_store().get_document(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    return doc


# --------------------------------------------------------------------------- #
# pages: upload + ingest
# --------------------------------------------------------------------------- #
@app.post("/api/documents/{doc_id}/pages")
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


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int) -> dict:
    job = services.get_store().get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# --------------------------------------------------------------------------- #
# pages: read
# --------------------------------------------------------------------------- #
@app.get("/api/documents/{doc_id}/pages/{page_number}")
def get_page(doc_id: int, page_number: int) -> dict:
    page = services.get_store().get_page(doc_id, page_number)
    if not page:
        raise HTTPException(404, "page not found")
    page.pop("image_path", None)  # internal path; image served via dedicated route
    return page


@app.get("/api/documents/{doc_id}/pages/{page_number}/image")
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
@app.post("/api/documents/{doc_id}/ask")
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
# static frontend (mounted last so /api/* wins)
# --------------------------------------------------------------------------- #
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
