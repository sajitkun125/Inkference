# Inkference — HTR + RAG Platform: Build & Deployment Plan

> Source of truth for turning the HTR research repo into the **Inkference** product
> (per the Claude design in [../design/Inkference.html](../design/Inkference.html)).
> Build progress is tracked at the bottom.

## Context

We have a mature **HTR research repo** (fine-tuned TrOCR + Kraken/DocLayout-YOLO line
segmentation across Bentham/Bullinger/READ/Washington/GT4Hist, all in notebooks) and a
polished design ("Inkference") describing the product to ship. The design's three views
map almost 1:1 onto existing work:

| Design view | Function | Repo status |
|---|---|---|
| **Upload & Process** | Drop scans → pipeline **Segmentation → Recognition → Confidence**, batch queue, line-box overlays, flag low-confidence lines | Segmentation ✅, Recognition ✅, Confidence = NEW |
| **Reader** | Scan ⟷ transcription split, per-word confidence coloring, avg-confidence %, page nav | TrOCR output ✅, word-confidence + multi-page doc model = NEW |
| **Ask the Archive** | RAG chat over all transcribed pages, answers with **Source page citations** | Entirely NEW |

**The gap = the platform layer**: notebooks → reusable modules, a confidence-scoring
recognizer, a RAG pipeline, a FastAPI backend, the Inkference frontend, deployment.
Goal: working end-to-end on a **free tier** now, with a clear paid **production** path.

### Decisions locked
1. **Frontend** = faithful custom Inkference UI (HTML/CSS + light JS) → FastAPI backend.
2. **RAG generation** = local `sentence-transformers` embeddings + FAISS retrieval +
   **free-tier LLM API (Gemini or Groq)** for the written answer + citations. Cost target $0.
3. **HTR compute** = **live upload must run on free CPU.** Design *around* the slowness
   (base model, image downscaling, async job + polling, progress UI) rather than hide it.

## Free-tier feasibility verdict

**Fully $0 is achievable**, with honest limits:
- **Frontend**: static on Vercel / Netlify / GitHub Pages (free).
- **Backend**: **FastAPI in a Hugging Face *Docker* Space**, free CPU (~2 vCPU, **16 GB
  RAM** — enough for TrOCR-base + Kraken + embeddings together; Streamlit Cloud ~1 GB and
  Render free 512 MB are **not**).
- **RAG**: embeddings + FAISS run locally in that 16 GB; generation calls a free-tier API
  key (Space **secret**).
- **Limits to design for**: free Space **sleeps** (cold start ~30–60 s); **CPU OCR is
  minutes/page** (design assumes ~38 s on A100); Space disk is **ephemeral** → persist the
  doc store / bundle a seed corpus.

**CPU mitigations**: base TrOCR (not large); downscale pages to ~2000 px long edge; batch
line crops (`num_beams=1`, beams only on retry); ingestion as an **async background job**
with `/jobs/{id}` polling (HF proxy times out long requests); bundle one pre-processed seed
doc from `transcriptions/` so Reader/Ask are populated on first load.

## Target architecture (free tier)

```
 Browser ──► Inkference frontend (static, Vercel/Pages)
                 │  REST + polling (fetch)
                 ▼
        FastAPI backend  (HF Docker Space, free CPU 16 GB)
        ├─ POST /documents            create doc, enqueue pages
        ├─ POST /documents/{id}/pages upload scan(s) → background ingest job
        ├─ GET  /jobs/{id}            stage/progress (Segmentation→Recognition→Confidence)
        ├─ GET  /documents/{id}/pages/{n}  scan + lines + words + confidence
        └─ POST /documents/{id}/ask   RAG query → answer + source page citations
                 │
                 ├─ htr/    Kraken+DocLayout-YOLO → TrOCR(+confidence)
                 ├─ rag/    sentence-transformers → FAISS → Gemini/Groq generate
                 └─ store/  SQLite + filesystem (scans, line crops, page text, index)
```

Same code runs in prod; only the **HTR executor** and **infra** swap out.

## Build phases

- **Phase 0** — Repo restructure: notebooks → installable `src/inkference/` package + `pyproject.toml`.
- **Phase 1** — HTR pipeline incl. NEW **Confidence** stage (`generate(output_scores=True)` →
  `compute_transition_scores` → per-word/line probability → review flags).
- **Phase 2** — Strip checkpoint to inference artifacts (weights + processor from base) and
  push to HF Hub; backend pulls by id. Prefer **base**-size model for free CPU.
- **Phase 3** — SQLite store (`documents/pages/lines/words`) + filesystem assets + seed loader.
- **Phase 4** — RAG: chunk page text (carry `page_number`), embed (`all-MiniLM-L6-v2`),
  FAISS on disk, retrieve top-k → free-tier LLM grounded answer + distinct source pages.
- **Phase 5** — FastAPI backend; ingestion as background job feeding the queue/stepper UI.
- **Phase 6** — Inkference frontend: Upload/Reader/Ask views faithful to the design
  (palette `#1f1810`/`#faf6ed`/`#8a3a2f`, Spectral + IBM Plex Sans, confidence gradient,
  source chips).
- **Phase 7** — Free deploy: Docker Space backend (Kraken sys-deps, model from Hub, LLM key
  as secret) + static frontend on Vercel/Pages.
- **Phase 8 (production)** — HTR on serverless GPU (Modal/Replicate, `HTR_EXECUTOR=remote`);
  always-on backend (Render/Railway/Fly/Cloud Run); managed vector DB (Qdrant/pgvector);
  object storage (S3/R2); Redis+RQ/Celery queue; upgrade generation to Claude Haiku/Sonnet;
  auth + multi-tenant library + rate limiting + observability.

## Critical files
- New package: `src/inkference/{htr,rag,store,api}/` + `pyproject.toml`
- Port from: `notebooks/segmentManualTranscriptionsIntoLinesWithKraken.ipynb`,
  `notebooks/edaAndModelingOcrVersionForGPU.ipynb`,
  `notebooks/captainCookManualTranscriptsNotebooks/compareExperimentResults.ipynb`
- Reference (don't edit): `info_files/line_segmentation_output.txt`, `design/Inkference.html`,
  `models/bentham_trocr_checkpoint/`
- Update: `requirements.txt` (pin every new dep), `frontend/`, `Dockerfile`, `.github/workflows/`

## Verification (end-to-end)
1. `pipeline.py` on a known `transcriptions/B*_P*` page → CER/WER within notebook range; confidence populated.
2. Index seed corpus, ask a question whose answer lives on a known page → answer cites that page.
3. `POST` a small scan → poll `/jobs/{id}` through the 3 stages → page shows tinted words → `/ask` returns answer + source chips.
4. All three views render against the live Space; upload→read→ask works on free CPU.
5. Cold-start the Space, confirm the full loop from the deployed static frontend.

## Open follow-ups (non-blocking)
- Naming: design mixes **Inkference** (brand) + **Palimpsest** (assistant voice) +
  `palimpsest.app` — pick the canonical name before publishing.
- Confirm which checkpoint is the production model (best CER from the comparison notebook).

---

## Build progress

- ✅ **Phase 0** — `src/inkference/` package, `pyproject.toml`, `config.py`, `schemas.py`
  (dependency-free dataclasses: `Word`/`Line`/`PageResult`, `Stage`/`JobStatus` enums).
  Installed editable; imports verified.
- ✅ **Phase 1** — `htr/segmentation.py` (KrakenSegmenter + dependency-light
  ProjectionSegmenter fallback, `get_segmenter("auto")`), `htr/recognition.py` (TrOCR +
  per-word/line confidence via `compute_transition_scores`), `htr/pipeline.py` (3-stage
  orchestrator with progress callback). **Verified**: confidence tracks correctness
  (e.g. "Plymouth" 0.998, wrong "directors" 0.14); full synthetic-page run segments →
  recognizes → flags low-confidence lines, all stage events fire.
- ✅ **Phase 2** — `scripts/prepare_model.py`; assembled `models/inkference_trocr_infer/`
  (1.3 GB, weights + processor from base) and verified it loads + beats base on cursive
  ("We weighed anchor **&** left Plymouth" exact). *TODO (manual): `huggingface-cli upload`
  to HF Hub, then set `TROCR_MODEL_ID`.* The training checkpoint has **no processor files**.
- ✅ **Phase 3** — `store/store.py` (SQLite: documents/pages/lines/words/jobs, WAL),
  `store/seed.py` (loads ../transcriptions, stitches line crops into page images). Seeded
  **26 pages / 762 lines**.
- ✅ **Phase 4** — `rag/index.py` (chunk + MiniLM embed + FAISS, persisted), `rag/llm.py`
  (Gemini/Groq/OpenAI/Claude over REST + $0 extractive fallback), `rag/answer.py`. Verified
  semantic retrieval + source-page citations.
- ✅ **Phase 5** — `api/main.py` + `api/services.py` (singletons + single-worker background
  ingest). Verified full upload→poll→read and all read/ask endpoints.
- ✅ **Phase 6** — `frontend/` (Reader / Ask / Upload), faithful to the design; all three
  views screenshot-verified. Hash deep-linking (`#reader|#ask|#upload`).
- 🔄 **Phase 7** — `deploy/Dockerfile`, `deploy/requirements-space.txt` (lean CPU),
  `deploy/README.md` (Space card). **Files written; Docker build not yet tested.** Push to
  HF Space is a manual step.

### Repo layout (post-reorg)
- **All platform code lives under `app/`** (`app/src/inkference`, `app/frontend`,
  `app/deploy`, `app/scripts`, `app/pyproject.toml`). Original research repo (notebooks,
  `data/`, `models/`, `transcriptions/`) untouched at root. `.dockerignore` stays at repo
  root (Docker build context = root). Install with `pip install -e ./app`.
- Config paths: `APP_ROOT`=`app/`, `PROJECT_ROOT`=repo root, `FRONTEND_DIR`=`app/frontend`,
  `TRANSCRIPTIONS_ROOT`=`<repo>/transcriptions`, `DATA_ROOT`=`app/.inkference_data`
  (all env-overridable; Docker sets `INKFERENCE_TRANSCRIPTIONS_ROOT=/app/transcriptions`).

### Segmentation: Kraken-only (projection fallback removed)
- Per user request, `htr/segmentation.py` now uses **Kraken baseline segmentation only**
  (bundled `blla.mlmodel` → baseline + boundary polygon → polygon-masked crop with
  `MaxFilter` dilation), faithful to `segmentManualTranscriptionsIntoLinesWithKraken.ipynb`.
  ProjectionSegmenter is commented out; `get_segmenter` raises if Kraken is missing.
- **Kraken installed in the main venv** → downgraded **torch 2.12 → 2.10** (+ torchvision
  0.25, CUDA/lightning ~3 GB). Verified TrOCR (0.994) + embeddings still work. `requirements.txt`
  torch pins updated to match; `app/deploy/requirements-space.txt` now includes kraken.
- Verified on a page: Kraken found ~31 lines (baseline polygons follow the cursive), TrOCR
  ~4 s/line on CPU. Recognition uses fine-tuned `models/trocr_best_from_bentham` (via `app/.env`).
- ⚠️ **Model swap:** the user replaced the assembled model with `models/trocr_best_from_bentham/`
  (self-contained: weights + tokenizer + processor). `.env` sets `TROCR_MODEL_ID` to it; config
  resolves local model paths against PROJECT_ROOT and loads `.env` (python-dotenv).

### Environment notes
- venv at `.venv/` (Python 3.11). torch **2.10** (CUDA), transformers 5.10, kraken 7.0.2,
  sentence-transformers 5.6, faiss-cpu 1.14, fastapi 0.138. `doclayout-yolo` still not installed
  (layout gate optional/off).
- HF cache has `microsoft/trocr-base-handwritten` and `all-MiniLM-L6-v2` (offline dev works).
- New deps pinned in root `requirements.txt`; runtime set in `app/deploy/requirements-space.txt`.
