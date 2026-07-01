---
title: Inkference
emoji: 🪶
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Inkference — HTR + RAG for historical handwriting

Reader · Ask the Archive · Upload. A FastAPI backend (TrOCR recognition with
per-word confidence + FAISS retrieval) serving the Inkference frontend, shipped
with a seeded Captain Cook voyage corpus.

## Deploying to a free Hugging Face Docker Space

All platform code lives under `app/`; the original research repo (notebooks, data,
models, transcriptions) stays at the repo root.

1. **Create a Space** → SDK **Docker** (free CPU, 16 GB RAM).
2. Push this repo to the Space. HF builds the root `Dockerfile`, so copy ours up
   first (it expects the repo root as build context, which it gets on a Space):
   `cp app/deploy/Dockerfile Dockerfile` before pushing.
3. **Secrets** (Space → Settings → Variables and secrets):
   - `LLM_PROVIDER` = `gemini` (or `groq`)
   - `LLM_API_KEY` = your free-tier key
   - *(optional)* `TROCR_MODEL_ID` = `<your-user>/inkference-trocr` for the
     fine-tuned model (otherwise the base model is used).
4. The container seeds the demo corpus on boot, then serves the UI at the Space URL.

Without `LLM_API_KEY` the app still runs — answers fall back to extractive
snippets (still $0, still cited).

## Split hosting (frontend on Vercel/Pages, backend on the Space)

Host `app/frontend/` statically and point it at the Space backend by injecting,
before `app.js`:

```html
<script>window.INKFERENCE_API = "https://<your-space>.hf.space";</script>
```

Set `CORS_ORIGINS` on the Space to your frontend origin.

## Local run

```bash
# from the repo root
pip install -r requirements.txt && pip install -e ./app
python -m inkference.store.seed
uvicorn inkference.api.main:app --reload
# open http://127.0.0.1:8000
```

## Config (environment variables)

| Var | Default | Purpose |
|---|---|---|
| `TROCR_MODEL_ID` | `microsoft/trocr-base-handwritten` | recognition model (HF id or local path) |
| `LLM_PROVIDER` / `LLM_API_KEY` | – | answer generation (gemini/groq/openai/claude) |
| `HTR_EXECUTOR` | `local` | `remote` → serverless GPU (production) |
| `HTR_MAX_LONG_EDGE` | `2000` | downscale cap for free-CPU survival |
| `INKFERENCE_DATA_ROOT` | `.inkference_data` | SQLite + assets + FAISS index |
| `CORS_ORIGINS` | `*` | allowed frontend origins |
