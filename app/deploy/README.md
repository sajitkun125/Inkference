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

Reader · Ask the Archive · Upload. One FastAPI container serves the Inkference UI +
API: the 36-page **Book 1** corpus is baked in and seeded on boot, and **live upload**
runs the full pipeline (Kraken → TrOCR → confidence → Groq correction). "Ask the
Archive" answers over the corrected text with Gemini, citing source pages.

## What the Space runs
- **Frontend + API**: same URL (the app serves `frontend/` at `/`).
- **Preseeded Book 1**: baked from `app/deploy/book1_data/` (images + confidence +
  corrected/green), re-seeded into the ephemeral `/data` on every boot.
- **Live upload**: Kraken segmentation + TrOCR recognition + per-word confidence +
  Qwen post-correction (Groq). Works on the free 16 GB Space, but CPU-slow
  (~minutes/page) and uploaded pages are lost on restart (ephemeral `/data`).
- **Ask the Archive**: MiniLM + FAISS retrieval → Gemini answer + page citations,
  with an extractive fallback if the LLM is unavailable.

## Deploy steps

1. **Log in to Hugging Face**: `hf auth login` (token from
   https://huggingface.co/settings/tokens), or use the web UI.

2. *(Recommended)* **Push the fine-tuned recognizer to the Hub** so uploads get good OCR
   (otherwise the base model is used):
   ```bash
   hf upload <user>/inkference-trocr models/trocr_best_from_bentham
   ```
   Then set the Space variable `TROCR_MODEL_ID=<user>/inkference-trocr`.

3. **Create a Space** → SDK **Docker** (free CPU, 16 GB).

4. **Populate the Space repo** with ONLY what the image needs (do NOT push `data/`,
   `models/`, or `notebooks/` — they're huge). From a clean checkout:
   ```bash
   cp app/deploy/Dockerfile Dockerfile          # HF builds the ROOT Dockerfile
   # keep: Dockerfile, app/ (src, frontend, pyproject.toml, deploy/), app/deploy/book1_data/
   git add Dockerfile app/ && git commit -m "Inkference Space" && git push <space-remote> main
   ```

5. **Secrets** (Space → Settings → Variables and secrets):
   - `GEMINI_API_KEY` — Ask-the-Archive answers
   - `GROQ_API_KEY` — post-correction
   - *(optional)* `TROCR_MODEL_ID` — your Hub recognizer

6. HF builds the image (~4–5 GB; a few minutes) and boots: it seeds Book 1, then serves.

Without the keys the app still runs — correction and answers degrade to their fallbacks
(raw OCR / extractive retrieval), still $0.

## Config (env vars / Space variables)

| Var | Default | Purpose |
|---|---|---|
| `TROCR_MODEL_ID` | `microsoft/trocr-base-handwritten` | recognizer (set to your Hub model) |
| `HTR_MAX_LONG_EDGE` | `1600` | downscale cap (speed vs accuracy) |
| `CORRECTION_ENABLED` / `CORRECTION_BACKEND` | `true` / `api` | Qwen correction via Groq |
| `CORRECTION_API_MODEL` | `qwen/qwen3-32b` | Groq model |
| `GROQ_API_KEY` | – (secret) | correction key |
| `LLM_PROVIDER` / `LLM_MODEL` | `gemini` / `gemini-2.5-flash` | Ask-the-Archive |
| `GEMINI_API_KEY` | – (secret) | RAG answer key |
| `INKFERENCE_DATA_ROOT` | `/data` | ephemeral corpus store |

## Caveats (free tier)
- **Ephemeral storage**: `/data` resets on restart → Book 1 re-seeds automatically, but
  uploaded pages are lost. For persistence, attach paid persistent storage or move the
  store to a managed DB.
- **CPU speed**: live upload is minutes/page (design assumed a GPU). For production, run
  HTR on a serverless GPU (Modal/Replicate) via a `remote` executor.
- **Sleep**: free Spaces sleep on inactivity (cold start ~30–60 s).

## Local run

```bash
pip install -r requirements.txt && pip install -e ./app
python -m inkference.store.seed_book1 --alex ~/Downloads/AlexFiles   # or store.seed for the demo
uvicorn inkference.api.main:app --port 8000
```

See [../projectNotes/running_and_seeds.md](../projectNotes/running_and_seeds.md) for seeds/data-roots
and [../projectNotes/inkference_platform_plan.md](../projectNotes/inkference_platform_plan.md) for the plan.
