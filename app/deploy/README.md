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
runs the full pipeline (Kraken → TrOCR → confidence → Qwen correction). "Ask the
Archive" answers over the corrected text (with page citations), and can also answer
**in character as the author** (Forster).

## What the Space runs
- **Frontend + API**: same URL (the app serves `frontend/` at `/`).
- **Preseeded Book 1**: baked from `app/deploy/book1_data/` (images + confidence +
  corrected/green), re-seeded into the ephemeral `/data` on every boot.
- **Live upload**: Kraken segmentation + TrOCR recognition + per-word confidence +
  Qwen post-correction (Groq). Works on the free 16 GB Space, but CPU-slow
  (~minutes/page) and uploaded pages are lost on restart (ephemeral `/data`).
- **Ask the Archive**: MiniLM + FAISS retrieval → LLM answer + page citations.
  Answer generation uses a **fallback chain**: primary **Groq `openai/gpt-oss-120b`**
  → **Gemini `gemini-2.5-flash-lite`** (when Groq is rate-limited/unavailable) →
  **extractive** passage (always works, cited, $0). The **"Answer as Author"** button
  answers in first person as Forster with an *IN CHARACTER* tag.

## Deploy steps

1. **Log in to Hugging Face**: `hf auth login` (token from
   https://huggingface.co/settings/tokens); confirm with `hf auth whoami`.

2. *(Recommended)* **Push the fine-tuned recognizer to the Hub** so uploads get good OCR
   (otherwise the base model is used):
   ```bash
   hf upload <user>/inkference-trocr models/trocr_best_from_bentham --repo-type model
   ```
   Then set the Space variable `TROCR_MODEL_ID=<user>/inkference-trocr`.

3. **Create a Space** (Docker SDK, free CPU):
   ```bash
   hf repo create inkference --repo-type space --space-sdk docker
   ```

4. **Populate the Space repo** with ONLY what the image needs (never `data/`, `models/`,
   or `notebooks/`). Clone the Space and copy the required files in:
   ```bash
   REPO=$(pwd)                        # this project's root
   git clone https://huggingface.co/spaces/<user>/inkference ~/hf-inkference
   cd ~/hf-inkference
   cp "$REPO/app/deploy/Dockerfile" Dockerfile     # HF builds the ROOT Dockerfile
   cp "$REPO/app/deploy/README.md"  README.md      # HF frontmatter (sdk: docker, app_port)
   mkdir -p app/deploy
   cp    "$REPO/app/pyproject.toml"                app/
   cp -r "$REPO/app/src"                           app/
   cp -r "$REPO/app/frontend"                      app/
   cp    "$REPO/app/deploy/requirements-space.txt" app/deploy/
   cp -r "$REPO/app/deploy/book1_data"             app/deploy/
   ```
   Push with **`hf upload`** (uses your login token — avoids the git-credential prompt
   that makes `git push` hang):
   ```bash
   hf upload <user>/inkference . --repo-type space --exclude ".git/*"
   ```

5. **Secrets** (Space → Settings → Variables and secrets):
   - `GROQ_API_KEY` — post-correction **and** primary Ask-the-Archive answers
   - `GEMINI_API_KEY` — Ask-the-Archive fallback (used when Groq is rate-limited)
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
| `CORRECTION_API_MODEL` | `qwen/qwen3-32b` | Groq correction model |
| `LLM_PROVIDER` / `LLM_MODEL` | `groq` / `openai/gpt-oss-120b` | primary Ask-the-Archive model |
| `LLM_FALLBACK` | `gemini:gemini-2.5-flash-lite` | ordered `provider:model` fallback chain |
| `RAG_USE_CORRECTED` | `true` | index post-corrected text (`false` = raw TrOCR) |
| `GROQ_API_KEY` | – (secret) | correction + primary RAG |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | – (secret) | RAG fallback |
| `INKFERENCE_LOG_LEVEL` | `INFO` | `DEBUG` for per-page/stage + provider logs |
| `INKFERENCE_DATA_ROOT` | `/data` | ephemeral corpus store |
| `CORS_ORIGINS` | `*` | allowed frontend origins |

Key resolution is provider-aware: `LLM_PROVIDER=groq` uses `GROQ_API_KEY`, `=gemini`
uses `GEMINI_API_KEY`/`GOOGLE_API_KEY` — so switching the provider "just works".

## Logs
All app logs use the `inkference.*` loggers (`inkference.api`, `.rag`, `.ingest`,
`.correction`) and print to the container console (visible in the Space **Logs** tab).
API keys are **redacted** from logs, and provider errors are logged server-side only —
never returned to the client. Set `INKFERENCE_LOG_LEVEL=DEBUG` for verbose detail.

## Caveats (free tier)
- **Ephemeral storage**: `/data` resets on restart → Book 1 re-seeds automatically, but
  uploaded pages are lost. For persistence, attach paid persistent storage or a managed DB.
- **CPU speed**: live upload is minutes/page (design assumed a GPU). For production, run
  HTR on a serverless GPU (Modal/Replicate) via a `remote` executor.
- **Sleep**: free Spaces sleep on inactivity (cold start ~30–60 s).
- **LLM free-tier limits**: Groq gpt-oss-120b ≈ 8k tokens/min (plenty for a demo);
  Gemini free tier is stingy — the fallback chain + extractive default keep answers flowing.

## Local run

```bash
pip install -r requirements.txt && pip install -e ./app
python -m inkference.store.seed_book1 --alex ~/Downloads/AlexFiles   # or store.seed for the demo
uvicorn inkference.api.main:app --port 8000 --log-level info
```

See [../projectNotes/running_and_seeds.md](../projectNotes/running_and_seeds.md) for seeds/data-roots
and [../projectNotes/inkference_platform_plan.md](../projectNotes/inkference_platform_plan.md) for the plan.