# Running the app & loading seeds

The corpus the app serves is controlled by **`INKFERENCE_DATA_ROOT`** — the folder
holding that corpus's SQLite DB (`inkference.db`), page images (`assets/`), and FAISS
index (`index/`). A **seeder** writes a corpus into that folder; the **app** reads from
the same folder. Run everything from the repo root.

## Available seeders

| Corpus | Module | Notes |
|---|---|---|
| **Book 1 — Alex's real data (36 pages)** | `inkference.store.seed_book1` | page images + confidence + corrected(green) + RAG; `--alex <AlexFiles dir>` |
| **Captain Cook demo (stitched line crops)** | `inkference.store.seed` | ground-truth demo built from `transcriptions/` |

## Pattern: seed a fresh corpus, then run it

```bash
cd .../Capstone-Project

# 1) choose a data root and seed into it (rm first for a clean re-seed)
rm -rf app/.inkference_data_book1
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_book1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m inkference.store.seed_book1 --alex ~/Downloads/AlexFiles

# 2) run the app on that same data root
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_book1 \
  .venv/bin/uvicorn inkference.api.main:app --port 8000
# open http://127.0.0.1:8000/
```

Swap the folder + seeder to load a different corpus, e.g. the Captain Cook demo:

```bash
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_demo .venv/bin/python -m inkference.store.seed
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_demo .venv/bin/uvicorn inkference.api.main:app --port 8000
```

## Notes

- **`app/.env` pins `INKFERENCE_DATA_ROOT`** (currently the Book 1 folder), so a plain
  `uvicorn inkference.api.main:app` uses that. To change the default permanently, edit that
  line in `app/.env`; to override once, pass `INKFERENCE_DATA_ROOT=…` inline — a real env
  var always wins over `.env`.
- **Seeds are idempotent**: they skip if that corpus's document already exists in the data
  root. For a clean rebuild, `rm -rf` the data folder first.
- One data root **can hold multiple documents** (run both seeders into the same folder), but
  the Reader opens the most-recent one (no document picker yet).
- `.venv/bin/` prefix is unnecessary if the venv is activated.
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` forces the cached embedding/recognition models
  (drop them to allow downloads).

## Stop the app

```bash
pkill -f "uvicorn inkference"
```

## Related env vars (set in `app/.env`)

| Var | Purpose |
|---|---|
| `INKFERENCE_DATA_ROOT` | active corpus folder (DB + assets + index) |
| `TROCR_MODEL_ID` | recognition model (local path or HF id) |
| `LLM_PROVIDER` / `LLM_MODEL` | Ask-the-Archive generation (e.g. `gemini` / `gemini-2.5-flash`) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini key (resolved when `LLM_PROVIDER=gemini`) |
| `CORRECTION_ENABLED` / `CORRECTION_BACKEND` / `CORRECTION_API_MODEL` | Qwen post-correction (Groq) |
| `GROQ_API_KEY` | Groq key (correction, and RAG when `LLM_PROVIDER=groq`) |

See also: [inkference_platform_plan.md](inkference_platform_plan.md) (build plan) and
[qwen_post_correction.md](qwen_post_correction.md) (post-correction design).
