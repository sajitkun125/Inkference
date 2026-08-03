# Inkference — application

Everything in this `app/` folder was built **on top of** the HTR research repo to
turn it into the Inkference platform (Reader · Ask the Archive · Upload). The
original research assets (notebooks, `data/`, `models/`, `transcriptions/`) remain
at the repo root, untouched.

```
app/
├── pyproject.toml          installable package metadata
├── src/inkference/
│   ├── config.py           env-driven config + path layout
│   ├── schemas.py          domain types (Word / Line / PageResult, enums)
│   ├── htr/                segmentation → recognition (+confidence) → pipeline
│   ├── rag/                chunk + embed + FAISS retrieval + grounded answers
│   ├── agent/              LangGraph research agent over the corpus (see below)
│   ├── store/              SQLite store + seed loader (from ../transcriptions)
│   └── api/                FastAPI app + background ingestion jobs
├── frontend/               Inkference UI (index.html / styles.css / app.js)
├── scripts/                prepare_model.py (assemble inference model)
├── tests/                  offline unit tests (no model loads, no network)
└── deploy/                 Dockerfile + Space README + lean CPU requirements
```

## Run locally (from the repo root)

```bash
pip install -r requirements.txt && pip install -e ./app
python -m inkference.store.seed          # loads the demo corpus
uvicorn inkference.api.main:app --reload # http://127.0.0.1:8000
```

See [deploy/README.md](deploy/README.md) for free Hugging Face Space deployment and
the full environment-variable reference. The build plan and progress log live in
[../projectNotes/inkference_platform_plan.md](../projectNotes/inkference_platform_plan.md).

## Ask the Archive: two modes

**Fast path — `POST /api/documents/{id}/ask`** (unchanged, still the default). One
dense top-k lookup, one LLM call, ~2 s. Right for factoid questions.

**Deep research — `POST /api/documents/{id}/agent`.** A LangGraph agent for the two
things the fast path structurally cannot do:

- **Narrative questions.** The journal is a *diary*. "What happened in the days after
  they reached Plymouth?" needs consecutive pages read in order; five scattered
  600-char chunks cannot answer it. The agent searches to locate the topic, then
  reads page *ranges*.
- **Follow-ups.** `/ask` is stateless, so "and what was the weather like there?" has
  no referent. The agent keeps per-thread conversation memory and rewrites the
  question to stand on its own before retrieving.

```
prepare → plan ─┬─(tool)→ act ─┬─(budget left)→ plan
                │              └─(spent)──────→ compose → END
                └─(answer)────────────────────→ compose → END
```

Tools (`agent/corpus.py`, all backed by existing store/index methods): `search`,
`read_page`, `read_range`, `overview`.

Two design decisions worth knowing:

1. **`rag/llm.py` is still the only model layer.** LangGraph does orchestration only —
   no langchain chat models, no provider SDKs — so the groq → gemini → extractive
   fallback chain keeps working inside the agent.
2. **Tools are called via a JSON text protocol, not provider-native tool calling.**
   The fallback chain switches provider *per call*, and a native tool transcript is a
   provider-specific object graph (OpenAI `tool_calls` vs Gemini `functionCall` vs
   Anthropic `tool_use`). With a text protocol the transcript is just a string, so any
   provider can resume the run at any step. `agent/protocol.py` carries the tolerant
   parser this requires.

With **no API key at all** the agent degrades to a single deterministic retrieval plus
the extractive fallback — i.e. exactly what `/ask` does — rather than erroring.

```bash
# narrative
curl -s localhost:8000/api/documents/1/agent -H 'content-type: application/json' \
  -d '{"question":"What happened in the days after they reached Plymouth?","thread_id":"t1"}'
# follow-up on the same thread — "there" resolves to Plymouth
curl -s localhost:8000/api/documents/1/agent -H 'content-type: application/json' \
  -d '{"question":"And what was the weather like there?","thread_id":"t1"}'
# forget a conversation
curl -s -X DELETE localhost:8000/api/documents/1/agent/threads/t1
```

Budgets (all env-overridable, see `AgentConfig` in `config.py`): `AGENT_MAX_STEPS=4`,
`AGENT_TIME_BUDGET_S=45`, `AGENT_LLM_TIMEOUT_S=30`, `AGENT_EVIDENCE_CHARS=9000`.
A typical run is 2 LLM calls, a narrative one 3–4, worst case 7. `AGENT_ENABLED=false`
turns the endpoints off and hides the UI toggle.

Conversation memory is a LangGraph SQLite checkpointer at
`{INKFERENCE_DATA_ROOT}/agent_checkpoints.db` — deliberately **not** inside
`inkference.db`, which `deploy/deploy_all_books.sh` copies into the *public* HF seed
dataset. On the free Space `/data` is ephemeral, so memory lasts until the Space sleeps.

## Tests

```bash
cd app && ../.venv/bin/python -m pytest tests -q
```

Offline by design — no model loads and no provider calls, so the suite runs in ~2 s.
Covers the action parser (the highest-risk component), the corpus tools against a
fixture store, graph control flow with a scripted planner, and the `rag/llm.py`
fallback behaviour.

## For empty data or seed 
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_fresh \
  uvicorn inkference.api.main:app --port 8000 --log-level debug 2>&1 | tee app/server.log


## Rebuild the seed corpus (all 6 books, with post-correction)

Rebuilds the local DB + FAISS index that the Space deploy ships. The seeder reads from
`~/Downloads/AlexFiles`:
- `moredata/confidence/B<N>_P<NNN>.txt` — raw TrOCR words + per-word confidence
- `moredata/corrected_transcriptions/B<N>_P<NNN>.txt` — post-corrected text (for `--with-correction`)

Page images are **not** copied — the DB stores relative keys (`book<N>/forster<N>/…jpg`)
resolved at serve time against `INKFERENCE_IMAGES_ROOT`.

```bash
# Stop any running server first (the DB is locked while it's up).
# 1. Wipe the old corpus — the seeder skips ("already seeded") if the doc still exists.
rm -f  app/.inkference_data_all_books_corrected/inkference.db \
       app/.inkference_data_all_books_corrected/inkference.db-shm \
       app/.inkference_data_all_books_corrected/inkference.db-wal
rm -rf app/.inkference_data_all_books_corrected/index

# 2. Re-seed all 6 books WITH post-correction (~923 pages; a few minutes).
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_all_books_corrected \
  .venv/bin/python -m inkference.store.seed_all_books --alex ~/Downloads/AlexFiles --with-correction
```

Drop `--with-correction` for the raw-only corpus (slug `all-books-without-post-correction`).
Any live-uploaded pages in that data root are discarded by the rebuild (they only ever
lived in the DB). After rebuilding, redeploy with step 2 below to push the new DB+index.


## Deploy to Hugging Face Space (full corpus, all 6 books)

Reuses the existing Space `sajitkun125/inkference`. Everything large lives in a **public
HF dataset** (HF's Docker build ships Git-LFS as pointers, so large/binary files can't go
in the Space repo). The image pulls them at build with `snapshot_download` (which resolves
LFS):
- **scans (1.3 GB)** → streamed from the dataset CDN (not baked)
- **prebuilt DB + FAISS index (~40 MB)** → downloaded at build → instant boot, no re-seed

**1. Upload the scans once (public dataset):**
```bash
hf auth whoami                                          # ensure logged in (else: hf auth login)
hf repo create inkference-book-images --repo-type dataset
hf upload sajitkun125/inkference-book-images ~/Downloads/AlexFiles . \
  --repo-type dataset --include "book*/forster*/*.jpg"
```

**2. Deploy** — uploads the prebuilt DB+index to the dataset (`seed_data/`) and the app
code to the Space (removes stale Book-1 files):
```bash
bash app/deploy/deploy_all_books.sh sajitkun125/inkference
```
What this does, step by step:
1. **Checkpoints the SQLite WAL** into `inkference.db` so the uploaded DB is self-contained.
2. **Uploads DB + FAISS index** to the dataset under `seed_data/` (the ~923-page corpus).
3. **Uploads the app code** (Dockerfile, `src/`, `frontend/`, requirements) to the Space,
   `--delete`-ing stale files. This push triggers an automatic Space rebuild, during which
   the image `snapshot_download`s the fresh `seed_data/` and copies it into `/data` on boot.

The script **aborts before uploading** if the DB contains any live-uploaded pages (rows with
an absolute `image_path`). Uploaded pages save a machine-local path (`/data/assets/…`) that
doesn't exist on the Space, so shipping them would render broken scans — rebuild the seed
cleanly (above, server stopped) if you hit this.

**3. On the Space → Settings → Variables and secrets** (only needed once; these persist):
- Variable `INKFERENCE_IMAGES_BASE_URL` =
  `https://huggingface.co/datasets/sajitkun125/inkference-book-images/resolve/main`
- Secrets `GROQ_API_KEY`, `GEMINI_API_KEY`

**4. Factory reboot the Space** (Settings → "Factory reboot"). **Required to serve a fresh
corpus and drop pages uploaded on the Space.** `/data` is ephemeral but the boot copy uses
`cp -rn` (no-clobber), so it won't overwrite an existing `/data/inkference.db`. A Factory
reboot wipes `/data` and bypasses the cached build layer, so the container starts empty and
loads *only* the fresh 923-page seed — any pages users uploaded on the Space are discarded.
(A plain restart keeps whatever is already in `/data`.)

The Dockerfile's `SEED_DATASET` must match your dataset (default
`sajitkun125/inkference-book-images`). Redeploy after code/corpus changes: rebuild the seed
if the corpus changed (see above), re-run step 2, then Factory-reboot (step 4). Full
details: [../projectNotes/deploy_all_books.md](../projectNotes/deploy_all_books.md).

### Alternative: Book-1-only demo (self-contained, base64 images)
```bash
bash app/deploy/deploy_to_hf.sh sajitkun125/inkference
```