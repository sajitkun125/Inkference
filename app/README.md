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
│   ├── store/              SQLite store + seed loader (from ../transcriptions)
│   └── api/                FastAPI app + background ingestion jobs
├── frontend/               Inkference UI (index.html / styles.css / app.js)
├── scripts/                prepare_model.py (assemble inference model)
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

## For empty data or seed 
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_fresh \
  uvicorn inkference.api.main:app --port 8000 --log-level debug 2>&1 | tee app/server.log



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

**3. On the Space → Settings → Variables and secrets:**
- Variable `INKFERENCE_IMAGES_BASE_URL` =
  `https://huggingface.co/datasets/sajitkun125/inkference-book-images/resolve/main`
- Secrets `GROQ_API_KEY`, `GEMINI_API_KEY` (persist across redeploys)

The Dockerfile's `SEED_DATASET` must match your dataset (default
`sajitkun125/inkference-book-images`). Redeploy after code/corpus changes: re-run the seed
if the corpus changed, then re-run step 2 (Factory-reboot the Space if a cached build layer
serves an old corpus). Full details: [../projectNotes/deploy_all_books.md](../projectNotes/deploy_all_books.md).

### Alternative: Book-1-only demo (self-contained, base64 images)
```bash
bash app/deploy/deploy_to_hf.sh sajitkun125/inkference
```