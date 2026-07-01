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