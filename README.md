# Inkference

**Inkference** turns handwritten manuscripts into a searchable, question-answerable
archive. It was built on the six-book manuscript journal of **Johann Reinhold Forster**,
the naturalist on Captain Cook's second voyage (1772–1775) — ~923 pages of 18th-century
handwriting — but the pipeline works on any handwritten scans.

It combines **HTR** (handwritten text recognition) with **RAG** (retrieval-augmented
generation): every page is segmented, transcribed, confidence-scored, LLM-proofread, and
indexed so you can read it side-by-side with the scan or just *ask the archive* a question.

## The three views

| View | What it does |
|---|---|
| **Reader** | Scan next to its transcription; per-word **confidence tinting**, page-average %, and a Raw ⟷ Corrected toggle (green = LLM edits). Page navigation is cyclic. |
| **Ask the Archive** | Ask a natural-language question; get a grounded answer with **source-page citations**, or an in-character **"Answer as Author"** (Forster) response. |
| **Upload** | Drop new scans → live pipeline (**Segmentation → Recognition → Confidence → Correction**) with line-box overlays and a progress stepper. |

## How it works

- **Segmentation** — Kraken `blla` baseline segmentation into line polygons.
- **Recognition** — fine-tuned **TrOCR** (base, Bentham-trained) with **per-word confidence**.
- **Post-correction** — page-level, few-shot LLM proofreading (Qwen via Groq) that fixes OCR
  errors while preserving archaic spelling; corrected words keep their confidence tint.
- **RAG** — `all-MiniLM-L6-v2` embeddings + **FAISS** retrieval → grounded answer from a
  free-tier LLM (Groq `gpt-oss-120b` → Gemini fallback → extractive), with page citations.
- **Serving** — **FastAPI** backend + static frontend; **SQLite** store; background ingest jobs.

The application lives in [`app/`](app/); the original HTR research (notebooks, data, models)
remains at the repo root.

## Run the app locally

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt && pip install -e ./app

# 2. Start the server (reads app/.env for data root, model, and API keys)
uvicorn inkference.api.main:app --port 8000

# 3. Open http://127.0.0.1:8000
```

The seeded corpus (DB + FAISS index) loads on startup; page scans resolve from
`INKFERENCE_IMAGES_ROOT`. To (re)build the corpus or deploy to a free Hugging Face Space,
see [`app/README.md`](app/README.md). Design notes and the build log are in
[`projectNotes/`](projectNotes/).

## Using it

1. **Reader** — flip through pages; toggle Raw/Corrected; hover low-confidence words to see scores.
2. **Ask the Archive** — type a question (e.g. *"What did Forster note about the weather?"*);
   answers cite the pages they came from. Try **Answer as Author** for an in-character reply.
3. **Upload** — drag in a scan; watch the four-stage pipeline; the new page joins the Reader
   and the search index automatically. (On free CPU, a page takes a few minutes.)

## Research notebooks

[`notebooks/`](notebooks/) holds the research that produced the segmentation settings and the
fine-tuned recognizer the app uses. Suggested reading order (it mirrors the pipeline):

1. **Segment** — [`segmentManualTranscriptionsIntoLinesWithKraken.ipynb`](notebooks/segmentManualTranscriptionsIntoLinesWithKraken.ipynb)
   → cut manuscript pages into line images with Kraken.
2. **Organize & label** — [`organizeLinesIntoFinalTranscriptionFolders.ipynb`](notebooks/organizeLinesIntoFinalTranscriptionFolders.ipynb),
   [`writeGroundTruthForB6P060.ipynb`](notebooks/writeGroundTruthForB6P060.ipynb),
   [`visualizeFinalTranscriptionLinesAndGT.ipynb`](notebooks/visualizeFinalTranscriptionLinesAndGT.ipynb)
   → build line/ground-truth pairs and eyeball them.
3. **EDA & modeling** — [`edaAndModelingOcrVersionForGPU.ipynb`](notebooks/edaAndModelingOcrVersionForGPU.ipynb)
   → the core TrOCR training/inference loop.
4. **Cross-dataset training** — `cross_validation*.ipynb`, `downloadBullingerTrainAndTestData.ipynb`,
   `fineTuneTrOcrCheckPointWithReadOrGermanDataSet*.ipynb`, `furtherfinetuningTrOcrWithBulliger*.ipynb`,
   `zeroShotEvalReadCheckpointOnBullingerGerman.ipynb`
   → pre-train/evaluate on public HTR corpora (Bentham, Bullinger, READ/German, Washington).
5. **Target-domain fine-tuning** — [`notebooks/captainCookManualTranscriptsNotebooks/`](notebooks/captainCookManualTranscriptsNotebooks/):
   `zeroShotTrocrBaseUsingGoogleColab` (baseline) → `finetuneTrocrBase`/`finetuneTrocrLarge`/
   `finetuneBenthamCheckpoint` (the winning lineage → `models/trocr_best_from_bentham`) →
   [`compareExperimentResults.ipynb`](notebooks/captainCookManualTranscriptsNotebooks/compareExperimentResults.ipynb)
   → CER/WER comparison that picked the production checkpoint.

**Navigating them:** notebooks ending in `…FromGoogleColabDrive` / `…FromDrive` / `…UsingGoogleColab`
are the **GPU (Google Colab)** versions — open them in Colab with the datasets mounted from Drive;
the plain-named ones run locally. Training notebooks expect the public HTR datasets and a GPU;
the segmentation/organization/visualization notebooks run on CPU against
[`transcriptions/`](transcriptions/) and [`data/`](data/). The extracted, importable version of
this logic lives in [`app/src/inkference/htr/`](app/src/inkference/htr/).

## Tech stack

Python · FastAPI · PyTorch · Transformers (TrOCR) · Kraken · sentence-transformers · FAISS ·
SQLite · Groq / Gemini APIs · vanilla HTML/CSS/JS · Docker (Hugging Face Space).