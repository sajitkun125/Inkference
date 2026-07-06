# Deploying the full corpus (all 6 books) to a HF Space

The all-books corpus is ~923 pages and **1.3 GB of scans** — too big for the Space repo.
And HF's **Docker build ships Git-LFS files as pointers**, so large/binary files (scans,
the 34 MB SQLite DB, the FAISS index) can't be committed to the Space repo at all. Solution
(Option A): put everything large in a **public HF dataset** and pull it at build with
`snapshot_download`, which resolves LFS to real bytes.

- **Prebuilt DB + FAISS index (~40 MB)** → dataset `seed_data/`, `snapshot_download`ed at
  build → copied to `/data` on boot → instant cold start, no re-seed/re-embed.
- **Scans (1.3 GB)** → dataset `book*/forster*/`, streamed via a **CDN redirect**
  (`INKFERENCE_IMAGES_BASE_URL`); not downloaded into the image.
- **Live upload** still works (Kraken + TrOCR + Groq correction); uploaded pages save
  absolute paths under `/data` and are served directly (ephemeral).

This is all **free**: public dataset storage, CPU Space, and CDN bandwidth.

## How it fits together
- Seeder stores **relative image keys** (`book<N>/forster<N>/B<N>_P_<nnn>.jpg`).
- Local: `INKFERENCE_IMAGES_ROOT=~/Downloads/AlexFiles` resolves them on disk.
- Space: `INKFERENCE_IMAGES_BASE_URL=https://huggingface.co/datasets/<user>/inkference-book-images/resolve/main`
  redirects them to the CDN.
- Same DB works in both — only the env var differs.

## One-time: build the corpus + upload the images dataset

```bash
cd <repo>
# 1. seed locally (with post-correction) — builds DB + FAISS index
INKFERENCE_DATA_ROOT=$(pwd)/app/.inkference_data_all_books_corrected \
  .venv/bin/python -m inkference.store.seed_all_books --alex ~/Downloads/AlexFiles --with-correction
#   (drop --with-correction for the raw corpus)

# 2. log in + upload the 1.3 GB scans to a PUBLIC dataset (preserves book*/forster*/ layout)
hf auth login
hf repo create inkference-book-images --repo-type dataset            # public
hf upload sajitkun125/inkference-book-images ~/Downloads/AlexFiles . \
  --repo-type dataset --include "book*/forster*/*.jpg"
```

## Deploy the Space

The deploy script uploads the prebuilt DB+index to the dataset (`seed_data/`) AND the app
code to the Space:
```bash
# reuse the existing Space (or create one first time):
# hf repo create inkference --repo-type space --space-sdk docker
bash app/deploy/deploy_all_books.sh sajitkun125/inkference
#   args: <user>/<space> [dataset] [data_root]   (dataset default: sajitkun125/inkference-book-images)
```

Then on the Space (Settings → Variables and secrets):
- **Variable** `INKFERENCE_IMAGES_BASE_URL` =
  `https://huggingface.co/datasets/sajitkun125/inkference-book-images/resolve/main`
- **Secrets** `GROQ_API_KEY`, `GEMINI_API_KEY`

The Dockerfile's `SEED_DATASET` env must match your dataset. HF builds (~4–5 GB image from
the upload/OCR deps; a few minutes), `snapshot_download`s `seed_data/` from the dataset, and
boots **instantly** (copies the DB into `/data`, no re-seed). Scans stream from the CDN.

## Redeploy after code/corpus changes
Re-run the seed (if the corpus changed), then re-run the deploy script. If a cached build
layer serves an old corpus, **Factory reboot** the Space to force a fresh
`snapshot_download`. (To deploy the RAW corpus, pass its data root as the 3rd arg.)

## Files
- `app/src/inkference/store/seed_all_books.py` — all-books seeder (`--with-correction`)
- `app/deploy/Dockerfile.allbooks` — `snapshot_download`s DB+index from the dataset; images via CDN; full pipeline
- `app/deploy/deploy_all_books.sh` — uploads DB+index to the dataset + app code to the Space
- Image resolution: `config.py` (`IMAGES_ROOT`, `IMAGES_BASE_URL`) + `api/main.py` `get_page_image`

## Notes
- Large/binary files must go through the **dataset** (never the Space repo) — HF's Docker
  build materializes dataset downloads but ships Space-repo LFS as pointers.
- Uploaded pages are ephemeral (`/data` resets on restart); the baked corpus always returns.
- The Book-1 base64 Space (`app/deploy/Dockerfile`) is a separate, self-contained deployment.
