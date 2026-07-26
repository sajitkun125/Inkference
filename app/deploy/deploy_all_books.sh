#!/usr/bin/env bash
# Deploy the FULL-corpus (all 6 books) Inkference Space.
#
# Large/binary files can't live in the Space repo (HF's Docker build ships Git-LFS as
# pointers). So both the scans AND the prebuilt DB+index live in a public HF DATASET,
# and the image pulls them with snapshot_download at build (which resolves LFS):
#   * seed_data/inkference.db + seed_data/index  -> downloaded, baked into /data (instant boot)
#   * book*/forster*/*.jpg                       -> streamed via the CDN redirect (not downloaded)
#
# Usage:  bash app/deploy/deploy_all_books.sh <user>/<space> [dataset] [data_root]
#   e.g.  bash app/deploy/deploy_all_books.sh sajitkun125/inkference
set -euo pipefail

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
  echo "usage: bash app/deploy/deploy_all_books.sh <user>/<space> [dataset] [data_root]"; exit 1
fi
DATASET="${2:-sajitkun125/inkference-book-images}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${3:-$REPO/app/.inkference_data_all_books_corrected}"
STAGE="${STAGE:-$HOME/hf-inkference-allbooks}"

[ -f "$DATA/inkference.db" ] || { echo "no DB at $DATA — seed it first"; exit 1; }

# Guard: refuse to deploy a seed polluted by live uploads. Uploaded pages store an
# ABSOLUTE local path (e.g. /home/.../assets/... or /data/assets/...) which does NOT
# exist on the Space, so they'd render broken/wrong scans. Seeded pages use relative
# keys. Abort if any absolute-path page is present so a test upload can't ship.
STRAY="$("$REPO/.venv/bin/python" -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute(\"SELECT count(*) FROM pages WHERE image_path LIKE '/%'\").fetchone()[0]); c.close()" "$DATA/inkference.db")"
if [ "$STRAY" != "0" ]; then
  echo "ABORT: $DATA/inkference.db has $STRAY uploaded (absolute-path) page(s)."
  echo "These are non-portable and would break on the Space. Rebuild the seed cleanly"
  echo "(see app/README.md → 'Rebuild the seed corpus') with the server stopped, then retry."
  exit 1
fi

# 1. Upload the prebuilt corpus (DB + index) to the dataset under seed_data/.
"$REPO/.venv/bin/python" -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()" "$DATA/inkference.db"
SEED="$(mktemp -d)"
cp "$DATA/inkference.db" "$SEED/"
cp -r "$DATA/index" "$SEED/"
echo "Uploading prebuilt corpus (DB + index) to dataset: $DATASET (seed_data/)"
# Use the Python API (upload_folder), NOT the `hf upload` CLI: the CLI calls the
# repo-create endpoint, which now returns 402 for free-tier Docker Spaces. upload_folder
# commits to the existing repo without that call.
"$REPO/.venv/bin/python" - "$SEED" "$DATASET" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
folder, repo = sys.argv[1], sys.argv[2]
try:
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"(create_repo note: {str(e)[:120]})")
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="dataset",
                  path_in_repo="seed_data", commit_message="update prebuilt corpus")
print("dataset upload ok")
PY
rm -rf "$SEED"

# 2. Upload the app code + Dockerfile to the Space (NO large files here).
echo "Staging Space code in: $STAGE"
mkdir -p "$STAGE"
rm -rf "$STAGE/app" "$STAGE/Dockerfile" "$STAGE/README.md"
cp "$REPO/app/deploy/Dockerfile.allbooks" "$STAGE/Dockerfile"     # HF builds root Dockerfile
cp "$REPO/app/deploy/README.md"           "$STAGE/README.md"      # HF frontmatter
mkdir -p "$STAGE/app/deploy"
cp    "$REPO/app/pyproject.toml"                "$STAGE/app/"
cp -r "$REPO/app/src"                           "$STAGE/app/"
cp -r "$REPO/app/frontend"                      "$STAGE/app/"
cp    "$REPO/app/deploy/requirements-space.txt" "$STAGE/app/deploy/"
find "$STAGE/app" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE/app" -type d -name "*.egg-info"  -prune -exec rm -rf {} + 2>/dev/null || true

echo "Uploading app to space: $SPACE"
# Python API (upload_folder), NOT `hf upload`: the CLI's repo-create call returns 402 for
# free-tier Docker Spaces. delete_patterns clears stale files (Book-1 base64 images, old
# seed_data); a commit here auto-triggers the Space rebuild.
"$REPO/.venv/bin/python" - "$STAGE" "$SPACE" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
folder, repo = sys.argv[1], sys.argv[2]
try:
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
except Exception as e:
    # Expected 402 when the Space exists on free-tier Docker; upload_folder still works.
    print(f"(create_repo skipped: {str(e)[:120]})")
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="space",
                  ignore_patterns=[".git/*"],
                  delete_patterns=["app/deploy/book1_data/*", "app/deploy/seed_data/*"],
                  commit_message="deploy app")
print("space upload ok")
PY
echo "Done. Ensure the Dockerfile's SEED_DATASET matches '$DATASET', set the Space"
echo "variable INKFERENCE_IMAGES_BASE_URL + secrets, then watch Logs:"
echo "  https://huggingface.co/spaces/$SPACE"