#!/usr/bin/env bash
# Redeploy Inkference to a Hugging Face Docker Space.
# Refreshes a staging folder with ONLY the files the image needs (never data/
# models/ notebooks/), then pushes with `hf upload` (uses your `hf auth login`
# token — no git-credential prompt). Re-run any time to push the latest code.
#
# Usage:  bash app/deploy/deploy_to_hf.sh <user>/<space> [stage_dir]
#   e.g.  bash app/deploy/deploy_to_hf.sh sajitkun125/inkference
#         bash app/deploy/deploy_to_hf.sh sajitkun125/inkference ~/hf-inkference
# stage_dir defaults to ~/hf-inkference (reused if it already exists).
set -euo pipefail

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
  echo "usage: bash app/deploy/deploy_to_hf.sh <user>/<space> [stage_dir]"; exit 1
fi
STAGE="${2:-$HOME/hf-inkference}"

# repo root = two levels up from this script (app/deploy/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$STAGE"
echo "Staging Space files in: $STAGE (repo: $REPO)"

# Replace the previously-staged app files (keep any .git so it stays a valid clone).
rm -rf "$STAGE/app" "$STAGE/Dockerfile" "$STAGE/README.md"
cp "$REPO/app/deploy/Dockerfile"                "$STAGE/Dockerfile"     # HF builds root Dockerfile
cp "$REPO/app/deploy/README.md"                 "$STAGE/README.md"      # HF frontmatter
mkdir -p "$STAGE/app/deploy"
cp    "$REPO/app/pyproject.toml"                "$STAGE/app/"
cp -r "$REPO/app/src"                           "$STAGE/app/"
cp -r "$REPO/app/frontend"                      "$STAGE/app/"
cp    "$REPO/app/deploy/requirements-space.txt" "$STAGE/app/deploy/"
cp -r "$REPO/app/deploy/book1_data"             "$STAGE/app/deploy/"

# strip build/cache cruft so it isn't uploaded
find "$STAGE/app" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE/app" -type d -name "*.egg-info"  -prune -exec rm -rf {} + 2>/dev/null || true

echo "Uploading to space: $SPACE"
# Images are shipped as base64 (.jpg.b64) text so HF stores them as regular blobs,
# not Git-LFS pointers (which HF's Docker build does NOT materialize). Delete any
# stale binary .jpg pointers left in the forster1 dir from earlier uploads.
hf upload "$SPACE" "$STAGE" . --repo-type space \
  --exclude ".git/*" \
  --delete "app/deploy/book1_data/book1/forster1/*.jpg"
echo "Done. Build/logs: https://huggingface.co/spaces/$SPACE"