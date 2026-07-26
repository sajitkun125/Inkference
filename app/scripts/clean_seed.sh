#!/usr/bin/env bash
# Clean the local seed corpus: remove live-uploaded pages (rows with an absolute
# image_path), their lines, and stray job rows; delete the assets folder; rebuild the
# FAISS index. Leaves only the seeded pages (relative image keys), so the DB is portable
# and passes the deploy guard.
#
# Uploaded pages store a machine-local absolute path that breaks on the Space, and each
# local test upload re-pollutes the deployable seed — run this before deploying.
#
# Usage:  bash app/scripts/clean_seed.sh [data_root]
#   default data_root: app/.inkference_data_all_books_corrected
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${1:-$REPO/app/.inkference_data_all_books_corrected}"
PY="$REPO/.venv/bin/python"

[ -f "$DATA/inkference.db" ] || { echo "no DB at $DATA"; exit 1; }

# Stop any running server so the DB isn't locked (and can't re-add a page mid-clean).
pkill -9 -f "inkference.api" 2>/dev/null || true
sleep 1

# 1. Delete uploaded (absolute-path) pages + their lines + all job rows.
"$PY" - "$DATA/inkference.db" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1], timeout=10)
ids = [r[0] for r in c.execute("select id from pages where image_path like '/%'")]
for i in ids:
    c.execute("DELETE FROM lines WHERE page_id=?", (i,))
    c.execute("DELETE FROM pages WHERE id=?", (i,))
c.execute("DELETE FROM jobs")
c.commit()
print(f"removed {len(ids)} uploaded page(s); pages now:",
      c.execute("select count(*) from pages").fetchone()[0])
c.close()
PY

# 2. Drop the assets folder (uploaded scan files).
rm -rf "$DATA/assets"

# 3. Rebuild the FAISS index so removed pages' chunks are gone.
INKFERENCE_DATA_ROOT="$DATA" "$PY" -c "
from inkference.rag.index import RagIndex
from inkference.store import DocumentStore
print('index rebuilt:', RagIndex().build_from_store(1, DocumentStore()), 'chunks')
"

echo "seed clean: $DATA"