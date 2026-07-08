# Inkference — Hugging Face Docker Space (CPU) with LIVE UPLOAD.
#
# Ships the full pipeline (Kraken + TrOCR + confidence + Groq correction + Gemini RAG)
# and bakes the 36-page Book 1 corpus (app/deploy/book1_data), seeded on boot.
#
# HF builds the repo-ROOT Dockerfile, so before pushing to the Space:
#     cp app/deploy/Dockerfile Dockerfile
# Build context = repo root.
FROM python:3.11-slim

# System libs for the Pillow / torch / kraken image stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces expect a non-root uid 1000 with a writable home.
RUN useradd -m -u 1000 user
WORKDIR /app

# Python deps first (cached layer). Includes kraken (torch 2.10) + TrOCR + RAG + API.
COPY app/deploy/requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# App code + frontend + baked Book 1 data (36 pages: images + confidence + corrected).
COPY app/pyproject.toml ./
COPY app/src/ ./src/
COPY app/frontend/ ./frontend/
COPY app/deploy/book1_data/ ./book1_data/
RUN pip install --no-cache-dir -e . --no-deps && \
    mkdir -p /data && chown -R user:user /app /data

ENV INKFERENCE_DATA_ROOT=/data \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860 \
    # --- recognition (live upload) ---
    TROCR_MODEL_ID=microsoft/trocr-base-handwritten \
    HTR_MAX_LONG_EDGE=1600 \
    HTR_NUM_BEAMS=1 \
    INKFERENCE_LOG_LEVEL=INFO \
    # --- post-correction via Groq (GROQ_API_KEY is a Space secret) ---
    CORRECTION_ENABLED=true \
    CORRECTION_BACKEND=api \
    CORRECTION_API_BASE=https://api.groq.com/openai/v1 \
    CORRECTION_API_MODEL=qwen/qwen3-32b \
    # --- Ask-the-Archive: Groq gpt-oss-120b -> Gemini fallback -> extractive ---
    LLM_PROVIDER=groq \
    LLM_MODEL=openai/gpt-oss-120b \
    LLM_FALLBACK=gemini:gemini-2.5-flash-lite \
    RAG_USE_CORRECTED=true
# Set as Space secrets (Settings -> Variables and secrets):
#   GROQ_API_KEY   -> post-correction AND primary RAG answers
#   GEMINI_API_KEY -> RAG fallback when Groq is rate-limited/unavailable
# To use your fine-tuned recognizer, push it to the Hub and set TROCR_MODEL_ID=<user>/inkference-trocr.

USER user

# Pre-download models so the first request is fast (runs as 'user' -> HF_HOME matches runtime).
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel as M; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); M.from_pretrained('microsoft/trocr-base-handwritten')"

EXPOSE 7860

# Seed Book 1 into the (ephemeral) data root on boot, then serve API + frontend.
CMD ["sh", "-c", "python -m inkference.store.seed_book1 --alex /app/book1_data || true; uvicorn inkference.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
