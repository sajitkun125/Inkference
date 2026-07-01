# Inkference — free Hugging Face Docker Space (CPU).
# Build context = repo ROOT (needs transcriptions/):
#   docker build -f app/deploy/Dockerfile -t inkference .
FROM python:3.11-slim

# System libs needed by Pillow/torch image stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Run as the non-root user HF Spaces expects (uid 1000) with a writable home.
RUN useradd -m -u 1000 user
WORKDIR /app

# Python deps first (cached layer).
COPY app/deploy/requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# App code + seed corpus + frontend (paths are under app/ in the repo).
COPY app/pyproject.toml ./
COPY app/src/ ./src/
COPY app/frontend/ ./frontend/
COPY transcriptions/ ./transcriptions/
RUN pip install --no-cache-dir -e . --no-deps && \
    mkdir -p /data && chown -R user:user /app /data

ENV INKFERENCE_DATA_ROOT=/data \
    INKFERENCE_TRANSCRIPTIONS_ROOT=/app/transcriptions \
    TROCR_MODEL_ID=microsoft/trocr-base-handwritten \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860
# LLM_PROVIDER + LLM_API_KEY are set as Space secrets (gemini/groq/...).
# For the fine-tuned model, set TROCR_MODEL_ID to your HF Hub model id.

USER user

# Pre-download embedding + base recognition models into the runtime cache so the
# first request is fast (runs as 'user' so HF_HOME matches at runtime).
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel as M; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); M.from_pretrained('microsoft/trocr-base-handwritten')"

EXPOSE 7860

# Seed the demo corpus on boot (idempotent), then serve API + frontend.
CMD ["sh", "-c", "python -m inkference.store.seed || true; uvicorn inkference.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
