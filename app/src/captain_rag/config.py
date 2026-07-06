"""Central configuration for the RAG Q&A chat. Everything overridable via env vars,
mirroring the pattern in data/Capstone-Project-inkferenceApp/app/src/inkference/config.py."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Path layout: this file is app/src/captain_rag/config.py
#   parents[2] = app/        (APP_ROOT)
#   parents[3] = repo root   (PROJECT_ROOT — holds data/transcriptions)
APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("CAPTAIN_RAG_PROJECT_ROOT", APP_ROOT.parent))

try:
    from dotenv import load_dotenv

    load_dotenv(APP_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

TRANSCRIPTIONS_ROOT = Path(
    os.getenv("CAPTAIN_RAG_TRANSCRIPTIONS_ROOT", PROJECT_ROOT / "data" / "transcriptions")
)
INDEX_DIR = Path(os.getenv("CAPTAIN_RAG_INDEX_DIR", APP_ROOT / ".rag_index"))


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


@dataclass
class RAGConfig:
    embed_model_id: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    # Provider for the written answer: gemini | groq | claude | openai | mistral
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))

    _PROVIDER_KEYS = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "groq": ("GROQ_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "claude": ("ANTHROPIC_API_KEY",),
        "mistral": ("MISTRAL_API_KEY",),
    }

    def __post_init__(self) -> None:
        # If no explicit LLM_API_KEY, pull the key for the SELECTED provider so a
        # provider switch (e.g. gemini -> groq) uses the right key automatically.
        if not self.llm_api_key:
            for env in self._PROVIDER_KEYS.get((self.llm_provider or "").lower(), ()):
                if os.getenv(env):
                    self.llm_api_key = os.getenv(env)
                    break


rag = RAGConfig()
