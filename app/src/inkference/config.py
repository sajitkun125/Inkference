"""Central configuration. All values overridable via environment variables so the
same code runs on a laptop, a free HF Docker Space, or a GPU prod host."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


# Path layout: this file is app/src/inkference/config.py
#   parents[2] = app/        (APP_ROOT — holds frontend/, deploy/, scripts/)
#   parents[3] = repo root   (PROJECT_ROOT — holds transcriptions/, models/, data/)
APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("INKFERENCE_PROJECT_ROOT", APP_ROOT.parent))
# Back-compat alias (older modules referenced REPO_ROOT).
REPO_ROOT = PROJECT_ROOT

# Load .env for local dev (app/.env takes precedence over repo/.env). Real
# environment variables always win — load_dotenv does not override them.
try:
    from dotenv import load_dotenv

    load_dotenv(APP_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

DATA_ROOT = Path(os.getenv("INKFERENCE_DATA_ROOT", APP_ROOT / ".inkference_data"))
FRONTEND_DIR = Path(os.getenv("INKFERENCE_FRONTEND_DIR", APP_ROOT / "frontend"))
TRANSCRIPTIONS_ROOT = Path(
    os.getenv("INKFERENCE_TRANSCRIPTIONS_ROOT", PROJECT_ROOT / "transcriptions")
)
# Base dir for resolving RELATIVE page-image keys stored in the DB (e.g.
# "book1/forster1/B1_P_012.jpg"). Lets one seeded DB stay portable: point this at
# the local source (~/Downloads/AlexFiles) in dev, or the downloaded images dataset
# (/app/book_images) on the Space. Absolute image paths in the DB are used as-is.
_images_root = os.getenv("INKFERENCE_IMAGES_ROOT")
IMAGES_ROOT = Path(_images_root) if _images_root else None
# Alternative to IMAGES_ROOT for deployment: if set, relative image keys are served
# by REDIRECTING to "{IMAGES_BASE_URL}/{key}" (e.g. a public HF dataset resolve URL),
# so large image sets don't need to be baked into the Space image.
IMAGES_BASE_URL = os.getenv("INKFERENCE_IMAGES_BASE_URL") or None


@dataclass
class HTRConfig:
    """Knobs ported from info_files/line_segmentation_output.txt plus runtime ones."""

    # Recognition model: a HF Hub id (prod) or a local checkpoint path (dev).
    trocr_model_id: str = field(
        default_factory=lambda: os.getenv(
            "TROCR_MODEL_ID", "microsoft/trocr-base-handwritten"
        )
    )
    # local | remote  (remote = serverless GPU endpoint in production)
    executor: str = field(default_factory=lambda: os.getenv("HTR_EXECUTOR", "local"))
    device: str = field(default_factory=lambda: os.getenv("HTR_DEVICE", "auto"))

    # Segmentation knobs (from the notebook config cell).
    pad_x: int = 6
    pad_y: int = 2
    mask_to_polygon: bool = True
    poly_pad: int = 2
    min_w: int = 40
    min_h: int = 12
    use_layout_filter: bool = field(
        default_factory=lambda: _env_bool("HTR_USE_LAYOUT_FILTER", False)
    )

    # Free-CPU survival: cap the long edge before segmentation/recognition.
    max_page_long_edge: int = field(
        default_factory=lambda: _env_int("HTR_MAX_LONG_EDGE", 2000)
    )
    num_beams: int = field(default_factory=lambda: _env_int("HTR_NUM_BEAMS", 1))
    max_target_length: int = 128
    recognition_batch_size: int = field(
        default_factory=lambda: _env_int("HTR_BATCH_SIZE", 8)
    )

    # Confidence: words below this are "needs review" (design's <60% flag).
    low_confidence_threshold: float = field(
        default_factory=lambda: _env_float("HTR_LOW_CONF", 0.60)
    )

    def __post_init__(self) -> None:
        # If trocr_model_id names a local folder (absolute, CWD-relative, or
        # PROJECT_ROOT-relative), resolve it to an absolute path so the model
        # loads regardless of working directory. Otherwise leave it as a Hub id.
        mid = self.trocr_model_id
        for candidate in (Path(mid), PROJECT_ROOT / mid):
            if candidate.exists():
                self.trocr_model_id = str(candidate.resolve())
                break


@dataclass
class RAGConfig:
    embed_model_id: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    # Index the post-corrected text (True) or the raw TrOCR text (False). Corrected
    # is cleaner -> better retrieval/answers; pages with no correction fall back to raw.
    use_corrected_text: bool = field(
        default_factory=lambda: _env_bool("RAG_USE_CORRECTED", True)
    )
    # Primary provider for the written answer: gemini | groq | claude | openai
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    # Ordered fallback chain tried when the primary errors/rate-limits, as a
    # comma-separated "provider:model" list. After all fail -> extractive fallback.
    llm_fallback: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK", "gemini:gemini-2.5-flash-lite")
    )

    _PROVIDER_KEYS = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "groq": ("GROQ_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "claude": ("ANTHROPIC_API_KEY",),
    }

    def __post_init__(self) -> None:
        # If no explicit LLM_API_KEY, pull the key for the SELECTED provider so a
        # provider switch (e.g. gemini -> groq) uses the right key automatically.
        if not self.llm_api_key:
            self.llm_api_key = self.key_for(self.llm_provider)

    def key_for(self, provider: str) -> str:
        """Resolve the API key for a provider (used per-attempt in the chain)."""
        provider = (provider or "").lower()
        if self.llm_api_key and provider == (self.llm_provider or "").lower():
            return self.llm_api_key
        for env in self._PROVIDER_KEYS.get(provider, ()):
            if os.getenv(env):
                return os.getenv(env)
        return ""

    def attempts(self) -> list[tuple[str, str]]:
        """Ordered (provider, model) attempts: primary first, then the fallback chain."""
        out: list[tuple[str, str]] = [((self.llm_provider or "").lower(), self.llm_model)]
        for part in self.llm_fallback.split(","):
            part = part.strip()
            if not part:
                continue
            provider, _, model = part.partition(":")
            out.append((provider.strip().lower(), model.strip()))
        return out


@dataclass
class StoreConfig:
    db_path: Path = field(default_factory=lambda: DATA_ROOT / "inkference.db")
    assets_dir: Path = field(default_factory=lambda: DATA_ROOT / "assets")
    index_dir: Path = field(default_factory=lambda: DATA_ROOT / "index")


@dataclass
class CorrectionConfig:
    """Qwen few-shot page-level post-correction (runs after recognition)."""

    enabled: bool = field(default_factory=lambda: _env_bool("CORRECTION_ENABLED", True))
    # backend: local (transformers) | api (OpenAI-compatible hosted Qwen)
    backend: str = field(default_factory=lambda: os.getenv("CORRECTION_BACKEND", "local"))
    # local: a CPU-friendly Qwen3 by default; use Qwen/Qwen3-4B on a GPU.
    model_id: str = field(
        default_factory=lambda: os.getenv("CORRECTION_MODEL_ID", "Qwen/Qwen3-1.7B")
    )
    device: str = field(default_factory=lambda: os.getenv("CORRECTION_DEVICE", "auto"))
    # api backend (hosted Qwen via Groq/OpenRouter/DashScope/etc.)
    api_base: str = field(default_factory=lambda: os.getenv("CORRECTION_API_BASE", ""))
    # falls back to GROQ_API_KEY / OPENROUTER_API_KEY so an existing key just works
    api_key: str = field(
        default_factory=lambda: (
            os.getenv("CORRECTION_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        )
    )
    api_model: str = field(
        default_factory=lambda: os.getenv("CORRECTION_API_MODEL", "qwen/qwen3-32b")
    )
    # few-shot
    num_shots: int = field(default_factory=lambda: _env_int("CORRECTION_NUM_SHOTS", 2))
    examples_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "CORRECTION_EXAMPLES",
                str(Path(__file__).resolve().parent / "htr" / "few_shot_examples.json"),
            )
        )
    )
    max_new_tokens: int = field(
        default_factory=lambda: _env_int("CORRECTION_MAX_NEW_TOKENS", 2048)
    )
    temperature: float = field(
        default_factory=lambda: _env_float("CORRECTION_TEMPERATURE", 0.2)
    )


htr = HTRConfig()
rag = RAGConfig()
store = StoreConfig()
correction = CorrectionConfig()
