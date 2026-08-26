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
    # Paired with api_reasoning_effort="none" below — the two only make sense together.
    #
    # qwen3.6 respects the historical spelling this corpus is transcribed in: it fixes
    # "Carpinter" -> "Carpenter" while leaving "repaird", "provd" and "ye scorbutick"
    # alone. Both gpt-oss sizes modernise those, which is wrong for an 18th-century
    # journal, and the 20b additionally leaks few-shot exemplar text into its output.
    #
    # Also deliberately NOT the model RAGConfig uses. Groq meters tokens per model
    # (verified: draining gpt-oss-120b left gpt-oss-20b's budget untouched), so sharing
    # one model would let a bulk ingest eat the same 8000 TPM bucket "Ask the Archive"
    # answers from — degrading it to the Gemini fallback, and only after burning ~40s
    # in _post_retry's backoff first.
    api_model: str = field(
        default_factory=lambda: os.getenv("CORRECTION_API_MODEL", "qwen/qwen3.6-27b")
    )
    # Sent only when non-empty; accepted values are model-specific (qwen3.6: none |
    # default, gpt-oss: low | medium | high). Clear it for a model that rejects it.
    #
    # "none" is the API-level thinking switch, and the only setting that makes qwen3.6
    # usable here — the in-prompt "/no_think" hint below is advisory and this model
    # ignores it. Left thinking-enabled it spends the whole token budget reasoning,
    # gets truncated before the closing </think>, and its monologue (dense with
    # "N| ..." fragments) parses as corrected text. Off, a page costs ~450 tokens
    # instead of ~4000 and the reply is only the answer.
    #
    # The cost: on the longest pages (~49 lines) it sometimes returns every line
    # unchanged rather than correcting. That is a no-op, not a corruption — the lines
    # still align 1:1 — so the page keeps its raw text and can be re-run.
    api_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("CORRECTION_API_REASONING_EFFORT", "none").strip()
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
    # Generous headroom rather than a working limit: with reasoning off a 49-line page
    # spends ~650 completion tokens, so this is never the binding constraint. It is
    # bounded from above by Groq's free tier, which counts `prompt + max_tokens` against
    # an 8000 tokens/minute limit and rejects the request outright (413) when the sum
    # exceeds it — so a bigger cap is not free headroom, it is a request that never runs.
    # A 49-line page prompts at ~1100 tokens, leaving 4096 comfortably inside the budget.
    max_new_tokens: int = field(
        default_factory=lambda: _env_int("CORRECTION_MAX_NEW_TOKENS", 4096)
    )
    temperature: float = field(
        default_factory=lambda: _env_float("CORRECTION_TEMPERATURE", 0.2)
    )


@dataclass
class AgentConfig:
    """LangGraph research agent behind "Ask the Archive" (POST /documents/{id}/agent).

    The agent is an ADDITION: POST /ask stays the one-shot fast path. Budgets exist
    because a run costs 2-7 LLM calls against a free Groq tier on a free CPU Space.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("AGENT_ENABLED", True))

    # -- budgets ------------------------------------------------------------ #
    # Tool calls (search/read/...) per question. Each costs a plan LLM call.
    max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 4))
    max_rewrites: int = field(default_factory=lambda: _env_int("AGENT_MAX_REWRITES", 1))
    max_verify_retries: int = field(
        default_factory=lambda: _env_int("AGENT_MAX_VERIFY_RETRIES", 1)
    )
    # Wall-clock ceiling for one turn, checked in EVERY node so a slow provider
    # can't run away. Below the per-call timeout * max_steps on purpose.
    time_budget_s: float = field(
        default_factory=lambda: _env_float("AGENT_TIME_BUDGET_S", 45.0)
    )
    # Per-LLM-call timeout for agent calls (rag/llm.py defaults to 60s for /ask).
    llm_timeout_s: float = field(
        default_factory=lambda: _env_float("AGENT_LLM_TIMEOUT_S", 30.0)
    )

    # -- retrieval / evidence ----------------------------------------------- #
    search_k: int = field(default_factory=lambda: _env_int("AGENT_SEARCH_K", 6))
    # Cosine floor applied in the TOOL layer only, so /ask keeps its current recall.
    score_floor: float = field(
        default_factory=lambda: _env_float("AGENT_SCORE_FLOOR", 0.25)
    )
    # Truncation caps that keep the prompt inside a free-tier token budget.
    page_chars: int = field(default_factory=lambda: _env_int("AGENT_PAGE_CHARS", 1800))
    evidence_chars: int = field(
        default_factory=lambda: _env_int("AGENT_EVIDENCE_CHARS", 9000)
    )
    # read_range spans at most this many pages and never crosses a book boundary.
    max_span: int = field(default_factory=lambda: _env_int("AGENT_MAX_SPAN", 6))

    # -- conversation memory ------------------------------------------------ #
    # Deliberately NOT inside inkference.db: deploy_all_books.sh copies that file
    # into the PUBLIC HF seed dataset, and conversation history must never ship.
    checkpoint_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("AGENT_CHECKPOINT_PATH", str(DATA_ROOT / "agent_checkpoints.db"))
        )
    )
    thread_ttl_days: int = field(
        default_factory=lambda: _env_int("AGENT_THREAD_TTL_DAYS", 7)
    )
    # Prior turns replayed into the plan/compose prompts.
    history_turns: int = field(default_factory=lambda: _env_int("AGENT_HISTORY_TURNS", 6))


@dataclass
class AuthConfig:
    """Accounts and sessions for the sign-in page.

    Accounts live in PostgreSQL (see DatabaseConfig), deliberately NOT in
    inkference.db. deploy_all_books.sh copies that file into a PUBLIC HF dataset, so
    anything stored there is published — password hashes and session tokens must
    never ship. Same reasoning as AgentConfig's checkpoint_path above.
    """

    # When False the API stays open and the frontend skips the sign-in gate. Set this
    # for a public demo Space, where requiring an account would lock every visitor out.
    required: bool = field(default_factory=lambda: _env_bool("INKFERENCE_AUTH_REQUIRED", True))
    cookie_name: str = field(
        default_factory=lambda: os.getenv("INKFERENCE_AUTH_COOKIE", "inkference_session")
    )
    session_ttl_days: int = field(
        default_factory=lambda: _env_int("INKFERENCE_SESSION_TTL_DAYS", 30)
    )
    min_password_length: int = field(
        default_factory=lambda: _env_int("INKFERENCE_MIN_PASSWORD_LEN", 10)
    )
    # Signup checks the address's domain actually accepts mail — a DNS/MX lookup,
    # which catches the typo domains ("gmial.com") no regex can. Answers are cached
    # per domain and the resolver fails open when it cannot reach a nameserver, so
    # the cost is one lookup per new domain and never a failed signup because our
    # DNS blinked. Off for tests, which must not touch the network.
    email_check_deliverability: bool = field(
        default_factory=lambda: _env_bool("INKFERENCE_EMAIL_CHECK_DELIVERABILITY", True)
    )
    # email_validator's own default is 15s — far too long to hold a signup request
    # open behind a slow resolver.
    email_dns_timeout_seconds: int = field(
        default_factory=lambda: _env_int("INKFERENCE_EMAIL_DNS_TIMEOUT", 5)
    )
    # scrypt cost. n must be a power of two; 2**14 keeps a hash near ~100ms on the
    # free CPU Space, which is slow enough to matter and fast enough to log in.
    scrypt_n: int = field(default_factory=lambda: _env_int("INKFERENCE_SCRYPT_N", 2**14))
    scrypt_r: int = 8
    scrypt_p: int = 1
    # Send the session cookie only over HTTPS. Defaults off so http://localhost works;
    # every real deployment must set INKFERENCE_COOKIE_SECURE=true.
    cookie_secure: bool = field(
        default_factory=lambda: _env_bool("INKFERENCE_COOKIE_SECURE", False)
    )
    # Absolute external origin, e.g.
    #   https://inkference.happysea-1a2b.westeurope.azurecontainerapps.io
    # Behind Container Apps' ingress the app sees plain http on an internal address, so
    # an OAuth redirect URI built from the request would read http://10.0.0.4:8000/… and
    # be rejected by both providers. Set this and the redirect URI is exact; leave it
    # empty and we fall back to the X-Forwarded-* headers, then to the raw request URL.
    public_base_url: str = field(
        default_factory=lambda: os.getenv("INKFERENCE_PUBLIC_BASE_URL", "").strip().rstrip("/")
    )
    # Trust X-Forwarded-Proto/Host only when the app really is behind a proxy that sets
    # them. A client can forge those headers, so honouring them on a directly exposed
    # app would let a caller steer the OAuth redirect at a host they control. Container
    # Apps' ingress always sets them, hence the default.
    trust_proxy_headers: bool = field(
        default_factory=lambda: _env_bool("INKFERENCE_TRUST_PROXY_HEADERS", True)
    )
    # Seconds a half-finished sign-in stays valid — the window between bouncing the
    # user to the provider and their return. Short: it is a round trip, not a session.
    oauth_state_ttl_seconds: int = field(
        default_factory=lambda: _env_int("INKFERENCE_OAUTH_STATE_TTL", 600)
    )
    # HMAC key for the short-lived OAuth state cookie. MUST be set, and set to the
    # same value on every replica: the sign-in that starts on replica A finishes on
    # replica B, and a per-process random key would reject it as forged. Unset, we
    # generate an ephemeral one and log a warning — workable for a single container,
    # broken the moment Container Apps scales past one. Generate with:
    #   python -c "import secrets; print(secrets.token_urlsafe(48))"
    secret_key: str = field(
        default_factory=lambda: os.getenv("INKFERENCE_SECRET_KEY", "").strip()
    )


@dataclass
class DatabaseConfig:
    """PostgreSQL, everywhere. No file-backed fallback, on purpose.

    Accounts moved off the old sqlite3 file when this stopped being a single-process
    demo. Container Apps gives each replica its own container filesystem, so a SQLite
    auth.db would mean every replica held a different set of users — all of them
    discarded on the next revision. Postgres is also what lets a session outlive a
    deploy.

    DATABASE_URL is the single knob. Azure Database for PostgreSQL requires TLS, so
    the deployed URL must carry `?sslmode=require`.
    """

    # The default matches deploy/docker-compose.dev.yml, so a fresh checkout runs after
    # `docker compose -f app/deploy/docker-compose.dev.yml up -d` with nothing exported.
    url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "").strip()
        or "postgresql+psycopg://inkference:inkference@localhost:5432/inkference"
    )
    # Per-replica pool. Azure Postgres Flexible Server on the small burstable tiers caps
    # connections in the low tens and Container Apps may hold several replicas, so keep
    # this modest: (pool_size + max_overflow) * replicas must stay under the server's
    # max_connections, or a scale-out event exhausts it.
    pool_size: int = field(default_factory=lambda: _env_int("DB_POOL_SIZE", 5))
    max_overflow: int = field(default_factory=lambda: _env_int("DB_MAX_OVERFLOW", 5))
    # Azure's gateway drops idle TCP connections at around four minutes, without a FIN
    # the client can see. Recycling inside that window is what prevents the classic
    # "server closed the connection unexpectedly" on the first request after a lull.
    pool_recycle_seconds: int = field(default_factory=lambda: _env_int("DB_POOL_RECYCLE", 180))
    # Cheap liveness check before a pooled connection is handed out: costs a round trip,
    # saves a 500 when a connection died between checkouts.
    pool_pre_ping: bool = field(default_factory=lambda: _env_bool("DB_POOL_PRE_PING", True))
    echo: bool = field(default_factory=lambda: _env_bool("DB_ECHO", False))
    # Statement timeout (ms) applied to every connection. Auth queries are all indexed
    # point lookups; anything still running after seconds is a fault, not slowness.
    statement_timeout_ms: int = field(
        default_factory=lambda: _env_int("DB_STATEMENT_TIMEOUT_MS", 10_000)
    )
    # Run `alembic upgrade head` during startup. Convenient for a single-replica
    # Container App. Turn it OFF once you scale out or migrate from a release job, so N
    # replicas don't race to migrate the same database on deploy.
    migrate_on_startup: bool = field(
        default_factory=lambda: _env_bool("DB_MIGRATE_ON_STARTUP", True)
    )

    @property
    def normalized_url(self) -> str:
        """Force the driver we actually ship.

        Managed Postgres services hand out `postgres://` or bare `postgresql://` URLs,
        and SQLAlchemy maps the latter to psycopg2, which is not installed. Rewriting
        here means a connection string pasted straight from the Azure portal works
        without anyone having to know the driver name.
        """
        url = self.url
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url

    @property
    def safe_url(self) -> str:
        """The URL with the password blanked, for logs and /api/health."""
        url = self.normalized_url
        if "://" not in url or "@" not in url:
            return url
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"


@dataclass(frozen=True)
class OIDCProvider:
    """One OpenID Connect identity provider.

    Google and Microsoft Entra ID differ only in these fields — discovery document,
    credentials, a couple of authorize-time parameters — so the flow is written once
    against this shape instead of once per provider.
    """

    key: str                 # URL segment: /api/auth/oidc/{key}/start
    label: str               # shown in the UI
    discovery_url: str
    client_id: str
    client_secret: str
    scopes: str = "openid email profile"
    # Appended to the authorize URL. `prompt=select_account` on both providers: without
    # it a browser already signed into one account reuses it silently, which makes
    # "wrong account, let me pick another" impossible from inside the app.
    extra_auth_params: tuple[tuple[str, str], ...] = (("prompt", "select_account"),)

    @property
    def enabled(self) -> bool:
        """Both halves or nothing. A client id without a secret still renders a working
        button, then fails at the token exchange — after the user has already been
        bounced to the provider and back."""
        return bool(self.client_id and self.client_secret)


def _google_provider() -> OIDCProvider:
    return OIDCProvider(
        key="google",
        label="Google",
        discovery_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip(),
        client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip(),
    )


def _microsoft_provider() -> OIDCProvider:
    # The tenant decides who may sign in: a tenant GUID locks it to one organisation,
    # "organizations" admits any work/school account, "common" adds personal Microsoft
    # accounts. Defaulting to "organizations" keeps the door narrower than "common" —
    # widen it deliberately, per deployment.
    tenant = os.getenv("MICROSOFT_OAUTH_TENANT", "organizations").strip() or "organizations"
    return OIDCProvider(
        key="microsoft",
        label="Microsoft",
        discovery_url=(
            f"https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration"
        ),
        client_id=os.getenv("MICROSOFT_OAUTH_CLIENT_ID", "").strip(),
        client_secret=os.getenv("MICROSOFT_OAUTH_CLIENT_SECRET", "").strip(),
    )


@dataclass
class OIDCConfig:
    """The provider registry.

    Unconfigured providers stay in the registry but report enabled=False, which renders
    their button greyed out with an explanation rather than hiding it.
    """

    providers: dict[str, OIDCProvider] = field(
        default_factory=lambda: {p.key: p for p in (_google_provider(), _microsoft_provider())}
    )

    def get(self, key: str) -> OIDCProvider | None:
        return self.providers.get(key)

    @property
    def any_enabled(self) -> bool:
        return any(p.enabled for p in self.providers.values())


htr = HTRConfig()
rag = RAGConfig()
store = StoreConfig()
correction = CorrectionConfig()
agent = AgentConfig()
auth = AuthConfig()
database = DatabaseConfig()
oidc = OIDCConfig()
