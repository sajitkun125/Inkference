"""LLM answer generation behind a provider-agnostic REST call.

Providers are called directly over HTTP with `requests` (already a dependency) so
no provider SDK needs installing. Supported: gemini, groq, openai, claude. With no
API key configured, falls back to an extractive answer (stitched top snippets) so
the app still works at $0 and offline.
"""
from __future__ import annotations

import logging
import re
import time

from ..config import RAGConfig
from ..config import rag as default_rag

logger = logging.getLogger("inkference.rag")

# Strip secrets before anything is logged (Gemini puts ?key= in the URL; bearer tokens too).
_SECRET_RE = re.compile(r"(key=)[\w.\-]+|(AIza[\w\-]{20,})|(gsk_[A-Za-z0-9]{20,})|(Bearer\s+\S+)")


def _redact(text: str) -> str:
    return _SECRET_RE.sub("\\1***", str(text))


def _post_retry(url: str, retries: int = 3, **kwargs):
    """POST that retries transient 429/503 (rate limit / overload) with backoff."""
    import requests

    last = None
    for attempt in range(retries):
        last = requests.post(url, **kwargs)
        if last.status_code in (429, 503) and attempt < retries - 1:
            wait = float(last.headers.get("retry-after", 2 ** attempt))
            time.sleep(min(wait, 20))
            continue
        return last
    return last


class LLMUnavailable(RuntimeError):
    """Every provider in the chain failed, or none was configured.

    generate_answer() catches this and degrades to the extractive fallback. Agent
    nodes let it propagate so the graph can decide (usually: stop planning and
    compose with whatever evidence it already has).
    """


_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
}

_SYSTEM = (
    "You are Inkference, an assistant answering questions about a historical "
    "handwritten document using ONLY the provided transcribed page excerpts. "
    "Ground every claim in the excerpts. If the answer is not present, say so. "
    "Do not invent facts. Write 2–4 sentences in a clear, scholarly tone."
)

# In-character persona: answer AS Captain Cook, still grounded in the excerpts.
_SYSTEM_COOK = (
    "You are Johann Reinhold Forster, the naturalist aboard HMS Resolution during "
    "Captain Cook's second voyage and the author of this journal (Books 1-6, "
    "transcribed from your own handwriting). Answer the reader's question in the "
    "first person, as yourself, drawing ONLY on the provided excerpts from your own "
    "journal as your memory of the voyage — do not break character and do not refer "
    "to yourself as an AI or assistant. Write in a reflective, learned 18th-century "
    "voice, but keep the language clear for a modern reader. If your journal "
    "excerpts do not cover the question, say so honestly as yourself rather than "
    "inventing facts. Write 2-4 sentences."
)


def _system_for(persona: str | None) -> str:
    return _SYSTEM_COOK if (persona or "").lower() == "cook" else _SYSTEM


def _build_prompt(
    question: str,
    contexts: list[tuple[int, str]],
    history: list[tuple[str, str]] | None = None,
) -> str:
    blocks = "\n\n".join(f"[Page {pn}]\n{txt}" for pn, txt in contexts)
    prefix = ""
    if history:
        turns = "\n".join(f"{role.capitalize()}: {text}" for role, text in history)
        prefix = (
            "Earlier in this conversation:\n\n"
            f"{turns}\n\n"
            "Use it only to interpret what the new question refers to; ground the "
            "answer itself in the excerpts below.\n\n"
        )
    return (
        f"{prefix}Transcribed excerpts:\n\n{blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above."
    )


def _extractive_fallback(question: str, contexts: list[tuple[int, str]]) -> str:
    if not contexts:
        return "No transcribed text is available to answer this question yet."
    top = contexts[0][1].replace("\n", " ").strip()
    pages = ", ".join(str(pn) for pn, _ in contexts)
    return (
        f"Showing the most relevant transcribed passage from "
        f"page{'s' if ',' in pages else ''} {pages}:\n\n“{top}”"
    )


def _dispatch(
    provider: str, model: str, system: str, prompt: str, key: str,
    max_tokens: int | None, temperature: float, timeout: float,
) -> str:
    if provider == "gemini":
        return _call_gemini(model, system, prompt, key, max_tokens, temperature, timeout)
    if provider in ("groq", "openai"):
        return _call_openai_compatible(
            provider, model, system, prompt, key, max_tokens, temperature, timeout
        )
    if provider == "claude":
        return _call_claude(model, system, prompt, key, max_tokens, temperature, timeout)
    raise ValueError(f"unknown provider: {provider!r}")


def _run_chain(
    system: str,
    prompt: str,
    cfg: RAGConfig = default_rag,
    *,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    timeout: float = 60.0,
    label: str = "RAG",
) -> str:
    """Walk the provider chain (primary -> cfg.attempts() fallbacks) and return the
    first successful completion.

    Raises LLMUnavailable if every configured provider fails or none has a key.
    Callers decide how to degrade: generate_answer() falls back to an extractive
    passage; agent nodes stop planning.
    """
    seen: set[tuple[str, str]] = set()
    tried_any = False
    for provider, model in cfg.attempts():
        if provider not in _DEFAULT_MODELS:
            logger.debug("skip unknown provider %r", provider)
            continue
        model = model or _DEFAULT_MODELS[provider]
        if (provider, model) in seen:
            continue
        seen.add((provider, model))
        key = cfg.key_for(provider)
        if not key:
            logger.debug("skip %s:%s (no API key configured)", provider, model)
            continue
        tried_any = True
        logger.info("%s via %s:%s", label, provider, model)
        try:
            out = _dispatch(
                provider, model, system, prompt, key, max_tokens, temperature, timeout
            )
            logger.info("%s OK via %s:%s (%d chars)", label, provider, model, len(out))
            return out
        except Exception as exc:  # rate limit / network / quota -> try next provider
            # Redacted: never let the API key (in ?key= / bearer) reach the logs.
            logger.warning("%s provider %s:%s failed, falling back: %s",
                           label, provider, model, _redact(exc))

    if tried_any:
        raise LLMUnavailable("all providers failed")
    raise LLMUnavailable("no LLM provider configured")


def complete(
    system: str,
    prompt: str,
    cfg: RAGConfig = default_rag,
    *,
    max_tokens: int = 800,
    temperature: float = 0.1,
    timeout: float = 30.0,
    label: str = "agent",
) -> str:
    """Single completion over the provider chain, for callers that are not answering
    a RAG question (the agent's plan / grade / rewrite nodes).

    Lower temperature than generate_answer because these outputs are parsed, not
    read. Raises LLMUnavailable — there is no extractive fallback for a decision.
    """
    return _run_chain(
        system, prompt, cfg, max_tokens=max_tokens, temperature=temperature,
        timeout=timeout, label=label,
    )


def generate_answer(
    question: str, contexts: list[tuple[int, str]], cfg: RAGConfig = default_rag,
    persona: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """contexts = [(page_number, text), ...] in relevance order.

    Tries the provider chain (primary -> fallbacks from cfg.attempts()); on a
    provider error/rate-limit it moves to the next, and if all fail returns the
    extractive fallback. persona="cook" answers in Captain Cook's voice.
    history = [(role, text), ...] prior turns, used only for coreference."""
    system = _system_for(persona)
    prompt = _build_prompt(question, contexts, history)
    if persona:
        logger.info("RAG answering with persona=%s", persona)
    try:
        return _run_chain(system, prompt, cfg, label="RAG answer")
    except LLMUnavailable as exc:
        # Client never sees provider errors/keys — only the clean extractive passage.
        logger.warning("RAG %s; using extractive fallback", exc)
        return _extractive_fallback(question, contexts)


# --------------------------------------------------------------------------- #
# provider calls
# --------------------------------------------------------------------------- #
def _call_gemini(
    model: str, system: str, prompt: str, api_key: str,
    max_tokens: int | None = None, temperature: float = 0.2, timeout: float = 60.0,
) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    gen: dict = {"temperature": temperature}
    if max_tokens is not None:
        # Thinking models spend reasoning tokens against this cap, so only send it
        # when a caller deliberately wants a short reply (the agent's plan/grade).
        gen["maxOutputTokens"] = max_tokens
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen,
    }
    r = _post_retry(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openai_compatible(
    provider: str, model: str, system: str, prompt: str, api_key: str,
    max_tokens: int | None = None, temperature: float = 0.2, timeout: float = 60.0,
) -> str:
    base = "https://api.groq.com/openai/v1" if provider == "groq" else "https://api.openai.com/v1"
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        # Omitted by default: reasoning models (gpt-oss) bill thinking tokens
        # against this, and a low cap silently truncates the answer.
        body["max_tokens"] = max_tokens
    r = _post_retry(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_claude(
    model: str, system: str, prompt: str, api_key: str,
    max_tokens: int | None = None, temperature: float = 0.2, timeout: float = 60.0,
) -> str:
    # Uses _post_retry like the other providers so a 429 on the Anthropic path
    # backs off instead of failing the whole attempt.
    r = _post_retry(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            # Required by the Anthropic API, so unlike the others it always has a value.
            "max_tokens": max_tokens if max_tokens is not None else 600,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()
