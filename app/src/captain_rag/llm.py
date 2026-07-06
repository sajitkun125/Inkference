"""LLM answer generation behind a provider-agnostic REST call.

Providers are called directly over HTTP with `requests` so no provider SDK needs
installing. Supported: gemini, groq, openai, claude, mistral. With no API key
configured, falls back to an extractive answer (stitched top snippet) so the chat
still works at $0 and offline.
"""
from __future__ import annotations

import time

from .config import RAGConfig
from .config import rag as default_rag


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


_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
    "mistral": "mistral-small-latest",
}

_SYSTEM = (
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


_GROUP_SUMMARY_SYSTEM = (
    "You are Johann Reinhold Forster, the naturalist aboard HMS Resolution during "
    "Captain Cook's second voyage and the author of this journal (Books 1-6, "
    "transcribed from your own handwriting). You will be given excerpts sampled "
    "evenly across one group of pages from your journal (e.g. a whole book, or "
    "every page mentioning a given month/season). Summarize what that group has "
    "in common — its main events and themes — in EXACTLY one sentence, in the "
    "first person, as yourself, do not break character. Base it only on the "
    "excerpts given."
)


def _build_prompt(question: str, contexts: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"[{page_id}]\n{txt}" for page_id, txt in contexts)
    return (
        f"Excerpts from your journal:\n\n{blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer in character, using only the excerpts above."
    )


def _build_group_prompt(label: str, contexts: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"[{page_id}]\n{txt}" for page_id, txt in contexts)
    return (
        f"Excerpts sampled across '{label}' in your journal:\n\n{blocks}\n\n"
        f"Summarize '{label}' in exactly one sentence, in character."
    )


def _extractive_fallback(question: str, contexts: list[tuple[str, str]]) -> str:
    if not contexts:
        return "No transcribed text is available to answer this question yet."
    top = contexts[0][1].replace("\n", " ").strip()
    pages = ", ".join(pid for pid, _ in contexts)
    return (
        f"(No language model configured — showing the most relevant transcribed "
        f"passage from {pages}.)\n\n“{top}”"
    )


def _generate(system: str, prompt: str, cfg: RAGConfig) -> str | None:
    """Returns None if no provider/key is configured — caller decides the fallback."""
    provider = (cfg.llm_provider or "").lower()
    if not cfg.llm_api_key or provider not in _DEFAULT_MODELS:
        return None
    model = cfg.llm_model or _DEFAULT_MODELS[provider]
    if provider == "gemini":
        return _call_gemini(model, system, prompt, cfg.llm_api_key)
    if provider in ("groq", "openai", "mistral"):
        return _call_openai_compatible(provider, model, system, prompt, cfg.llm_api_key)
    if provider == "claude":
        return _call_claude(model, system, prompt, cfg.llm_api_key)
    return None


def generate_answer(
    question: str, contexts: list[tuple[str, str]], cfg: RAGConfig = default_rag
) -> str:
    """contexts = [(page_id, text), ...] in relevance order."""
    prompt = _build_prompt(question, contexts)
    try:
        text = _generate(_SYSTEM, prompt, cfg)
    except Exception as exc:  # network/quota/etc — degrade gracefully
        return _extractive_fallback(question, contexts) + f"\n\n[generation error: {exc}]"
    return text if text is not None else _extractive_fallback(question, contexts)


def generate_group_summary(
    label: str, page_excerpts: list[tuple[str, str]], cfg: RAGConfig = default_rag
) -> str:
    """page_excerpts = [(page_id, text), ...] sampled evenly across the group
    (e.g. a book, or every page tagged with a given month/season)."""
    prompt = _build_group_prompt(label, page_excerpts)
    try:
        text = _generate(_GROUP_SUMMARY_SYSTEM, prompt, cfg)
    except Exception as exc:
        text = None
        error = str(exc)
    else:
        error = None
    if text is not None:
        return text
    span = f"{page_excerpts[0][0]}–{page_excerpts[-1][0]}" if page_excerpts else "no pages"
    fallback = f"(No language model configured — '{label}' sampled from {span}.)"
    return fallback if error is None else fallback + f"\n[generation error: {error}]"


# --------------------------------------------------------------------------- #
# provider calls
# --------------------------------------------------------------------------- #
def _call_gemini(model: str, system: str, prompt: str, api_key: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    r = _post_retry(url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


_OPENAI_COMPATIBLE_BASES = {
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openai": "https://api.openai.com/v1",
}


def _call_openai_compatible(provider: str, model: str, system: str, prompt: str, api_key: str) -> str:
    base = _OPENAI_COMPATIBLE_BASES[provider]
    r = _post_retry(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_claude(model: str, system: str, prompt: str, api_key: str) -> str:
    import requests

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 600,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()
