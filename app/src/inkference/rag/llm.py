"""LLM answer generation behind a provider-agnostic REST call.

Providers are called directly over HTTP with `requests` (already a dependency) so
no provider SDK needs installing. Supported: gemini, groq, openai, claude. With no
API key configured, falls back to an extractive answer (stitched top snippets) so
the app still works at $0 and offline.
"""
from __future__ import annotations

from ..config import RAGConfig
from ..config import rag as default_rag

_DEFAULT_MODELS = {
    "gemini": "gemini-1.5-flash",
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


def _build_prompt(question: str, contexts: list[tuple[int, str]]) -> str:
    blocks = "\n\n".join(f"[Page {pn}]\n{txt}" for pn, txt in contexts)
    return (
        f"Transcribed excerpts:\n\n{blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above."
    )


def _extractive_fallback(question: str, contexts: list[tuple[int, str]]) -> str:
    if not contexts:
        return "No transcribed text is available to answer this question yet."
    top = contexts[0][1].replace("\n", " ").strip()
    pages = ", ".join(str(pn) for pn, _ in contexts)
    return (
        f"(No language model configured — showing the most relevant transcribed "
        f"passage from page{'s' if ',' in pages else ''} {pages}.)\n\n“{top}”"
    )


def generate_answer(
    question: str, contexts: list[tuple[int, str]], cfg: RAGConfig = default_rag
) -> str:
    """contexts = [(page_number, text), ...] in relevance order."""
    provider = (cfg.llm_provider or "").lower()
    if not cfg.llm_api_key or provider not in _DEFAULT_MODELS:
        return _extractive_fallback(question, contexts)

    model = cfg.llm_model or _DEFAULT_MODELS[provider]
    prompt = _build_prompt(question, contexts)
    try:
        if provider == "gemini":
            return _call_gemini(model, prompt, cfg.llm_api_key)
        if provider in ("groq", "openai"):
            return _call_openai_compatible(provider, model, prompt, cfg.llm_api_key)
        if provider == "claude":
            return _call_claude(model, prompt, cfg.llm_api_key)
    except Exception as exc:  # network/quota/etc — degrade gracefully
        return _extractive_fallback(question, contexts) + f"\n\n[generation error: {exc}]"
    return _extractive_fallback(question, contexts)


# --------------------------------------------------------------------------- #
# provider calls
# --------------------------------------------------------------------------- #
def _call_gemini(model: str, prompt: str, api_key: str) -> str:
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = {
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openai_compatible(provider: str, model: str, prompt: str, api_key: str) -> str:
    import requests

    base = "https://api.groq.com/openai/v1" if provider == "groq" else "https://api.openai.com/v1"
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_claude(model: str, prompt: str, api_key: str) -> str:
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
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()
