"""Guards the one behavioural seam the agent work touched in rag/llm.py.

generate_answer must still swallow a dead provider chain and return the extractive
passage, while complete() must let it propagate so the graph can decide.
"""
from __future__ import annotations

import pytest

from inkference.config import RAGConfig
from inkference.rag import llm


@pytest.fixture
def dead_chain(monkeypatch):
    def boom(*a, **kw):
        raise llm.LLMUnavailable("no LLM provider configured")

    monkeypatch.setattr(llm, "_run_chain", boom)


def test_generate_answer_falls_back_to_extractive(dead_chain):
    out = llm.generate_answer("What happened?", [(14, "They reached Plymouth.")])
    assert "They reached Plymouth." in out
    assert "14" in out


def test_generate_answer_with_no_contexts_is_still_a_sentence(dead_chain):
    assert "No transcribed text" in llm.generate_answer("q", [])


def test_complete_propagates_so_the_agent_can_degrade(dead_chain):
    with pytest.raises(llm.LLMUnavailable):
        llm.complete("sys", "prompt")


def test_run_chain_skips_providers_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_dispatch", lambda *a, **kw: pytest.fail("must not dispatch"))
    cfg = RAGConfig(llm_provider="groq", llm_model="m", llm_api_key="", llm_fallback="")
    with pytest.raises(llm.LLMUnavailable, match="no LLM provider configured"):
        llm._run_chain("sys", "prompt", cfg)


def test_run_chain_falls_through_to_the_next_provider(monkeypatch):
    calls = []

    def flaky(provider, model, system, prompt, key, max_tokens, temperature, timeout):
        calls.append(provider)
        if provider == "groq":
            raise RuntimeError("429 rate limited")
        return "second provider answered"

    monkeypatch.setattr(llm, "_dispatch", flaky)
    cfg = RAGConfig(
        llm_provider="groq", llm_model="m", llm_api_key="k",
        llm_fallback="gemini:gemini-2.5-flash-lite",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "k2")
    assert llm._run_chain("sys", "prompt", cfg) == "second provider answered"
    assert calls == ["groq", "gemini"]


def test_history_is_rendered_into_the_prompt():
    prompt = llm._build_prompt(
        "And the weather?", [(14, "text")],
        history=[("user", "What happened at Plymouth?"), ("assistant", "They arrived.")],
    )
    assert "Earlier in this conversation" in prompt
    assert "What happened at Plymouth?" in prompt


def test_prompt_without_history_is_unchanged():
    assert "Earlier in this conversation" not in llm._build_prompt("q", [(1, "t")])
