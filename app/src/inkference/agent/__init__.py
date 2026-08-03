"""LangGraph research agent behind "Ask the Archive".

The one-shot RAG turn (rag/answer.py) stays the fast path. This package adds a
tool-using loop for questions it cannot serve: narrative questions that need
adjacent pages read in order, and follow-ups that need conversation memory.

LangGraph does ORCHESTRATION ONLY. rag/llm.py remains the single model layer, so
the multi-provider fallback chain (groq -> gemini -> extractive) keeps working —
see protocol.py for why tools are called via a JSON text protocol rather than
provider-native tool calling.
"""
from .state import AgentAnswer

__all__ = ["AgentAnswer", "run_agent"]


def run_agent(*args, **kwargs):
    """Lazy re-export: importing runner pulls in langgraph, which the HTR-only
    entry points (seeders, scripts) must not need."""
    from .runner import run_agent as _run

    return _run(*args, **kwargs)
