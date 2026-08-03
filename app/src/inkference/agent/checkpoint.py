"""Conversation memory: a LangGraph SQLite checkpointer keyed by thread id.

The checkpoint DB is deliberately NOT inkference.db. deploy_all_books.sh copies
inkference.db into the PUBLIC HF seed dataset, and conversation history must never
ship with the corpus. It also keeps langgraph's checkpoint-schema churn away from
the corpus schema, and makes the whole thing safe to delete.
"""
from __future__ import annotations

import logging
import sqlite3
import threading

from ..config import agent as agent_cfg

logger = logging.getLogger("inkference.agent")

_checkpointer = None
_lock = threading.Lock()


def get_checkpointer():
    """Process-wide checkpointer. Degrades to in-memory rather than failing to boot."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    with _lock:
        if _checkpointer is not None:
            return _checkpointer
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            path = agent_cfg.checkpoint_path
            path.parent.mkdir(parents=True, exist_ok=True)
            # One shared connection guarded by langgraph's own lock. This departs
            # from store.py's fresh-connection-per-op convention on purpose:
            # SqliteSaver owns its connection for the process lifetime, and FastAPI
            # runs sync endpoints in a threadpool, hence check_same_thread=False.
            conn = sqlite3.connect(str(path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            _checkpointer = SqliteSaver(conn)
            logger.info("agent: checkpointer=sqlite at %s", path)
        except Exception as exc:  # missing dep, read-only fs, locked db
            from langgraph.checkpoint.memory import MemorySaver

            logger.warning(
                "agent: sqlite checkpointer unavailable (%s); "
                "falling back to in-memory (history lost on restart)", exc,
            )
            _checkpointer = MemorySaver()
        return _checkpointer


def kind() -> str:
    """"sqlite" | "memory" — surfaced in GET /api/health."""
    return type(get_checkpointer()).__name__.replace("Saver", "").lower()


def delete_thread(thread_id: str) -> bool:
    """Forget one conversation ("New conversation" in the UI)."""
    saver = get_checkpointer()
    try:
        saver.delete_thread(thread_id)
        return True
    except Exception as exc:
        logger.warning("agent: could not delete thread %r: %s", thread_id, exc)
        return False
