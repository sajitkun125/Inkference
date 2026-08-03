"""The JSON action protocol the agent uses instead of provider-native tool calling.

WHY NOT NATIVE TOOL CALLING: RAGConfig.attempts() falls back PER CALL (groq ->
gemini -> extractive). An agent run makes several sequential calls carrying an
accumulating transcript. With native tools that transcript is a provider-specific
object graph — OpenAI `tool_calls`, Gemini `functionCall`, Anthropic `tool_use`
blocks — so a mid-run fallback would mean translating an in-flight tool history
between three schemas, on exactly the rate-limit path that is hardest to test.
Here the transcript is just text, so any provider can resume at any step.

The cost is that we must parse model prose robustly. Everything below exists to
make that never raise: a lenient extractor, then a validator, then a coercion to
`{"action": "answer"}` so a bad reply ends the turn gracefully instead of 500ing.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("inkference.agent")

# Keep this tiny. Every extra action is another thing a small model gets wrong.
ACTIONS: dict[str, dict[str, Any]] = {
    "search": {
        "args": {"query": str, "k": int},
        "required": ("query",),
        "doc": 'Semantic search over the whole journal. {"action":"search","query":"arrival at Plymouth","k":6}',
    },
    "read_page": {
        "args": {"page": int},
        "required": ("page",),
        "doc": 'Read one full page by its corpus page number. {"action":"read_page","page":118}',
    },
    "read_range": {
        "args": {"start": int, "end": int},
        "required": ("start", "end"),
        "doc": (
            'Read consecutive pages in order — use this for "what happened next" '
            'questions. {"action":"read_range","start":118,"end":123}'
        ),
    },
    "overview": {
        "args": {},
        "required": (),
        "doc": 'Document title, page count, and which page range each book covers. {"action":"overview"}',
    },
    "answer": {
        "args": {"why": str},
        "required": (),
        "doc": 'Stop searching and write the answer from the evidence gathered. {"action":"answer"}',
    },
}

TERMINAL = "answer"

REPAIR_PROMPT = (
    "Your last reply was not a valid action. Reply with ONLY a single JSON object "
    "and nothing else — no explanation, no markdown fence."
)


def render_tool_catalog(actions: list[str] | None = None) -> str:
    """The action menu injected into the plan system prompt."""
    names = actions or list(ACTIONS)
    return "\n".join(f"- {ACTIONS[n]['doc']}" for n in names if n in ACTIONS)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def _first_balanced_object(text: str) -> str | None:
    """First {...} with balanced braces, ignoring braces inside strings.

    Cheaper and more predictable than a regex for nested objects, and it survives
    the common failure of a model wrapping JSON in a sentence.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


# Repairs for the JSON errors small models actually make, applied in order and
# cumulatively. Each one was added in response to a real observed failure — resist
# adding speculative ones, since a repair that "fixes" valid JSON is worse than a
# parse failure (which costs one retry, not a wrong action).
_REPAIRS = (
    # Trailing comma before a closer: {"a":1,}
    (re.compile(r",\s*([}\]])"), r"\1"),
    # Stray quote after a number: {"end":26"}  — observed from gpt-oss-120b.
    (re.compile(r'(:\s*-?\d+(?:\.\d+)?)"+\s*(?=[,}\]])'), r"\1"),
)


def _loads_lenient(blob: str) -> dict | None:
    candidate = blob
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        obj = None
        for pattern, repl in _REPAIRS:
            candidate = pattern.sub(repl, candidate)
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if obj is None:
            return None
    return obj if isinstance(obj, dict) else None


def extract_json(text: str) -> dict | None:
    """Pull one JSON object out of a model reply. None if there isn't one."""
    if not text:
        return None
    fenced = _FENCE_RE.search(text)
    for candidate in (fenced.group(1) if fenced else None, text.strip(),
                      _first_balanced_object(text)):
        if candidate and (obj := _loads_lenient(candidate)) is not None:
            return obj
    return None


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return int(m.group())
    return None


def validate(obj: dict) -> dict:
    """Normalise a parsed object into a known action.

    Never raises: an unknown action or missing required arg degrades to `answer`,
    which lets the graph compose from whatever evidence it already holds.
    """
    name = str(obj.get("action", "")).strip().lower()
    if name not in ACTIONS:
        logger.debug("agent: unknown action %r -> answer", name)
        return {"action": TERMINAL, "why": f"unrecognised action {name!r}"}

    spec = ACTIONS[name]
    out: dict[str, Any] = {"action": name}
    for arg, typ in spec["args"].items():
        if arg not in obj or obj[arg] is None:
            continue
        if typ is int:
            coerced = _coerce_int(obj[arg])
            if coerced is not None:
                out[arg] = coerced
        else:
            out[arg] = str(obj[arg]).strip()

    missing = [a for a in spec["required"] if a not in out]
    if missing:
        logger.debug("agent: action %r missing %s -> answer", name, missing)
        return {"action": TERMINAL, "why": f"action {name!r} missing {missing}"}
    return out


def parse_action(text: str) -> tuple[dict | None, dict | None]:
    """Parse a plan reply.

    Returns (action, extras). `action` is None only when nothing JSON-shaped was
    found — the caller may then spend one repair retry. `extras` carries non-action
    keys the plan prompt asks for on the first turn (notably standalone_question).
    """
    obj = extract_json(text)
    if obj is None:
        return None, None
    action = validate(obj)
    extras = {}
    if isinstance(obj.get("standalone_question"), str):
        sq = obj["standalone_question"].strip()
        if sq:
            extras["standalone_question"] = sq
    return action, extras
