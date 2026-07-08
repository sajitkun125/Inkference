"""Qwen few-shot page-level post-correction.

Runs after recognition: takes a page's raw TrOCR lines (low-confidence words wrapped
`<<word>>`), primes Qwen with a few in-context (raw -> corrected) exemplars, and asks it
to fix OCR errors while preserving archaic spelling and the line count.

Backends (config.CorrectionConfig.backend):
  local  Qwen3 via transformers (notebook `run_qwen` pattern: /no_think, strip <think>)
  api    hosted Qwen over an OpenAI-compatible endpoint (OpenRouter/DashScope/…)

Heavy imports (torch/transformers/requests) are lazy.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import CorrectionConfig
from ..config import correction as default_correction

if TYPE_CHECKING:  # pragma: no cover
    from ..schemas import Line

_SYSTEM = """\
You are an expert transcriber correcting OCR output of an 18th-century handwritten \
manuscript journal (Johann Reinhold Forster's journal of Captain Cook's second voyage, \
1772–1774: nautical, botanical and zoological terminology).

Your job: fix obvious OCR recognition errors — misread letters, split or merged words, \
stray characters and spacing.

Each input line is prefixed with a number and a pipe, like "3| some text".

Strict rules:
- PRESERVE the historical spelling, capitalisation, abbreviations and punctuation. Do NOT \
modernise (keep forms like "ye", "Tuns", "&", "o'clock").
- Do NOT add, remove, translate, summarise or reorder content.
- Words wrapped in << >> are low-confidence — focus your corrections there, but you may \
also fix clear errors elsewhere.
- Return EVERY input line, each prefixed with its SAME number and a pipe: "3| corrected \
text". Keep the same count and order. Remove the << >> markers. No commentary.\
"""


def _number(lines: list[str]) -> str:
    return "\n".join(f"{i + 1}| {ln}" for i, ln in enumerate(lines))


_NUM_RE = re.compile(r"^\s*(\d+)\s*\|\s?(.*)$")


@dataclass
class CorrectionResult:
    # per-line corrected text aligned 1:1 with the input lines, or None if the model
    # returned a different number of lines (caller falls back to page_text)
    lines: list[str] | None
    page_text: str  # full corrected block as returned
    raw_prompt_lines: list[str] = field(default_factory=list)


def _marked_line(line: "Line") -> str:
    """Rebuild a line's text with low-confidence words wrapped <<word>>."""
    if line.words:
        return " ".join(
            f"<<{w.text}>>" if w.needs_review else w.text for w in line.words
        )
    return line.text


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class PostCorrector:
    def __init__(self, cfg: CorrectionConfig = default_correction) -> None:
        self.cfg = cfg
        self._examples: list[dict] | None = None
        self._tokenizer = None
        self._model = None

    # -- few-shot exemplars ------------------------------------------------- #
    def _load_examples(self) -> list[dict]:
        if self._examples is None:
            try:
                data = json.loads(self.cfg.examples_path.read_text(encoding="utf-8"))
                self._examples = data.get("examples", []) if isinstance(data, dict) else data
            except Exception:
                self._examples = []
        return self._examples[: max(0, self.cfg.num_shots)]

    def build_messages(self, marked_lines: list[str]) -> list[dict]:
        """system + few-shot (raw->corrected) turns + the actual page.

        Exemplars and the target are numbered ("N| text") on the fly so the model
        echoes line numbers — that makes the reply robust to line drift."""
        messages = [{"role": "system", "content": _SYSTEM}]
        for ex in self._load_examples():
            messages.append({"role": "user", "content": _number(ex["raw"].split("\n"))})
            messages.append({"role": "assistant", "content": _number(ex["corrected"].split("\n"))})
        messages.append({"role": "user", "content": _number(marked_lines)})
        return messages

    # -- backends ----------------------------------------------------------- #
    def _ensure_local(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, dtype=dtype)
        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._torch = torch

    def _run_local(self, messages: list[dict]) -> str:
        self._ensure_local()
        torch, tok, model = self._torch, self._tokenizer, self._model
        # Qwen3: disable thinking (append /no_think, like the notebook).
        msgs = list(messages)
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n/no_think"}
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(self._device)
        gen_kwargs = dict(max_new_tokens=self.cfg.max_new_tokens,
                          pad_token_id=tok.eos_token_id)
        if self.cfg.temperature > 0.05:
            gen_kwargs.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return _strip_think(tok.decode(new_tokens, skip_special_tokens=True))

    def _run_api(self, messages: list[dict]) -> str:
        import requests

        # No silent default: fail loudly so the corrector never posts to an
        # unexpected host. Set CORRECTION_API_BASE (e.g. Groq's
        # https://api.groq.com/openai/v1) and a key in the environment/.env.
        base = (self.cfg.api_base or "").rstrip("/")
        if not base:
            raise ValueError(
                "CORRECTION_BACKEND=api but CORRECTION_API_BASE is not set "
                "(e.g. https://api.groq.com/openai/v1)."
            )
        if not self.cfg.api_key:
            raise ValueError(
                "CORRECTION_BACKEND=api but no API key found "
                "(set CORRECTION_API_KEY or GROQ_API_KEY)."
            )
        # Disable Qwen3 "thinking" (soft switch) so reasoning tokens don't eat the
        # completion budget and truncate long pages. Also cap output generously.
        msgs = list(messages)
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n/no_think"}
        payload = {"model": self.cfg.api_model, "messages": msgs,
                   "temperature": self.cfg.temperature,
                   "max_tokens": self.cfg.max_new_tokens}
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        # Retry on 429 (free-tier rate limits) honouring Retry-After, with backoff.
        for attempt in range(4):
            r = requests.post(f"{base}/chat/completions", headers=headers,
                              json=payload, timeout=120)
            if r.status_code == 429 and attempt < 3:
                wait = float(r.headers.get("retry-after", 2 ** attempt * 2))
                time.sleep(min(wait, 30))
                continue
            r.raise_for_status()
            return _strip_think(r.json()["choices"][0]["message"]["content"].strip())
        r.raise_for_status()  # exhausted retries
        return ""

    # -- public API --------------------------------------------------------- #
    def correct_page(self, lines: list["Line"]) -> CorrectionResult:
        marked = [_marked_line(ln) for ln in lines]
        if not marked:
            return CorrectionResult(lines=[], page_text="", raw_prompt_lines=[])
        messages = self.build_messages(marked)
        reply = self._run_api(messages) if self.cfg.backend == "api" else self._run_local(messages)

        # Parse "N| text" replies -> {line_number: corrected_text}. Robust to the
        # model dropping/merging lines: we map by number and fill any gaps from raw.
        parsed: dict[int, str] = {}
        for row in reply.split("\n"):
            m = _NUM_RE.match(row)
            if m:
                parsed[int(m.group(1))] = m.group(2).strip()

        n = len(lines)
        hits = sum(1 for i in range(1, n + 1) if i in parsed)
        if hits < max(1, n // 2):  # too few numbered lines came back — treat as failure
            return CorrectionResult(lines=None, page_text=reply.strip(), raw_prompt_lines=marked)

        # Fill gaps with the original line text (markers stripped) so output is 1:1.
        aligned = [parsed.get(i + 1, lines[i].text) for i in range(n)]
        return CorrectionResult(lines=aligned, page_text="\n".join(aligned), raw_prompt_lines=marked)
