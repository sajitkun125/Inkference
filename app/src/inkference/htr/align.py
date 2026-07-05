"""Align corrected words back to the original TrOCR words' confidence.

Ported from post_correction_full_test_v2.ipynb (`align_words`/`_normalise`). Page-level
correction rewrites text, so to keep the per-word confidence tint in the Reader we map
each corrected word to the original word(s) it came from via difflib.SequenceMatcher:

  equal   -> keep the original word's confidence
  replace -> average confidence of the replaced originals, flagged qwen_replaced
  insert  -> a word Qwen added with no original -> default confidence, qwen_replaced
  delete  -> original dropped -> skipped

Returns a fresh list[Word] for the corrected line.
"""
from __future__ import annotations

import difflib
import re

from ..schemas import Word

# Confidence for words Qwen inserted that have no original counterpart.
DEFAULT_INSERT_CONF = 0.5


def _normalise(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def align_line(
    original_words: list[Word],
    corrected_text: str,
    low_conf_threshold: float = 0.60,
) -> list[Word]:
    """Map the corrected line's words onto the original words' confidences."""
    corr_tokens = corrected_text.split()
    if not corr_tokens:
        return []
    if not original_words:
        return [
            Word(text=t, confidence=DEFAULT_INSERT_CONF,
                 needs_review=DEFAULT_INSERT_CONF < low_conf_threshold, qwen_replaced=True)
            for t in corr_tokens
        ]

    orig_keys = [_normalise(w.text) for w in original_words]
    corr_keys = [_normalise(t) for t in corr_tokens]
    matcher = difflib.SequenceMatcher(None, orig_keys, corr_keys, autojunk=False)

    out: list[Word] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                o = original_words[i]
                out.append(Word(text=corr_tokens[j], confidence=o.confidence,
                                needs_review=o.needs_review, qwen_replaced=False))
        elif tag == "replace":
            confs = [original_words[i].confidence for i in range(i1, i2)]
            avg = sum(confs) / len(confs) if confs else DEFAULT_INSERT_CONF
            for j in range(j1, j2):
                out.append(Word(text=corr_tokens[j], confidence=round(avg, 4),
                                needs_review=avg < low_conf_threshold, qwen_replaced=True))
        elif tag == "insert":
            for j in range(j1, j2):
                out.append(Word(text=corr_tokens[j], confidence=DEFAULT_INSERT_CONF,
                                needs_review=DEFAULT_INSERT_CONF < low_conf_threshold,
                                qwen_replaced=True))
        # tag == "delete": original word dropped by the corrector -> nothing emitted
    return out


def align_page(
    conf_words: list[tuple[str, float]],
    corrected_lines: list[str],
    low_conf_threshold: float = 0.60,
) -> list[list[Word]]:
    """Page-level alignment (faithful to post_correction_full_test_v2.ipynb).

    Aligns the flat stream of confidence-scored raw words against the flat stream
    of corrected words (which carry their corrected-line index), then groups the
    resulting words back by corrected line. Robust when the corrected text has a
    different line structure than the raw OCR (common here).

    Opcodes: equal -> original confidence; replace -> avg confidence of the
    replaced originals + qwen_replaced; insert -> default confidence (grey, not
    flagged, matching the v2 HTML); delete -> dropped.
    Returns one list[Word] per corrected line.
    """
    n_lines = len(corrected_lines)
    grouped: list[list[Word]] = [[] for _ in range(n_lines)]
    # flat corrected words tagged with their line index
    corr_wl: list[tuple[int, str]] = []
    for li, line in enumerate(corrected_lines):
        for w in line.split():
            corr_wl.append((li, w))
    if not corr_wl:
        return grouped

    orig_keys = [_normalise(w) for w, _ in conf_words]
    corr_keys = [_normalise(w) for _, w in corr_wl]
    matcher = difflib.SequenceMatcher(None, orig_keys, corr_keys, autojunk=False)

    def emit(li: int, text: str, conf: float, replaced: bool) -> None:
        grouped[li].append(Word(text=text, confidence=round(conf, 4),
                                needs_review=conf < low_conf_threshold,
                                qwen_replaced=replaced))

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                li, w = corr_wl[j]
                emit(li, w, conf_words[i][1], False)
        elif tag == "replace":
            confs = [conf_words[i][1] for i in range(i1, i2)]
            avg = sum(confs) / len(confs) if confs else DEFAULT_INSERT_CONF
            for j in range(j1, j2):
                li, w = corr_wl[j]
                emit(li, w, avg, True)
        elif tag == "insert":
            for j in range(j1, j2):
                li, w = corr_wl[j]
                emit(li, w, DEFAULT_INSERT_CONF, False)
        # delete: raw word dropped -> nothing
    return grouped
