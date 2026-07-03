# Qwen few-shot page-level post-correction

> LLM proofreading stage that runs **after** TrOCR recognition, per page, using
> **few-shot in-context examples**. Fixes OCR errors while preserving archaic spelling,
> then aligns the corrected words back to their original TrOCR confidence so the Reader
> keeps its confidence tint and highlights the model's edits.

## Why
TrOCR reads one line at a time with no language context → misreads, split/merged tokens,
spacing errors (e.g. low-confidence words like "directors"/"venue"). A page-level LLM pass
with a few worked examples fixes these and measurably lowers CER, improving both the Reader
and the RAG answers (which then index the cleaned text).

## Few-shot design (the core)
Per-page chat-template message list:
1. **system** — maritime-journal domain (Forster / Cook's 2nd voyage) + rules: *fix only OCR
   errors; preserve archaic spelling, capitalisation, punctuation; do not modernise or add
   content; return the same number of lines*; how to read `<<…>>` low-confidence markers.
2. **N example turns** (`CORRECTION_NUM_SHOTS`, default 2–3): `user` = raw page with low-conf
   words wrapped `<<word>>`, `assistant` = the hand-verified corrected page.
3. **final user turn** — the actual page in the same format.

Exemplars: [app/src/inkference/htr/few_shot_examples.json](../app/src/inkference/htr/few_shot_examples.json)
(`raw` → `corrected` pairs). Seeded from Captain Cook GT; **replace/extend with your own
verified corrected pages** for best results.

## Runtime (configurable — `CorrectionConfig`)
- **`local`** — Qwen3 via `transformers` (reuses the notebook's `run_qwen`: `apply_chat_template`,
  append `/no_think`, strip `<think>…</think>`). Default `CORRECTION_MODEL_ID=Qwen/Qwen3-1.7B`
  for CPU; use `Qwen/Qwen3-4B` on a GPU (float16 / 4-bit bitsandbytes, as in the notebook).
- **`api`** — hosted Qwen (OpenRouter/DashScope) via `_call_openai_compatible` in
  [rag/llm.py](../app/src/inkference/rag/llm.py). Best quality, needs a key; recommended for
  the free Space.
- **`CORRECTION_ENABLED`** (default true) — off = pipeline behaves exactly as before.

Env vars: `CORRECTION_ENABLED`, `CORRECTION_BACKEND` (local|api), `CORRECTION_MODEL_ID`,
`CORRECTION_API_BASE`, `CORRECTION_API_KEY`, `CORRECTION_NUM_SHOTS`, `CORRECTION_MAX_NEW_TOKENS`,
`CORRECTION_TEMPERATURE`.

## Confidence-preserving alignment (reused from the notebook)
Page-level correction rewrites text, so we can't keep per-word confidence naively. We port
the v2 notebook's **`difflib.SequenceMatcher`** alignment ([htr/align.py](../app/src/inkference/htr/align.py)):
map each corrected word back to the original TrOCR words by opcode —
`equal`→original confidence, `replace`→avg confidence of the replaced originals +
`qwen_replaced=True`, `insert`→default confidence, `delete`→skip. Result: corrected words
still carry confidence, and Qwen's edits are flagged.

## Data flow
```
segment → recognize (+word confidence) → [correction]
     lines[].words ──► PostCorrector.correct_page(lines)   # few-shot Qwen, page-level
                        └─► corrected line text (same #lines)
     align(orig words, corrected line) ──► corrected_words (confidence + qwen_replaced)
     → Line.corrected_text / Line.corrected_words, PageResult.corrected
     → store (pages.corrected_text, lines.corrected_text, words.qwen_replaced)
     → RAG indexes corrected text; Reader shows Raw · Corrected toggle
```

## Files
| File | Role |
|---|---|
| `app/src/inkference/htr/correction.py` | `PostCorrector` — few-shot prompt + local/API backends |
| `app/src/inkference/htr/align.py` | `SequenceMatcher` word alignment (ported) |
| `app/src/inkference/htr/few_shot_examples.json` | exemplar `raw → corrected` pairs |
| `app/src/inkference/config.py` | `CorrectionConfig` |
| `app/src/inkference/schemas.py` | `Stage.CORRECTION`, `JobStatus.CORRECTING`, `Word.qwen_replaced`, `Line.corrected_*`, `PageResult.corrected` |
| `app/src/inkference/htr/pipeline.py` | 4th stage: correction + align |
| `app/src/inkference/store/store.py` | corrected columns; `iter_pages_text(prefer_corrected=True)` |
| `app/src/inkference/api/*` | stage→status map, `get_corrector()`, health fields |
| `app/frontend/*` | Reader Raw·Corrected toggle (green edits); Upload 4th stepper step |

Prior art: `~/Downloads/AlexFiles/post_correction_full_test*.ipynb` (Qwen3-4B `run_qwen`,
weather/date normalisation, and the confidence-coloured HTML alignment this reuses).

## Free-tier / CPU note
Few-shot lengthens the prompt but page-level keeps it **one call per page**. On free CPU
prefer `Qwen3-1.7B` or the hosted API; `Qwen3-4B`/4-bit is the GPU path — swap via config,
no other code change.

## Verify
1. `correct_page` on a known-bad page → fixed, archaic spelling intact, line count preserved;
   compare zero-shot vs few-shot.
2. `align()` maps corrected→confidence; `qwen_replaced` set on edits.
3. Pipeline `Stage.CORRECTION` fires; `CORRECTION_ENABLED=false` → identical to before.
4. RAG uses corrected text; Reader toggle shows green edits + tint.
5. CER A/B (`jiwer`) raw vs corrected on the manual-transcription test set.
