# captain-rag — Q&A chat over the transcribed journal

A minimal RAG Q&A chat over `data/transcriptions/` (923 transcribed pages, Books 1-6
of Captain Cook's voyage journals), built as a leaner CLI-only version of the
RAG pipeline in `data/Capstone-Project-inkferenceApp/app/src/inkference/rag/`
(chunk → embed with `sentence-transformers` → FAISS retrieval → grounded LLM
answer with source-page citations, extractive fallback with no API key).

Answers are generated **in character as Johann Reinhold Forster**, the naturalist
who authored the journal — first person, grounded only in the retrieved excerpts
from his own writing (system prompt in `llm.py`). Default LLM provider is
**Groq running `openai/gpt-oss-120b`** (OpenAI's open-weight reasoning model,
free tier); Gemini, Mistral, OpenAI, and Claude also work.

```
app/
├── pyproject.toml
├── .env.example        copy to .env and add an LLM key (optional)
└── src/captain_rag/
    ├── config.py        paths + RAGConfig, env-driven
    ├── corpus.py         loads data/transcriptions/B<n>_P<page>.txt
    ├── index.py           chunk + embed + FAISS, persisted to app/.rag_index/
    ├── llm.py              gemini/groq/openai/claude/mistral over REST + $0 fallback
    ├── answer.py            retrieve -> generate -> answer + source pages
    └── chat.py               CLI REPL entrypoint
```

## Run (from the repo root)

```bash
pip install -r requirements.txt && pip install -e ./app
cp app/.env.example app/.env   # optional: add an LLM API key for generated answers
captain-rag                    # first run indexes data/transcriptions/ (~1-2 min)
```

Without an API key, answers fall back to the most relevant retrieved passage
(still useful — just not a generated summary). Flags: `--rebuild` to re-index,
`--debug` to print retrieved snippets, `--top-k N` to override retrieval depth.
