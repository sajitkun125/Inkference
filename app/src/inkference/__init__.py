"""Inkference — HTR + RAG platform for historical handwritten documents.

Subpackages:
    htr    line segmentation + TrOCR recognition + confidence scoring
    rag    embeddings + FAISS retrieval + grounded answer generation
    store  SQLite document/page/line/word store + filesystem assets
    api    FastAPI backend
"""

__version__ = "0.1.0"
