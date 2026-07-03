"""Shared singletons and the background ingestion job runner.

Heavy objects (TrOCR pipeline, embedder) load lazily on first use. Ingestion runs
on a single-worker thread pool so CPU-bound OCR never blocks the event loop and
free-CPU pages process serially; the frontend polls GET /jobs/{id} for progress.
"""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from ..config import htr as htr_cfg
from ..htr.pipeline import HTRPipeline
from ..rag.index import RagIndex
from ..schemas import JobStatus, Stage
from ..store import DocumentStore

# Map a pipeline Stage -> the job status shown in the queue UI.
_STAGE_STATUS = {
    Stage.SEGMENTATION: JobStatus.SEGMENTING,
    Stage.RECOGNITION: JobStatus.RECOGNIZING,
    Stage.CONFIDENCE: JobStatus.SCORING,
    Stage.CORRECTION: JobStatus.CORRECTING,
}

_store: DocumentStore | None = None
_pipeline: HTRPipeline | None = None
_index: RagIndex | None = None
_pipeline_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store


def get_index() -> RagIndex:
    global _index
    if _index is None:
        _index = RagIndex()
    return _index


def get_pipeline() -> HTRPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = HTRPipeline(htr_cfg)
    return _pipeline


def submit_ingest(doc_id: int, page_specs: list[tuple[int, int, str]], job_id: int) -> None:
    """page_specs = [(page_id, page_number, image_path), ...]"""
    _executor.submit(_run_ingest, doc_id, page_specs, job_id)


def _run_ingest(doc_id: int, page_specs: list[tuple[int, int, str]], job_id: int) -> None:
    store = get_store()
    total = len(page_specs)
    store.update_job(job_id, status=JobStatus.QUEUED, total_pages=total, done_pages=0)
    try:
        pipeline = get_pipeline()
        for done, (page_id, page_number, image_path) in enumerate(page_specs):
            def progress(stage: Stage, frac: float, msg: str, _done=done) -> None:
                overall = (_done + frac) / total
                store.update_job(
                    job_id,
                    status=_STAGE_STATUS.get(stage, JobStatus.RECOGNIZING),
                    stage=stage,
                    progress=round(overall, 4),
                    message=f"Page {page_number} — {msg}",
                )

            store.set_page_status(page_id, "processing")
            result = pipeline.process_path(image_path, page_number, progress)
            store.save_page_result(page_id, result)
            store.update_job(job_id, done_pages=done + 1)

        # Rebuild the retrieval index now that new pages exist.
        get_index().build_from_store(doc_id, store)
        store.update_job(
            job_id, status=JobStatus.COMPLETE, progress=1.0, message="Complete"
        )
    except Exception as exc:  # surface failure to the job poller
        store.update_job(
            job_id, status=JobStatus.FAILED,
            error=f"{exc}\n{traceback.format_exc()}", message=str(exc),
        )
