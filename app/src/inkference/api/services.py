"""Shared singletons and the background ingestion job runner.

Heavy objects (TrOCR pipeline, embedder) load lazily on first use. Ingestion runs
on a single-worker thread pool so CPU-bound OCR never blocks the event loop and
free-CPU pages process serially; the frontend polls GET /jobs/{id} for progress.
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from ..config import htr as htr_cfg
from ..htr.pipeline import HTRPipeline
from ..rag.index import RagIndex
from ..schemas import JobStatus, Stage
from ..store import DocumentStore

logger = logging.getLogger("inkference.ingest")

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
    logger.info("ingest job %s: doc=%s, %d page(s) queued", job_id, doc_id, total)
    store.update_job(job_id, status=JobStatus.QUEUED, total_pages=total, done_pages=0)
    try:
        pipeline = get_pipeline()
        for done, (page_id, page_number, image_path) in enumerate(page_specs):
            def progress(stage: Stage, frac: float, msg: str, _done=done, _pn=page_number) -> None:
                overall = (_done + frac) / total
                logger.debug("job %s page %s: %s %.0f%% — %s",
                             job_id, _pn, stage.value, frac * 100, msg)
                store.update_job(
                    job_id,
                    status=_STAGE_STATUS.get(stage, JobStatus.RECOGNIZING),
                    stage=stage,
                    progress=round(overall, 4),
                    message=f"Page {page_number} — {msg}",
                )

            def on_segmented(bboxes, w, h, _pn=page_number) -> None:
                # Surface line boxes the instant segmentation finishes (before the slow
                # recognition/correction) so the UI can overlay them right away.
                store.update_job(job_id, seg_preview=json.dumps(
                    {"page_number": _pn, "width": w, "height": h, "boxes": bboxes}))

            logger.info("job %s: processing page %s (%d/%d)", job_id, page_number, done + 1, total)
            store.set_page_status(page_id, "processing")
            result = pipeline.process_path(image_path, page_number, progress, on_segmented)
            store.save_page_result(page_id, result)
            store.update_job(job_id, done_pages=done + 1)
            logger.info("job %s: page %s done — %d lines, avg conf %.2f",
                        job_id, page_number, len(result.lines), result.avg_confidence)

        # Incrementally index only the newly-added pages (a full rebuild re-embeds the
        # whole corpus, ~90s for 900 pages, and needlessly delays "Complete").
        page_numbers = [pn for (_pid, pn, _path) in page_specs]
        n_chunks = get_index().add_pages(doc_id, store, page_numbers)
        store.update_job(
            job_id, status=JobStatus.COMPLETE, progress=1.0, message="Complete"
        )
        logger.info("ingest job %s complete; RAG index updated (+%d page(s), %s chunks)",
                    job_id, len(page_numbers), n_chunks)
    except Exception as exc:  # surface failure to the job poller
        logger.exception("ingest job %s FAILED: %s", job_id, exc)
        store.update_job(
            job_id, status=JobStatus.FAILED,
            error=f"{exc}\n{traceback.format_exc()}", message=str(exc),
        )
