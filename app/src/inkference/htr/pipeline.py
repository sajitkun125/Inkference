"""HTR pipeline orchestrator: page image -> segment -> recognize -> confidence.

Produces a PageResult and emits stage/progress events so the API can drive the
design's three-step stepper (Segmentation -> Recognition -> Confidence) and the
per-page progress bar (e.g. "Recognizing 64%").
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

logger = logging.getLogger("inkference.correction")

from ..config import CorrectionConfig, HTRConfig
from ..config import correction as default_correction
from ..config import htr as default_htr
from .align import align_line
from ..schemas import Line, PageResult, Stage
from .correction import PostCorrector
from .recognition import TrOCRRecognizer
from .segmentation import LineCrop, Segmenter, downscale, get_segmenter

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

# progress(stage, fraction_0_to_1, message)
ProgressCB = Callable[[Stage, float, str], None]
# on_segmented(bboxes, page_width, page_height) — fired the instant segmentation
# finishes so the UI can draw line boxes before recognition/confidence/correction.
SegmentedCB = Callable[[list, int, int], None]


def _noop(stage: Stage, frac: float, msg: str) -> None:  # pragma: no cover
    pass


class HTRPipeline:
    def __init__(
        self,
        cfg: HTRConfig = default_htr,
        segmenter: Segmenter | None = None,
        recognizer: TrOCRRecognizer | None = None,
        corrector: PostCorrector | None = None,
        correction_cfg: CorrectionConfig = default_correction,
    ) -> None:
        self.cfg = cfg
        self.correction_cfg = correction_cfg
        self.segmenter = segmenter or get_segmenter("auto", cfg)
        self.recognizer = recognizer or TrOCRRecognizer(cfg)
        # Lazily created on first use unless injected; skipped when disabled.
        self._corrector = corrector

    def process_image(
        self,
        page: "Image.Image",
        page_number: int = 1,
        progress: ProgressCB = _noop,
        on_segmented: "SegmentedCB | None" = None,
    ) -> PageResult:
        # --- Stage 1: segmentation ---------------------------------------- #
        progress(Stage.SEGMENTATION, 0.0, "Segmenting…")
        scaled, scale = downscale(page, self.cfg.max_page_long_edge)
        crops: list[LineCrop] = self.segmenter.segment(scaled)
        w, h = scaled.size
        progress(Stage.SEGMENTATION, 1.0, f"{len(crops)} lines detected")
        # Publish the boxes now (recognition/correction still to come) so the UI can
        # overlay them immediately instead of waiting minutes for the full result.
        if on_segmented is not None:
            on_segmented([list(c.bbox) for c in crops], w, h)

        if not crops:
            return PageResult(page_number, w, h, lines=[], scale=scale)

        # --- Stage 2: recognition ----------------------------------------- #
        progress(Stage.RECOGNITION, 0.0, "Recognizing…")
        recognitions = self.recognizer.recognize([c.image for c in crops])
        progress(Stage.RECOGNITION, 1.0, "Recognized")

        # --- Stage 3: confidence flagging --------------------------------- #
        progress(Stage.CONFIDENCE, 0.0, "Scoring confidence…")
        thr = self.cfg.low_confidence_threshold
        lines: list[Line] = []
        for crop, rec in zip(crops, recognitions):
            for word in rec.words:
                word.needs_review = word.confidence < thr
            line_needs_review = any(wd.needs_review for wd in rec.words)
            lines.append(
                Line(
                    index=crop.index,
                    bbox=crop.bbox,
                    text=rec.text,
                    confidence=rec.confidence,
                    words=rec.words,
                    needs_review=line_needs_review,
                )
            )
        progress(Stage.CONFIDENCE, 1.0, "Complete")

        # --- Stage 4: Qwen post-correction (optional) --------------------- #
        if self.correction_cfg.enabled:
            self._correct(lines, progress)

        return PageResult(page_number, w, h, lines=lines, scale=scale)

    def _corrector_instance(self) -> PostCorrector:
        if self._corrector is None:
            self._corrector = PostCorrector(self.correction_cfg)
        return self._corrector

    def _correct(self, lines: list[Line], progress: ProgressCB) -> None:
        """Run the corrector over the page and align results back onto each Line."""
        progress(Stage.CORRECTION, 0.0, "Proofreading…")
        try:
            result = self._corrector_instance().correct_page(lines)
        except Exception as exc:  # never fail ingest because of correction
            # Surface it: log to the server console AND put it in the stage message
            # so a failed correction is diagnosable instead of silently absent.
            logger.warning("post-correction failed, skipping page: %s", exc)
            progress(Stage.CORRECTION, 1.0, f"Correction skipped ({exc})")
            return
        if result.lines is None:
            logger.warning(
                "post-correction returned %d lines for a %d-line page — keeping raw",
                len(result.page_text.splitlines()), len(lines),
            )
            progress(Stage.CORRECTION, 1.0, "Complete (unaligned)")
            return
        thr = self.cfg.low_confidence_threshold
        for line, corrected in zip(lines, result.lines):
            line.corrected_text = corrected
            line.corrected_words = align_line(line.words, corrected, thr)
        progress(Stage.CORRECTION, 1.0, "Complete")

    def process_path(
        self,
        image_path: str | Path,
        page_number: int = 1,
        progress: ProgressCB = _noop,
        on_segmented: "SegmentedCB | None" = None,
    ) -> PageResult:
        from PIL import Image

        with Image.open(image_path) as im:
            return self.process_image(im.copy(), page_number, progress, on_segmented)
