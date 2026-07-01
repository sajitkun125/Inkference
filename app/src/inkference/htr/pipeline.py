"""HTR pipeline orchestrator: page image -> segment -> recognize -> confidence.

Produces a PageResult and emits stage/progress events so the API can drive the
design's three-step stepper (Segmentation -> Recognition -> Confidence) and the
per-page progress bar (e.g. "Recognizing 64%").
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..config import HTRConfig
from ..config import htr as default_htr
from ..schemas import Line, PageResult, Stage
from .recognition import TrOCRRecognizer
from .segmentation import LineCrop, Segmenter, downscale, get_segmenter

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

# progress(stage, fraction_0_to_1, message)
ProgressCB = Callable[[Stage, float, str], None]


def _noop(stage: Stage, frac: float, msg: str) -> None:  # pragma: no cover
    pass


class HTRPipeline:
    def __init__(
        self,
        cfg: HTRConfig = default_htr,
        segmenter: Segmenter | None = None,
        recognizer: TrOCRRecognizer | None = None,
    ) -> None:
        self.cfg = cfg
        self.segmenter = segmenter or get_segmenter("auto", cfg)
        self.recognizer = recognizer or TrOCRRecognizer(cfg)

    def process_image(
        self,
        page: "Image.Image",
        page_number: int = 1,
        progress: ProgressCB = _noop,
    ) -> PageResult:
        # --- Stage 1: segmentation ---------------------------------------- #
        progress(Stage.SEGMENTATION, 0.0, "Segmenting…")
        scaled, scale = downscale(page, self.cfg.max_page_long_edge)
        crops: list[LineCrop] = self.segmenter.segment(scaled)
        progress(Stage.SEGMENTATION, 1.0, f"{len(crops)} lines detected")

        w, h = scaled.size
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

        return PageResult(page_number, w, h, lines=lines, scale=scale)

    def process_path(
        self, image_path: str | Path, page_number: int = 1, progress: ProgressCB = _noop
    ) -> PageResult:
        from PIL import Image

        with Image.open(image_path) as im:
            return self.process_image(im.copy(), page_number, progress)
