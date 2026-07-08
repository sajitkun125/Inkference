"""Dependency-free domain types shared across htr / store / rag.

Plain dataclasses (no pydantic) so the HTR core imports without the API stack.
The API layer mirrors these as pydantic response models."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# Bounding box in pixel coordinates on the (possibly downscaled) page image.
BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1)


class Stage(str, Enum):
    """The pipeline stages shown in the design's stepper."""

    SEGMENTATION = "segmentation"
    RECOGNITION = "recognition"
    CONFIDENCE = "confidence"
    CORRECTION = "correction"  # optional Qwen post-correction stage


class JobStatus(str, Enum):
    QUEUED = "queued"
    SEGMENTING = "segmenting"
    RECOGNIZING = "recognizing"
    SCORING = "scoring"
    CORRECTING = "correcting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Word:
    text: str
    confidence: float  # 0..1 mean token probability for this word
    needs_review: bool = False  # set by the pipeline using the low-conf threshold
    qwen_replaced: bool = False  # set on corrected words the LLM changed


@dataclass
class Line:
    index: int  # 0-based, top-to-bottom reading order
    bbox: BBox
    text: str
    confidence: float  # mean token probability across the line
    words: list[Word] = field(default_factory=list)
    needs_review: bool = False
    # Populated by the post-correction stage (None/empty when correction is off).
    corrected_text: str | None = None
    corrected_words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageResult:
    page_number: int
    image_width: int
    image_height: int
    lines: list[Line] = field(default_factory=list)
    # scale applied to the original scan before processing (for overlay mapping)
    scale: float = 1.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def corrected(self) -> str:
        """Corrected page text (per-line corrected where available, else raw)."""
        return "\n".join(
            (line.corrected_text if line.corrected_text is not None else line.text)
            for line in self.lines
        )

    @property
    def has_correction(self) -> bool:
        return any(line.corrected_text is not None for line in self.lines)

    @property
    def avg_confidence(self) -> float:
        confs = [w.confidence for line in self.lines for w in line.words]
        return sum(confs) / len(confs) if confs else 0.0

    @property
    def low_confidence_word_count(self) -> int:
        return sum(1 for line in self.lines for w in line.words if w.needs_review)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["text"] = self.text
        d["corrected"] = self.corrected
        d["has_correction"] = self.has_correction
        d["avg_confidence"] = self.avg_confidence
        d["low_confidence_word_count"] = self.low_confidence_word_count
        return d
