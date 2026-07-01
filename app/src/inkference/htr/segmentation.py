"""Line segmentation.

Two backends behind one interface:

  KrakenSegmenter      Kraken baseline segmentation (+ optional DocLayout-YOLO
                       content gate, + polygon-masked crops). Ports the knobs and
                       behaviour documented in info_files/line_segmentation_output.txt.

  ProjectionSegmenter  Dependency-light fallback using a horizontal projection
                       profile. No Kraken/torch needed. Good enough for clean,
                       single-column pages and for testing the full pipeline on
                       free CPU before Kraken is installed.

`get_segmenter("auto")` returns Kraken if importable, else the projection backend.

All heavy imports are lazy so this module imports cleanly with nothing installed.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..config import HTRConfig
from ..config import htr as default_htr
from ..schemas import BBox

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image


@dataclass
class LineCrop:
    """One detected line: its crop plus where it sits on the page."""

    index: int
    bbox: BBox
    image: "Image.Image"


class Segmenter(Protocol):
    def segment(self, page: "Image.Image") -> list[LineCrop]: ...


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def downscale(page: "Image.Image", max_long_edge: int) -> tuple["Image.Image", float]:
    """Cap the long edge to keep free-CPU work bounded. Returns (image, scale)."""
    w, h = page.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return page, 1.0
    scale = max_long_edge / long_edge
    new_size = (round(w * scale), round(h * scale))
    return page.resize(new_size), scale


def _pad_clip(bbox: BBox, pad_x: int, pad_y: int, w: int, h: int) -> BBox:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


# --------------------------------------------------------------------------- #
# Kraken backend
# --------------------------------------------------------------------------- #
class KrakenSegmenter:
    """Kraken baseline segmentation with polygon-masked crops."""

    def __init__(self, cfg: HTRConfig = default_htr) -> None:
        self.cfg = cfg
        self._model = None  # lazy: kraken's default blla model

    def _ensure_model(self):
        if self._model is None:
            from kraken.lib import vgsl  # noqa: F401  (import validates install)

            # blla.segment loads the bundled default model when model=None, so we
            # don't need to fetch one explicitly. Kept as a hook for a custom model.
            self._model = "default"
        return self._model

    def segment(self, page: "Image.Image") -> list[LineCrop]:
        from kraken import blla

        from PIL import Image, ImageDraw

        self._ensure_model()
        rgb = page.convert("RGB")
        seg = blla.segment(rgb)  # kraken.containers.Segmentation

        crops: list[LineCrop] = []
        w, h = rgb.size
        idx = 0
        for line in getattr(seg, "lines", []):
            boundary = list(getattr(line, "boundary", []) or [])
            if not boundary:
                continue
            xs = [int(p[0]) for p in boundary]
            ys = [int(p[1]) for p in boundary]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            if (x1 - x0) < self.cfg.min_w or (y1 - y0) < self.cfg.min_h:
                continue  # drop noise boxes (MIN_W / MIN_H)

            x0, y0, x1, y1 = _pad_clip((x0, y0, x1, y1), self.cfg.pad_x, self.cfg.pad_y, w, h)
            crop = rgb.crop((x0, y0, x1, y1))

            if self.cfg.mask_to_polygon:
                # White-out everything outside the line polygon so neighbour
                # ascenders/descenders don't bleed in (MASK_TO_POLYGON=True).
                mask = Image.new("L", crop.size, 0)
                shifted = [(px - x0, py - y0) for px, py in zip(xs, ys)]
                ImageDraw.Draw(mask).polygon(shifted, fill=255)
                white = Image.new("RGB", crop.size, (255, 255, 255))
                crop = Image.composite(crop, white, mask)

            crops.append(LineCrop(index=idx, bbox=(x0, y0, x1, y1), image=crop))
            idx += 1

        # Kraken returns reading order; sort top-to-bottom as a safety net.
        crops.sort(key=lambda c: c.bbox[1])
        for i, c in enumerate(crops):
            c.index = i
        return crops


# --------------------------------------------------------------------------- #
# Projection-profile fallback (no Kraken)
# --------------------------------------------------------------------------- #
class ProjectionSegmenter:
    """Find line bands from the horizontal ink-projection profile.

    Binarises the page, sums dark pixels per row, and splits into bands wherever
    ink rises above a fraction of the row maximum. Simple but robust for clean,
    single-column manuscript pages; not a replacement for Kraken on dense or
    multi-column layouts.
    """

    def __init__(self, cfg: HTRConfig = default_htr) -> None:
        self.cfg = cfg

    def segment(self, page: "Image.Image") -> list[LineCrop]:
        import numpy as np
        from PIL import ImageOps

        gray = ImageOps.grayscale(page)
        arr = np.asarray(gray, dtype=np.float32)
        h, w = arr.shape
        # Ink = dark pixels. Threshold at Otsu-ish midpoint of the histogram.
        thresh = float(arr.mean()) - 0.4 * float(arr.std())
        ink = (arr < thresh).astype(np.float32)
        row_ink = ink.sum(axis=1)

        if row_ink.max() <= 0:
            return []
        active = row_ink > (0.04 * row_ink.max())  # rows that carry text

        bands: list[tuple[int, int]] = []
        start = None
        for y in range(h):
            if active[y] and start is None:
                start = y
            elif not active[y] and start is not None:
                bands.append((start, y))
                start = None
        if start is not None:
            bands.append((start, h))

        rgb = page.convert("RGB")
        crops: list[LineCrop] = []
        idx = 0
        for (y0, y1) in bands:
            if (y1 - y0) < self.cfg.min_h:
                continue
            # Trim horizontal extent to the inked columns within this band.
            band_ink = ink[y0:y1].sum(axis=0)
            cols = np.where(band_ink > 0)[0]
            if cols.size == 0:
                continue
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            if (x1 - x0) < self.cfg.min_w:
                continue
            bbox = _pad_clip((x0, y0, x1, y1), self.cfg.pad_x, self.cfg.pad_y, w, h)
            crops.append(LineCrop(index=idx, bbox=bbox, image=rgb.crop(bbox)))
            idx += 1
        return crops


def kraken_available() -> bool:
    return importlib.util.find_spec("kraken") is not None


def get_segmenter(name: str = "auto", cfg: HTRConfig = default_htr) -> Segmenter:
    if name == "kraken":
        return KrakenSegmenter(cfg)
    if name == "projection":
        return ProjectionSegmenter(cfg)
    if name == "auto":
        return KrakenSegmenter(cfg) if kraken_available() else ProjectionSegmenter(cfg)
    raise ValueError(f"unknown segmenter: {name!r}")
