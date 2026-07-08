"""Line segmentation — Kraken baseline segmentation (`blla`).

Faithful port of notebooks/segmentManualTranscriptionsIntoLinesWithKraken*.ipynb:
Kraken's bundled baseline model predicts, per line, a **baseline + boundary
polygon**; we take the polygon, derive a bbox, drop noise boxes (MIN_W/MIN_H),
sort top-to-bottom by vertical centre, and crop each line **masked to its polygon**
(dilated by POLY_PAD) so neighbouring ascenders/descenders don't bleed in.

The old dependency-light ProjectionSegmenter is disabled — see the commented block
at the bottom. Kraken is now the only backend.

Heavy imports (kraken/torch/PIL) are lazy so this module imports with nothing installed.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..config import HTRConfig
from ..config import htr as default_htr
from ..schemas import BBox

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image


@dataclass
class LineCrop:
    """One detected line: its polygon-masked crop plus where it sits on the page."""

    index: int
    bbox: BBox
    image: "Image.Image"
    polygon: list[tuple[float, float]] = field(default_factory=list)


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
    return page.resize((round(w * scale), round(h * scale))), scale


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def _boundary_of(line):
    """A line's boundary polygon as a list of points, across kraken API versions."""
    b = getattr(line, "boundary", None)
    if b is None and isinstance(line, dict):
        b = line.get("boundary")
    return b


# --------------------------------------------------------------------------- #
# Kraken backend
# --------------------------------------------------------------------------- #
class KrakenSegmenter:
    """Kraken baseline segmentation with polygon-masked line crops."""

    def __init__(self, cfg: HTRConfig = default_htr) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self._seg_model = None
        self._model_loaded = False

    def _ensure_model(self):
        """Load Kraken's bundled default baseline model (blla.mlmodel) once.

        Falls back to None, in which case blla.segment uses its own default."""
        if self._model_loaded:
            return self._seg_model
        import os

        import kraken
        from kraken.lib import vgsl

        cand = os.path.join(os.path.dirname(kraken.__file__), "blla.mlmodel")
        if os.path.exists(cand):
            self._seg_model = vgsl.TorchVGSLModel.load_model(cand)
        else:
            self._seg_model = None
        self._model_loaded = True
        return self._seg_model

    def _run_blla(self, im):
        from kraken import blla

        model = self._ensure_model()
        # the `device` kwarg exists on newer kraken; fall back gracefully.
        try:
            return blla.segment(im, model=model, device=self.device)
        except TypeError:
            return blla.segment(im, model=model) if model is not None else blla.segment(im)

    def segment(self, page: "Image.Image") -> list[LineCrop]:
        from PIL import Image, ImageDraw, ImageFilter

        rgb = page.convert("RGB")
        seg = self._run_blla(rgb)
        lines = getattr(seg, "lines", None)
        if lines is None and isinstance(seg, dict):
            lines = seg.get("lines", [])
        lines = lines or []

        W, H = rgb.size
        cfg = self.cfg

        # 1. polygon -> bbox record, dropping noise boxes (MIN_W / MIN_H).
        records: list[tuple[BBox, list[tuple[float, float]]]] = []
        for ln in lines:
            poly = _boundary_of(ln)
            if not poly:
                continue
            pts = [(float(p[0]), float(p[1])) for p in poly]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            if (bbox[2] - bbox[0]) < cfg.min_w or (bbox[3] - bbox[1]) < cfg.min_h:
                continue
            records.append((bbox, pts))

        # 2. reading order: top-to-bottom by vertical centre.
        records.sort(key=lambda r: (r[0][1] + r[0][3]) / 2)

        # 3. polygon-masked crop per line (PAD_X/PAD_Y + MaxFilter(POLY_PAD) dilation).
        crops: list[LineCrop] = []
        for idx, (bbox, pts) in enumerate(records):
            x0, y0, x1, y1 = bbox
            cx0 = max(0, int(x0) - cfg.pad_x)
            cy0 = max(0, int(y0) - cfg.pad_y)
            cx1 = min(W, int(x1) + cfg.pad_x)
            cy1 = min(H, int(y1) + cfg.pad_y)
            crop = rgb.crop((cx0, cy0, cx1, cy1))

            if cfg.mask_to_polygon:
                mask = Image.new("L", crop.size, 0)
                ImageDraw.Draw(mask).polygon(
                    [(px - cx0, py - cy0) for px, py in pts], fill=255
                )
                if cfg.poly_pad > 0:  # dilate so the line's own strokes aren't clipped
                    mask = mask.filter(ImageFilter.MaxFilter(cfg.poly_pad * 2 + 1))
                white = Image.new("RGB", crop.size, (255, 255, 255))
                crop = Image.composite(crop, white, mask)

            crops.append(
                LineCrop(index=idx, bbox=(cx0, cy0, cx1, cy1), image=crop, polygon=pts)
            )
        return crops


def kraken_available() -> bool:
    return importlib.util.find_spec("kraken") is not None


def get_segmenter(name: str = "kraken", cfg: HTRConfig = default_htr) -> Segmenter:
    """Return the Kraken segmenter. The projection fallback is disabled."""
    if name in ("kraken", "auto"):
        if not kraken_available():
            raise ImportError(
                "Kraken is not installed. Install it (see the note in requirements.txt — "
                "Kraken pins its own torch, so a dedicated venv is recommended)."
            )
        return KrakenSegmenter(cfg)
    if name == "projection":
        raise ValueError("ProjectionSegmenter is disabled; use the Kraken segmenter.")
    raise ValueError(f"unknown segmenter: {name!r}")


# --------------------------------------------------------------------------- #
# DISABLED — dependency-light projection-profile fallback (kept for reference).
# Re-enable in get_segmenter() only if you deliberately want a Kraken-free path.
# --------------------------------------------------------------------------- #
# class ProjectionSegmenter:
#     """Find line bands from the horizontal ink-projection profile. Weak on dense
#     or multi-column layouts — replaced by Kraken baseline segmentation."""
#
#     def __init__(self, cfg: HTRConfig = default_htr) -> None:
#         self.cfg = cfg
#
#     def segment(self, page: "Image.Image") -> list[LineCrop]:
#         import numpy as np
#         from PIL import ImageOps
#
#         gray = ImageOps.grayscale(page)
#         arr = np.asarray(gray, dtype=np.float32)
#         h, w = arr.shape
#         thresh = float(arr.mean()) - 0.4 * float(arr.std())
#         ink = (arr < thresh).astype(np.float32)
#         row_ink = ink.sum(axis=1)
#         if row_ink.max() <= 0:
#             return []
#         active = row_ink > (0.04 * row_ink.max())
#         bands: list[tuple[int, int]] = []
#         start = None
#         for y in range(h):
#             if active[y] and start is None:
#                 start = y
#             elif not active[y] and start is not None:
#                 bands.append((start, y))
#                 start = None
#         if start is not None:
#             bands.append((start, h))
#         rgb = page.convert("RGB")
#         crops: list[LineCrop] = []
#         idx = 0
#         for (y0, y1) in bands:
#             if (y1 - y0) < self.cfg.min_h:
#                 continue
#             band_ink = ink[y0:y1].sum(axis=0)
#             cols = np.where(band_ink > 0)[0]
#             if cols.size == 0:
#                 continue
#             x0, x1 = int(cols[0]), int(cols[-1]) + 1
#             if (x1 - x0) < self.cfg.min_w:
#                 continue
#             x0p = max(0, x0 - self.cfg.pad_x); y0p = max(0, y0 - self.cfg.pad_y)
#             x1p = min(w, x1 + self.cfg.pad_x); y1p = min(h, y1 + self.cfg.pad_y)
#             crops.append(LineCrop(index=idx, bbox=(x0p, y0p, x1p, y1p),
#                                   image=rgb.crop((x0p, y0p, x1p, y1p))))
#             idx += 1
#         return crops
