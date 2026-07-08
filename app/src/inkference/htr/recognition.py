"""TrOCR recognition with per-word and per-line confidence.

Confidence comes from the decoder's own token probabilities: we generate with
`output_scores=True` and use `compute_transition_scores(..., normalize_logits=True)`
to get a calibrated log-prob per generated token, exp() it, group tokens into words,
and average. This drives the design's word-tinting, avg-confidence %, and the
"needs review" / "N words below 60%" flags.

Heavy imports (torch/transformers) are lazy so the module imports with nothing installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import HTRConfig
from ..config import htr as default_htr
from ..schemas import Word

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

# BPE/sentencepiece markers that indicate the start of a new word.
_WORD_START_MARKERS = ("Ġ", "▁")  # 'Ġ' (RoBERTa/GPT2), '▁' (sentencepiece)


@dataclass
class LineRecognition:
    text: str
    confidence: float
    words: list[Word]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


class TrOCRRecognizer:
    def __init__(self, cfg: HTRConfig = default_htr) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self._processor = None
        self._model = None

    # -- lazy model load ---------------------------------------------------- #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._processor = TrOCRProcessor.from_pretrained(self.cfg.trocr_model_id)
        model = VisionEncoderDecoderModel.from_pretrained(self.cfg.trocr_model_id)
        model.to(self.device)
        model.eval()
        self._model = model
        self._torch = torch

    # -- public API --------------------------------------------------------- #
    def recognize(self, images: list["Image.Image"]) -> list[LineRecognition]:
        """Recognize a list of line crops, batching per recognition_batch_size."""
        if not images:
            return []
        self._ensure_loaded()
        out: list[LineRecognition] = []
        bs = max(1, self.cfg.recognition_batch_size)
        for i in range(0, len(images), bs):
            out.extend(self._recognize_batch(images[i : i + bs]))
        return out

    def _recognize_batch(self, images: list["Image.Image"]) -> list[LineRecognition]:
        torch = self._torch
        proc = self._processor
        model = self._model

        pixel_values = proc(
            images=[im.convert("RGB") for im in images], return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            gen = model.generate(
                pixel_values,
                num_beams=self.cfg.num_beams,
                max_new_tokens=self.cfg.max_target_length,
                output_scores=True,
                return_dict_in_generate=True,
            )

        beam_indices = getattr(gen, "beam_indices", None)
        transition = model.compute_transition_scores(
            gen.sequences, gen.scores, beam_indices=beam_indices, normalize_logits=True
        )
        probs = transition.exp().cpu()  # [batch, gen_len-1]
        sequences = gen.sequences.cpu()  # [batch, gen_len]

        tokenizer = proc.tokenizer
        special_ids = set(tokenizer.all_special_ids)

        results: list[LineRecognition] = []
        for b in range(sequences.size(0)):
            # sequences[b,0] is the decoder start token; transition aligns to [1:].
            tok_ids = sequences[b, 1:].tolist()
            tok_probs = probs[b].tolist()
            results.append(self._aggregate(tok_ids, tok_probs, tokenizer, special_ids))
        return results

    def _aggregate(self, tok_ids, tok_probs, tokenizer, special_ids) -> LineRecognition:
        """Group sub-word tokens into words with averaged confidence."""
        words: list[Word] = []
        cur_tokens: list[str] = []
        cur_probs: list[float] = []

        def flush():
            if not cur_tokens:
                return
            text = tokenizer.convert_tokens_to_string(cur_tokens).strip()
            if text:
                conf = sum(cur_probs) / len(cur_probs)
                words.append(Word(text=text, confidence=round(conf, 4)))
            cur_tokens.clear()
            cur_probs.clear()

        for tid, prob in zip(tok_ids, tok_probs):
            if tid in special_ids:
                continue
            token = tokenizer.convert_ids_to_tokens(tid)
            starts_word = token.startswith(_WORD_START_MARKERS)
            if starts_word and cur_tokens:
                flush()
            cur_tokens.append(token)
            cur_probs.append(float(prob))
        flush()

        text = " ".join(w.text for w in words)
        line_conf = (
            sum(w.confidence for w in words) / len(words) if words else 0.0
        )
        return LineRecognition(text=text, confidence=round(line_conf, 4), words=words)
