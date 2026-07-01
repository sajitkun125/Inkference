"""Assemble a clean, loadable TrOCR inference model from a training checkpoint.

The training checkpoints under models/ contain weights + optimizer/scheduler state
but NO processor files (preprocessor_config.json, tokenizer). TrOCRProcessor must
therefore be taken from the base model the checkpoint was fine-tuned from. This
script copies the inference-relevant weights and writes the processor alongside
them, producing a directory that:

  * loads with TrOCRProcessor/VisionEncoderDecoderModel.from_pretrained(out_dir)
  * is ready to push:  huggingface-cli upload <user>/inkference-trocr <out_dir>

Usage:
  python scripts/prepare_model.py \
      --checkpoint models/bentham_trocr_checkpoint/checkpoint-5750-20260612T084901Z-3-001/checkpoint-5750 \
      --base microsoft/trocr-base-handwritten \
      --out models/inkference_trocr_infer
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Files needed for inference; everything else (optimizer, scheduler, rng, scaler,
# trainer_state, training_args) is training-only and dropped.
INFERENCE_FILES = ("model.safetensors", "config.json", "generation_config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained checkpoint dir")
    ap.add_argument("--base", default="microsoft/trocr-base-handwritten",
                    help="base model id the checkpoint was fine-tuned from (for the processor)")
    ap.add_argument("--out", required=True, help="output inference model dir")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Copy inference weights/config from the checkpoint.
    for name in INFERENCE_FILES:
        src = ckpt / name
        if not src.exists():
            raise FileNotFoundError(f"missing {src}")
        shutil.copy2(src, out / name)
        print(f"copied {name}  ({(out / name).stat().st_size / 1e6:.1f} MB)")

    # 2. Write the processor (preprocessor + tokenizer) from the base model.
    from transformers import TrOCRProcessor

    processor = TrOCRProcessor.from_pretrained(args.base)
    processor.save_pretrained(out)
    print(f"wrote processor from {args.base}")

    # 3. Sanity-load the assembled directory.
    from transformers import VisionEncoderDecoderModel

    VisionEncoderDecoderModel.from_pretrained(out)
    TrOCRProcessor.from_pretrained(out)
    print(f"\nOK — loadable inference model at: {out}")
    print("Push with:  huggingface-cli upload <user>/inkference-trocr", out)


if __name__ == "__main__":
    main()
