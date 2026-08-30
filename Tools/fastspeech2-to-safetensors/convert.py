#!/usr/bin/env python3
"""Convert a transformers FastSpeech2Conformer checkpoint to safetensors for InferKitMLX

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.
(NFKMLXFastSpeech2Net).

Names pass through — the module keys are the checkpoint's — and the batch-norm step counters are
dropped (they are bookkeeping, not parameters). The Swift loader transposes the convolution layouts.

Usage:
    python convert.py pytorch_model.bin fastspeech2-conformer.safetensors
    python convert.py pytorch_model.bin --list-keys

Weights: https://huggingface.co/espnet/fastspeech2_conformer (the LJSpeech release; its vocab.json is
the phoneme table the voice needs).

Requires: torch, safetensors
"""

import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="the released pytorch_model.bin")
    parser.add_argument("output", nargs="?", help="the safetensors to write")
    parser.add_argument("--list-keys", action="store_true", help="print the checkpoint keys and exit")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # The paired release nests the acoustic model under `model.`; take that half when present.
    if any(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items()
                 if key.startswith("model.")}

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return
    if not args.output:
        parser.error("output path required unless --list-keys")

    kept = {key: value.contiguous() for key, value in state.items()
            if not key.endswith("num_batches_tracked")}
    save_file(kept, args.output)
    print(f"wrote {args.output}: {len(kept)} tensors")


if __name__ == "__main__":
    main()
