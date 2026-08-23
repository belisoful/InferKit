#!/usr/bin/env python3
"""Convert the ECCV-16 colorization checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXColorizer` loads a safetensors checkpoint; MLX's `loadArrays` reads
safetensors/npz, not PyTorch `.pth`. The reference checkpoint (richzhang/colorization, `eccv16`)
stores `nn.Sequential` indices (`model1.0.weight` ... `model8.6.weight`, plus `model_out.weight`);
this tool performs the complete, deterministic rename to the module's flat names (`conv1_1`,
`norm1`, `deconv8_1`, `conv8_313`, `out_ab`) — no remap is left to the Swift side.

Two layout details are handled here so the Swift loader can apply its single uniform transpose
(`[out, in, kH, kW]` -> MLX `[out, kH, kW, in]`):
- `model8.0` is a ConvTranspose2d, which PyTorch stores as `[in, out, kH, kW]`; its weight is
  permuted to `[out, in, kH, kW]` on conversion.
- BatchNorm `num_batches_tracked` counters are dropped; `running_mean` / `running_var` are kept
  (the model runs in evaluation mode, which uses them).

The 313 ab bin centers need no separate file: the annealed-mean readout is the checkpoint's own
`model_out` 1x1 convolution (renamed to `out_ab`).

The tool is self-validating: it fails if an expected key is missing and reports any leftover keys.

Usage:
    python convert.py eccv16-9b330a0b.pth colorizer-eccv16.safetensors
    python convert.py eccv16-9b330a0b.pth out.safetensors --half

Get the weights from the reference release:
    https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys

import torch
from safetensors.torch import save_file

# (reference Sequential prefix, [(index, flat name), ...])
BLOCKS = [
    ("model1", [(0, "conv1_1"), (2, "conv1_2"), (4, "norm1")]),
    ("model2", [(0, "conv2_1"), (2, "conv2_2"), (4, "norm2")]),
    ("model3", [(0, "conv3_1"), (2, "conv3_2"), (4, "conv3_3"), (6, "norm3")]),
    ("model4", [(0, "conv4_1"), (2, "conv4_2"), (4, "conv4_3"), (6, "norm4")]),
    ("model5", [(0, "conv5_1"), (2, "conv5_2"), (4, "conv5_3"), (6, "norm5")]),
    ("model6", [(0, "conv6_1"), (2, "conv6_2"), (4, "conv6_3"), (6, "norm6")]),
    ("model7", [(0, "conv7_1"), (2, "conv7_2"), (4, "conv7_3"), (6, "norm7")]),
    ("model8", [(0, "deconv8_1"), (2, "conv8_2"), (4, "conv8_3"), (6, "conv8_313")]),
]
CONV_SUFFIXES = ("weight", "bias")
NORM_SUFFIXES = ("weight", "bias", "running_mean", "running_var")
TRANSPOSED_CONVS = ("deconv8_1",)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected the eccv16 state dict")


def remap(state):
    """Rename every expected key; fail on missing keys, report leftovers."""
    renamed, consumed = {}, set()
    for prefix, layers in BLOCKS:
        for index, name in layers:
            suffixes = NORM_SUFFIXES if name.startswith("norm") else CONV_SUFFIXES
            for suffix in suffixes:
                source = f"{prefix}.{index}.{suffix}"
                if source not in state:
                    raise SystemExit(f"missing expected key: {source}")
                value = state[source]
                if name in TRANSPOSED_CONVS and suffix == "weight":
                    value = value.permute(1, 0, 2, 3)          # [in, out, kH, kW] -> [out, in, kH, kW]
                renamed[f"{name}.{suffix}"] = value
                consumed.add(source)
    if "model_out.weight" not in state:
        raise SystemExit("missing expected key: model_out.weight")
    renamed["out_ab.weight"] = state["model_out.weight"]
    consumed.add("model_out.weight")

    leftovers = [k for k in state if k not in consumed and not k.endswith("num_batches_tracked")]
    if leftovers:
        print(f"note: {len(leftovers)} unconsumed keys: {leftovers[:8]}", file=sys.stderr)
    return renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to the eccv16 .pth")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    parser.add_argument("--passthrough", action="store_true",
                        help="write the checkpoint's own names unchanged, for the siggraph17 release: "
                             "its Sequential layout differs from eccv16's and the translation lives in "
                             "NFKMLXSiggraphColorizer.remapReferenceKey rather than here")
    args = parser.parse_args()

    extracted = extract_state_dict(load_checkpoint(args.input))
    state = extracted if args.passthrough else remap(extracted)

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone() for name, value in state.items()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
