#!/usr/bin/env python3
"""Convert a Depth Anything V2 checkpoint to safetensors for InferKitMLX (NFKMLXDepthAnything).

MLX loads safetensors/npz, not PyTorch `.pth`. This tool rewrites the release into safetensors,
preserving the reference parameter names (encoder under `pretrained.*`, DPT head under `depth_head.*`)
and PyTorch convolution layout `[out, in, kH, kW]`; the Swift loader transposes convolution weights to
MLX's channels-last layout at load.

The DINOv2 + DPT key layout is intricate, so this tool is **self-validating**: every checkpoint key is
matched against the layout `NFKMLXDepthAnythingNet` expects. Unmatched keys and missing sections are
reported, so a mismatch with a model variant is caught before use (feed the report to a `remap` in
`NFKMLXDepthAnything.loadWeights`).

Usage:
    python convert.py depth_anything_v2_vits.pth depth_anything_v2_vits.safetensors
    python convert.py in.pth out.safetensors --half
    python convert.py in.pth out.safetensors --validate-only    # report key coverage, write nothing

Weights (Small, ~99 MB): https://huggingface.co/depth-anything/Depth-Anything-V2-Small

Requires: torch, safetensors
"""

import argparse
import re
import sys

import torch
from safetensors.torch import save_file

# Key patterns NFKMLXDepthAnythingNet exposes (Small/Base/Large share this structure).
EXPECTED_PATTERNS = [
    r"pretrained\.cls_token",
    r"pretrained\.pos_embed",
    r"pretrained\.mask_token",                                    # present in DINOv2, unused by the head
    r"pretrained\.patch_embed\.proj\.(weight|bias)",
    r"pretrained\.blocks\.\d+\.norm[12]\.(weight|bias)",
    r"pretrained\.blocks\.\d+\.attn\.(qkv|proj)\.(weight|bias)",
    r"pretrained\.blocks\.\d+\.ls[12]\.gamma",
    r"pretrained\.blocks\.\d+\.mlp\.fc[12]\.(weight|bias)",
    r"pretrained\.norm\.(weight|bias)",
    r"depth_head\.projects\.\d+\.(weight|bias)",
    r"depth_head\.resize_layers\.\d+\.(weight|bias)",
    r"depth_head\.scratch\.layer\d_rn\.weight",
    r"depth_head\.scratch\.refinenet\d\.resConfUnit[12]\.conv[12]\.(weight|bias)",
    r"depth_head\.scratch\.refinenet\d\.out_conv\.(weight|bias)",
    r"depth_head\.scratch\.output_conv1\.(weight|bias)",
    r"depth_head\.scratch\.output_conv2\.[0-2]\.(weight|bias)",
]
EXPECTED = [re.compile(p + r"$") for p in EXPECTED_PATTERNS]
REQUIRED_SECTIONS = ("pretrained.patch_embed", "pretrained.blocks.0", "depth_head.scratch.output_conv2")


def load_state_dict(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not (isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values())):
        raise SystemExit("unrecognized checkpoint: expected a state dict of tensors")
    return obj


def validate(state):
    unmatched = [k for k in state if not any(p.match(k) for p in EXPECTED)]
    missing = [s for s in REQUIRED_SECTIONS if not any(k.startswith(s) for k in state)]
    print(f"{len(state)} tensors; {len(state) - len(unmatched)} match the expected layout")
    if missing:
        print(f"MISSING expected sections: {missing}", file=sys.stderr)
    if unmatched:
        print(f"{len(unmatched)} unmatched keys (map these with a remap):", file=sys.stderr)
        for key in unmatched[:40]:
            print(f"  {key}", file=sys.stderr)
        if len(unmatched) > 40:
            print(f"  … and {len(unmatched) - 40} more", file=sys.stderr)
    return not missing and not unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to the Depth Anything V2 .pth")
    parser.add_argument("output", nargs="?", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    parser.add_argument("--validate-only", action="store_true", help="report key coverage, write nothing")
    args = parser.parse_args()

    state = load_state_dict(args.input)
    clean = validate(state)
    if args.validate_only:
        sys.exit(0 if clean else 1)
    if not args.output:
        raise SystemExit("output path required (or pass --validate-only)")
    if not clean:
        print("proceeding despite mismatches; adjust with a remap in loadWeights", file=sys.stderr)

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone() for name, value in state.items()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
