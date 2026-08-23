#!/usr/bin/env python3
"""Convert a SimpleBaseline / HRNet pose checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXPose` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz,
not PyTorch `.pth`. This tool rewrites a checkpoint into safetensors, preserving PyTorch convolution
layout `[out, in, kH, kW]`. The transposed-convolution weights in the deconvolution head are stored by
PyTorch as `[in, out, kH, kW]`; this tool permutes them to `[out, in, kH, kW]` so the Swift loader's
single uniform transpose (`-> [out, kH, kW, in]`) covers every 4-D weight.

The reference SimpleBaseline checkpoint uses a torchvision ResNet-50 backbone (`conv1`, `bn1`,
`layer1.0.conv1`, ... with bottleneck blocks) and a `deconv_layers` sequence, which `NFKMLXPoseNet` now
mirrors — `NFKMLXPose.remapReferenceKey` handles the positional head and the prefixes the released
model adds, so this tool only rewrites the container. Pass `--list-keys` to print the names.

Usage:
    python convert.py td-hm_res50_coco-256x192.pth --list-keys
    python convert.py td-hm_res50_coco-256x192.pth pose.safetensors

Get the weights from the mmpose model zoo (SimpleBaseline, ResNet-50, COCO 256×192):
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/
        td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth
or the original release:
    https://github.com/microsoft/human-pose-estimation.pytorch

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys
import types

import torch
from safetensors.torch import save_file


def stub_training_packages():
    """Let a checkpoint unpickle without its training framework installed.

    An mmpose release stores its training configuration beside the weights, so unpickling reaches for
    `mmengine`. Nothing on the conversion path reads it, so a module that answers every attribute with a
    throwaway class is enough.
    """
    class Stub(types.ModuleType):
        __path__ = []

        def __getattr__(self, name):
            created = type(name, (dict,), {})
            setattr(self, name, created)
            return created

    class Finder:
        def find_module(self, fullname, path=None):
            return self if fullname.split(".")[0] in {"mmengine", "mmcv", "mmpose"} else None

        def load_module(self, fullname):
            module = Stub(fullname)
            sys.modules[fullname] = module
            return module

    sys.meta_path.append(Finder())


def load_checkpoint(path):
    stub_training_packages()
    # torch 2.6 defaults weights_only=True, which rejects a checkpoint carrying its training config.
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a pose state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a pose .pth")
    parser.add_argument("output", nargs="?", help="path to write .safetensors")
    parser.add_argument("--list-keys", action="store_true", help="print parameter names and exit")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    state = extract_state_dict(load_checkpoint(args.input))

    if args.list_keys:
        for name in sorted(state):
            print(f"{name}\t{tuple(state[name].shape)}")
        return

    if not args.output:
        raise SystemExit("output path required unless --list-keys")

    dtype = torch.float16 if args.half else torch.float32
    tensors = {}
    for name, value in state.items():
        if not value.is_floating_point() or name.endswith("num_batches_tracked"):
            continue
        # Permute transposed-conv weights [in, out, kH, kW] -> [out, in, kH, kW].
        if "deconv" in name and name.endswith("weight") and value.dim() == 4:
            value = value.permute(1, 0, 2, 3)
        tensors[name] = value.to(dtype).contiguous().clone()
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
