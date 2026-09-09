#!/usr/bin/env python3
"""Fetch a Hugging Face release's config and every tensor's shape WITHOUT downloading its weights.

A safetensors file opens with an 8-byte little-endian header length and a JSON header naming every
tensor's dtype, shape, and byte offsets, so two HTTP range requests per shard yield the complete
tensor inventory of a release at a few hundred kilobytes against hundreds of gigabytes. The result,
`shapes.json` (name -> shape), is what the structural tests hold a module against for a release that
cannot run on the machine (`NFKMLXHybridLanguageTests`, `NFKMLXDeepSeekTests`, the Qwen3-MoE check
in `NFKMLXLanguageModelTests`).

    python3 shapes.py Qwen/Qwen3-30B-A3B ~/.inferkit-validation/qwen3-30b-a3b

writes `config.json`, `model.safetensors.index.json` (when the release is sharded), `shapes.json`,
and `dtypes.json` into the directory. `HF_TOKEN` is sent when set, for a gated repository.
"""

import json
import os
import struct
import sys
import urllib.request


def _request(url, byte_range=None):
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request) as response:
        return response.read()


def header(url):
    """The safetensors JSON header of the file at `url`, by two range requests."""
    length = struct.unpack("<Q", _request(url, (0, 7)))[0]
    return json.loads(_request(url, (8, 8 + length - 1)).decode("utf-8"))


def main(argv):
    # `shapes.py <repo> <out> [subfolder]`. A diffusers pipeline keeps a component (the transformer, the
    # VAE) under a subfolder with a `diffusion_pytorch_model` weight name rather than `model` at the root.
    if len(argv) not in (3, 4):
        print(__doc__)
        return 2
    repo, out = argv[1], os.path.expanduser(argv[2])
    subfolder = argv[3] if len(argv) == 4 else ""
    os.makedirs(out, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/main"
    if subfolder:
        base = f"{base}/{subfolder}"

    config = _request(f"{base}/config.json")
    with open(os.path.join(out, "config.json"), "wb") as handle:
        handle.write(config)

    shards = None
    for weights_name in ("model", "diffusion_pytorch_model"):
        try:
            index = json.loads(_request(f"{base}/{weights_name}.safetensors.index.json"))
            with open(os.path.join(out, "model.safetensors.index.json"), "w") as handle:
                json.dump(index, handle, indent=2)
            shards = sorted(set(index["weight_map"].values()))
            break
        except urllib.error.HTTPError:
            continue
    if shards is None:
        shards = ["diffusion_pytorch_model.safetensors" if subfolder else "model.safetensors"]

    shapes, dtypes = {}, {}
    for shard in shards:
        for name, entry in header(f"{base}/{shard}").items():
            if name == "__metadata__":
                continue
            shapes[name] = entry["shape"]
            dtypes[name] = entry["dtype"]
        print(f"{shard}: {len(shapes)} tensors so far", file=sys.stderr)

    with open(os.path.join(out, "shapes.json"), "w") as handle:
        json.dump(shapes, handle, indent=1, sort_keys=True)
    with open(os.path.join(out, "dtypes.json"), "w") as handle:
        json.dump(dtypes, handle, indent=1, sort_keys=True)
    print(f"{repo}: {len(shapes)} tensors across {len(shards)} file(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
