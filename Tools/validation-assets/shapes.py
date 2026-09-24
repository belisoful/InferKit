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
import socket
import struct
import sys
import urllib.request


# `huggingface.co` advertises sixteen IPv6 addresses ahead of its IPv4 ones. On a host with no IPv6
# route, Python tries each in turn and waits out the connect timeout on every one, so a request that
# curl answers in under a second takes minutes; the shard walk below never finishes. Ordering the
# IPv4 addresses first restores the sub-second path and leaves an IPv6-only host working, because the
# IPv6 addresses stay in the list behind them.
_resolve = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    return sorted(_resolve(*args, **kwargs), key=lambda entry: entry[0] != socket.AF_INET)


socket.getaddrinfo = _ipv4_first


def _request(url, byte_range=None):
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    # A shard resolves to the LFS CDN, and a connect to it can hang indefinitely. urllib applies no
    # timeout of its own, so a stalled connect blocks the whole run rather than failing; one retry
    # covers a CDN edge that refuses the first connection.
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        # A status the server chose is its answer, not a flaky connection: retrying a 404 three times
        # only hides it, and the caller below reads one to decide which weight name a release uses.
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = error
    raise SystemExit(f"{url}: {last}")


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
        # An unsharded release names its single file either way, and the subfolder is no longer the
        # tell: FLUX.2-small-decoder keeps `config.json` AND `diffusion_pytorch_model.safetensors` at
        # the repository root. Try the likely name first and fall back rather than 404 on a repository
        # that is right there.
        preferred = "diffusion_pytorch_model.safetensors" if subfolder else "model.safetensors"
        other = "model.safetensors" if subfolder else "diffusion_pytorch_model.safetensors"
        shards = [preferred]
        try:
            header(f"{base}/{preferred}")
        except urllib.error.HTTPError:
            shards = [other]

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
