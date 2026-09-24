#!/usr/bin/env python3
"""Cut a Hugging Face release to its first N decoder layers, downloading only the shards they need.

A release too large for this machine at float32 is still checkable at float32 on a prefix of its depth:
the prefix keeps every geometry-specific piece (the widths, the head layout, the rotary, the norms, the
embedding) and drops only repeated blocks. The result is a valid small release that transformers and
the Swift loaders both read unchanged:

    python3 truncate.py Qwen/Qwen3-14B /Volumes/WindowsBoot/InferKit/validation/qwen3-14b-cut4 4

A fifth argument M also cuts an InternVL vision tower (`vision_model.encoder.layers`) to its first M
layers, for a release whose tower alone exceeds the machine at float32 (Sa2VA-26B's InternViT-6B):

    python3 truncate.py ByteDance/Sa2VA-26B ~/.inferkit-validation/sa2va-26b-cut4 4 4

The script writes `config.json` with the layer count set to N (every per-layer list cut to N, nested text configs
included), `model.safetensors.index.json` listing only the kept tensors, and one compact shard per
source shard holding exactly those tensors. Each tensor arrives by an HTTP range request against its
shard, so a 60 GB release costs the first N layers plus the non-layer tensors (the embedding, the final
norm, the head, a vision tower, kept whole). `HF_TOKEN` is sent when set. Requests go through `curl -4`,
since huggingface.co advertises IPv6 addresses this host cannot route.
"""

import json
import os
import re
import struct
import subprocess
import sys

LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
PER_LAYER_LISTS = ("layer_types", "layers_block_type", "hybrid_override_pattern", "sliding_window_pattern_list")


def curl(repo, path, byte_range=None, out=None):
    """The bytes of `path` (or its `byte_range`), written to `out` when given, else returned."""
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    command = ["curl", "-4", "-sfL", "--retry", "3", url]
    token = os.environ.get("HF_TOKEN")
    if token:
        command[1:1] = ["-H", f"Authorization: Bearer {token}"]
    if byte_range is not None:
        command += ["-r", f"{byte_range[0]}-{byte_range[1]}"]
    if out is not None:
        command += ["-o", out]
    return subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout


def fetch(repo, path, out):
    curl(repo, path, out=out)


def write_compact_shard(repo, shard, names, out):
    """A safetensors file at `out` holding `names` from `shard`, each read by its own byte range."""
    length = struct.unpack("<Q", curl(repo, shard, (0, 7)))[0]
    header = json.loads(curl(repo, shard, (8, 8 + length - 1)))
    base = 8 + length
    entries, offset = {}, 0
    for name in sorted(names):
        start, end = header[name]["data_offsets"]
        entries[name] = {"dtype": header[name]["dtype"], "shape": header[name]["shape"],
                         "data_offsets": [offset, offset + end - start]}
        offset += end - start
    encoded = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    part = out + ".part"
    with open(out, "wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for name in sorted(names):
            start, end = header[name]["data_offsets"]
            if end > start:
                curl(repo, shard, (base + start, base + end - 1), out=part)
                with open(part, "rb") as piece:
                    while chunk := piece.read(1 << 24):
                        file.write(chunk)
    if os.path.exists(part):
        os.remove(part)
    return offset


def remote_code_files(repo):
    """The repo's Python modules and small tokenizer tables, which `trust_remote_code` loads."""
    listing = json.loads(subprocess.run(["curl", "-4", "-sfL", f"https://huggingface.co/api/models/{repo}"],
                                        check=True, stdout=subprocess.PIPE).stdout)
    names = [entry["rfilename"] for entry in listing.get("siblings", [])]
    return [name for name in names
            if name.endswith(".py") or name in ("added_tokens.json", "vocab.json", "merges.txt")]


def cut_config(config, layers):
    """`config` with its decoder stack cut to `layers`, in place, nested text configs included."""
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if key in config and isinstance(config[key], int):
            config[key] = layers
    for key in PER_LAYER_LISTS:
        value = config.get(key)
        if isinstance(value, list):
            config[key] = value[:layers]
        elif isinstance(value, str) and key == "hybrid_override_pattern":
            config[key] = value[:layers]
    if isinstance(config.get("num_kv_shared_layers"), int):
        config["num_kv_shared_layers"] = 0
    for nested in ("text_config", "language_config", "llm_config"):
        if isinstance(config.get(nested), dict):
            cut_config(config[nested], layers)


def kept(name, layers, vision_prefixes, vision_layers=None):
    match = LAYER.search(name)
    if match is None:
        return True
    if any(name.startswith(prefix) for prefix in vision_prefixes):
        # A vision tower is kept whole unless its own cut is asked for; only its encoder layers are cut.
        if vision_layers is None or not name.startswith(("vision_model.", "model.vision_model.")):
            return True
        return int(match.group(1)) < vision_layers
    return int(match.group(1)) < layers


def main(argv):
    if len(argv) not in (4, 5):
        print(__doc__)
        return 2
    repo, out, layers = argv[1], os.path.expanduser(argv[2]), int(argv[3])
    vision_layers = int(argv[4]) if len(argv) == 5 else None
    os.makedirs(out, exist_ok=True)
    config_path = os.path.join(out, "config.json")
    fetch(repo, "config.json", config_path)
    config = json.load(open(config_path))
    cut_config(config, layers)
    if vision_layers is not None and isinstance(config.get("vision_config"), dict):
        config["vision_config"]["num_hidden_layers"] = vision_layers
    json.dump(config, open(config_path, "w"), indent=2)

    index_path = os.path.join(out, "model.safetensors.index.json")
    try:
        index = json.loads(curl(repo, "model.safetensors.index.json"))
    except subprocess.CalledProcessError:
        length = struct.unpack("<Q", curl(repo, "model.safetensors", (0, 7)))[0]
        header = json.loads(curl(repo, "model.safetensors", (8, 8 + length - 1)))
        index = {"weight_map": {name: "model.safetensors" for name in header if name != "__metadata__"}}
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "special_tokens_map.json", "tokenizer.model", "chat_template.jinja",
                  "preprocessor_config.json", "video_preprocessor_config.json"):
        try:
            fetch(repo, extra, os.path.join(out, extra))
        except subprocess.CalledProcessError:
            pass
    # A release run through `trust_remote_code` needs its own code and tokenizer tables beside the cut.
    for extra in remote_code_files(repo):
        fetch(repo, extra, os.path.join(out, extra))

    # A vision tower has its own `layers.N`; it is kept whole rather than cut with the decoder, as is
    # Sa2VA's SAM 2 grounding encoder, whose memory attention numbers its own layers.
    vision_prefixes = ("vision_tower.", "model.vision_tower.", "visual.", "model.visual.",
                       "vision_model.", "model.vision_model.", "multi_modal_projector.",
                       "model.multi_modal_projector.", "audio_tower.", "model.audio_tower.",
                       "grounding_encoder.", "model.model.vision_tower.", "model.model.visual.",
                       "model.model.multi_modal_projector.")
    weight_map = {name: shard for name, shard in index["weight_map"].items()
                  if kept(name, layers, vision_prefixes, vision_layers)}
    index["weight_map"] = weight_map
    shards = sorted(set(weight_map.values()))
    total = 0
    for shard in shards:
        names = [name for name, source in weight_map.items() if source == shard]
        print(f"  {shard}: {len(names)} tensors")
        total += write_compact_shard(repo, shard, names, os.path.join(out, shard))
    index["metadata"] = {"total_size": total}
    json.dump(index, open(index_path, "w"), indent=2)
    print(f"{repo}: {len(weight_map)} tensors in {len(shards)} shards, {layers} layers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
