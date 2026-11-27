#!/usr/bin/env python3
"""Builds a Wan-Dancer-14B test-dev release: the release's own file names, key names, and stored dtypes, at a small
geometry with seeded random weights.

The release is 85 GB (two 14B transformers at bfloat16, umT5-XXL, the Wan 2.1 autoencoder, and open-clip's
XLM-RoBERTa ViT-H/14), so the port's end-to-end parity is measured on this directory: the fork's own
`WanVideoPipeline` and the InferKitMLX generator both read it, and `run_reference.py wan_dancer_pipeline
--checkpoint <out>` records the reference.

Every component is built by the fork's own class and saved as the release stores it:

- the global and local transformers (`diffsynth.models.wan_video_dit.WanModel` with `dit_model_type=1`'s
  image-to-video geometry and every Wan-Dancer module the released scripts enable) from the release's own
  `config.json` with the size keys reduced (48 wide over 2 heads, 28 blocks so that all eight music injectors and
  the skipped block 9 exist, ffn 96, text 32), each its own seed, as `global_model.safetensors` and
  `local_model.safetensors` at bfloat16; the reduced `text_dim` is added to `config.json`;
- the Wan 2.1 autoencoder (`wan_video_vae.VideoVAE_`) at its released width, as `Wan2.1_VAE.pth` at float32;
- umT5 (`wan_video_text_encoder.WanTextEncoder`) at 32 wide over the released 256,384-token vocabulary, as
  `models_t5_umt5-xxl-enc-bf16.pth` at bfloat16; the release's `google/umt5-xxl` tokenizer is copied;
- CLIP (`wan_video_image_encoder.clip_xlm_roberta_vit_h_14`) with its vision tower at the released 1280 width,
  16 heads, and 224-pixel input (the transformer reads 257 tokens of width 1280) over 3 blocks, and its text tower
  at 32 wide over 1 block, as `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` at float32.

What stays at the released value: the channel counts, the music encoder (35 → 256, 4 heads, two layers), the
injected blocks, the rotary table's length, the image tokens, and the vocabulary.

Usage (the `wandancer` oracle env, with `IK_WAN_DANCER_SRC` set):
    make_testdev.py --release <dir of the release's small files> --out <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import torch

TRANSFORMER = {"dim": 48, "num_heads": 2, "num_layers": 28, "ffn_dim": 96}
TEXT_DIM = 32
TEXT_ENCODER = {"vocab": 256384, "dim": 32, "dim_attn": 32, "dim_ffn": 64, "num_heads": 2, "num_layers": 2,
                "num_buckets": 32, "shared_pos": False, "dropout": 0.0}
CLIP = {"vision_layers": 3, "text_dim": 32, "text_heads": 2, "text_layers": 1, "embed_dim": 32}
DANCER = {"has_image_input": True, "patch_size": [1, 2, 2], "enable_music_inject": True,
          "music_inject_layers": [0, 4, 8, 12, 16, 20, 24, 27], "enable_refimage": True, "has_ref_conv": True,
          "enable_global": True, "enable_dynamicfps": True, "enable_unimodel": True}
SEEDS = {"global": 501, "local": 502, "text_encoder": 503, "vae": 504, "clip": 505}
SCALE_SUFFIXES = (".gamma", "norm.weight", "norm1.weight", "norm2.weight", "norm3.weight", "norm_q.weight",
                  "norm_k.weight", "norm_k_img.weight", "pre_norm.weight", "post_norm.weight", "proj.0.weight",
                  "proj.4.weight")


def fork():
    source = os.environ.get("IK_WAN_DANCER_SRC")
    if not source:
        raise SystemExit("set IK_WAN_DANCER_SRC")
    sys.path.insert(0, source)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dancer_shims  # noqa: F401


def randomized(model, seed, scale):
    generator = torch.Generator().manual_seed(seed)
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key]
        if not tensor.is_floating_point():
            continue
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        state[key] = (1 + 0.1 * noise if key.endswith(SCALE_SUFFIXES) else scale * noise).to(tensor.dtype)
    model.load_state_dict(state)
    return model


def transformer_arguments(config):
    keys = {k: config[k] for k in ("dim", "in_dim", "ffn_dim", "out_dim", "freq_dim", "eps", "num_heads", "num_layers")}
    keys["text_dim"] = config["text_dim"]
    keys.update(DANCER)
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, help="the release's small files, in its layout")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    release, out = arguments.release, arguments.out
    os.makedirs(out, exist_ok=True)
    fork()
    from safetensors.torch import save_file
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.models.wan_video_image_encoder import clip_xlm_roberta_vit_h_14
    from diffsynth.models.wan_video_text_encoder import WanTextEncoder
    from diffsynth.models.wan_video_vae import VideoVAE_

    with open(os.path.join(release, "config.json")) as handle:
        config = json.load(handle)
    for key, value in TRANSFORMER.items():
        if key not in config:
            raise SystemExit(f"config.json has no {key}; the release's config changed")
        config[key] = value
    config["text_dim"] = TEXT_DIM
    with open(os.path.join(out, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    for stage in ("global", "local"):
        transformer = randomized(WanModel(**transformer_arguments(config)).float(), SEEDS[stage], 0.05)
        # The release stores no rotary buffer; the fork rebuilds it as the model is constructed.
        save_file({k: v.to(torch.bfloat16).contiguous() for k, v in transformer.state_dict().items()
                   if not k.endswith("rotary.freqs")}, os.path.join(out, f"{stage}_model.safetensors"))

    vae = randomized(VideoVAE_(z_dim=16).float(), SEEDS["vae"], 0.05)
    torch.save(vae.state_dict(), os.path.join(out, "Wan2.1_VAE.pth"))

    text_encoder = randomized(WanTextEncoder(**TEXT_ENCODER).float(), SEEDS["text_encoder"], 0.1)
    torch.save({k: v.to(torch.bfloat16) for k, v in text_encoder.state_dict().items()},
               os.path.join(out, "models_t5_umt5-xxl-enc-bf16.pth"))
    shutil.copytree(os.path.join(release, "google"), os.path.join(out, "google"), dirs_exist_ok=True)

    clip, _ = clip_xlm_roberta_vit_h_14(pretrained=False, return_transforms=True, return_tokenizer=False,
                                        dtype=torch.float32, device="cpu", **CLIP)
    randomized(clip.float(), SEEDS["clip"], 0.05)
    torch.save(clip.state_dict(), os.path.join(out, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))

    manifest = {}
    for root, _, files in os.walk(out):
        for file in sorted(files):
            if file == "testdev-manifest.json":
                continue
            path = os.path.join(root, file)
            with open(path, "rb") as handle:
                manifest[os.path.relpath(path, out)] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(out, "testdev-manifest.json"), "w") as handle:
        json.dump({"geometry": {"transformer": TRANSFORMER, "text_encoder": TEXT_ENCODER, "clip": CLIP},
                   "seeds": SEEDS, "sha256": manifest}, handle, indent=2, sort_keys=True)
    print(json.dumps({k: v[:16] for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    main()
