#!/usr/bin/env python3
"""Builds a Wan 2.2 S2V-14B test-dev release: the release's own original Wan layout, file names, key names,
and stored dtypes, at a small geometry with seeded random weights.

The release is 49 GB (the 14B transformer at bfloat16, umT5-XXL, the Wan 2.1 autoencoder, and
wav2vec2-large-xlsr-53-english), so the port's end-to-end parity is measured on this directory:
Wan-Video/Wan2.2's own `WanS2V.generate` and the InferKitMLX generator both read it, and
`run_reference.py wan_s2v_pipeline --checkpoint <out>` records the reference.

Every component is built by its own reference class and saved as the release stores it:

- the transformer (`wan.modules.s2v.WanModel_S2V`) from the release's own `config.json` with the size keys
  reduced (48 wide over 2 heads, 2 blocks, ffn 96, text 32, the audio width 32 of the small wav2vec2, audio
  injected after both blocks), its state dict in `diffusion_pytorch_model.safetensors` at bfloat16;
- the Wan 2.1 autoencoder (`wan.modules.vae2_1.WanVAE_` as `_video_vae` configures it) at its released width, as `Wan2.1_VAE.pth` at
  float32;
- umT5 (`wan.modules.t5.T5Encoder`) at 32 wide over the released 256,384-token vocabulary, as
  `models_t5_umt5-xxl-enc-bf16.pth` at bfloat16; the release's `google/umt5-xxl` tokenizer is copied;
- wav2vec2 (transformers' `Wav2Vec2ForCTC`, stable layer norm as released) at 32 wide over the released 24 layers, saved by
  `save_pretrained` (the weight norm renamed to the release's `weight_g`/`weight_v`) under `wav2vec2-large-xlsr-53-english/` beside the release's own processor files.

What stays at the released value: the channel counts the pipeline wires (16 latent, `cond_dim`), the frame
packer's buckets, the token counts, the zero timestep, AdaIN, the wav2vec2 depth (the audio encoder weighs a
fixed 25 hidden states), and the vocabulary.

Usage (the `videoxfun` oracle env, with `IK_WAN22_SRC` set):
    make_testdev.py --release <dir of the release's small files> --out <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import torch

TRANSFORMER = {"dim": 48, "num_heads": 2, "num_layers": 2, "ffn_dim": 96, "audio_dim": 32,
               "audio_inject_layers": [0, 1]}
TEXT_ENCODER = {"vocab": 256384, "dim": 32, "dim_attn": 32, "dim_ffn": 64, "num_heads": 2, "num_layers": 2,
                "num_buckets": 32, "shared_pos": False, "dropout": 0.0}
WAV2VEC2 = {"hidden_size": 32, "num_hidden_layers": 24, "num_attention_heads": 2, "intermediate_size": 64,
            "conv_dim": [32] * 7, "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2}
TEXT_DIM = 32
VAE = {"dim": 96, "z_dim": 16, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_scales": [],
       "temperal_downsample": [False, True, True], "dropout": 0.0}

SEEDS = {"transformer": 401, "text_encoder": 402, "vae": 403, "wav2vec2": 404}
SCALE_SUFFIXES = (".gamma", "norm.weight", "norm1.weight", "norm2.weight", "norm3.weight", "norm_q.weight",
                  "norm_k.weight", "layer_norm.weight")
# The release's wav2vec2 stores its positional convolution's weight norm under the older names.
WEIGHT_NORM_NAMES = {"original0": "weight_g", "original1": "weight_v"}
WAV2VEC2_DIRECTORY = "wav2vec2-large-xlsr-53-english"
WAV2VEC2_FILES = ["preprocessor_config.json", "vocab.json", "special_tokens_map.json", "alphabet.json"]


def wan():
    source = os.environ.get("IK_WAN22_SRC")
    if not source:
        raise SystemExit("set IK_WAN22_SRC")
    sys.path.insert(0, source)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wan_shims
    wan_shims.install()
    return source


def released_name(key):
    for current, released in WEIGHT_NORM_NAMES.items():
        if key.endswith("parametrizations.weight." + current):
            return key[: -len("parametrizations.weight." + current)] + released
    return key


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, help="the release's small files, in its layout")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    release, out = arguments.release, arguments.out
    os.makedirs(out, exist_ok=True)
    wan()
    from safetensors.torch import load_file, save_file
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
    from wan.modules.s2v.model_s2v import WanModel_S2V
    from wan.modules.t5 import T5Encoder
    from wan.modules.vae2_1 import WanVAE_

    with open(os.path.join(release, "config.json")) as handle:
        config = json.load(handle)
    for key, value in TRANSFORMER.items():
        if key not in config:
            raise SystemExit(f"config.json has no {key}; the release's config changed")
        config[key] = value
    config["text_dim"] = TEXT_DIM
    with open(os.path.join(out, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    keys = {k: v for k, v in config.items() if not k.startswith("_")}
    transformer = WanModel_S2V(**keys)
    randomized(transformer.float(), SEEDS["transformer"], 0.05)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in transformer.state_dict().items()},
              os.path.join(out, "diffusion_pytorch_model.safetensors"))

    vae = WanVAE_(**VAE)
    randomized(vae.float(), SEEDS["vae"], 0.05)
    torch.save(vae.state_dict(), os.path.join(out, "Wan2.1_VAE.pth"))

    text_encoder = T5Encoder(**TEXT_ENCODER)
    randomized(text_encoder.float(), SEEDS["text_encoder"], 0.1)
    torch.save({k: v.to(torch.bfloat16) for k, v in text_encoder.state_dict().items()},
               os.path.join(out, "models_t5_umt5-xxl-enc-bf16.pth"))
    shutil.copytree(os.path.join(release, "google"), os.path.join(out, "google"), dirs_exist_ok=True)

    with open(os.path.join(release, WAV2VEC2_DIRECTORY, "config.json")) as handle:
        audio_config = json.load(handle)
    audio_config.update(WAV2VEC2)
    torch.manual_seed(SEEDS["wav2vec2"])
    wav2vec2 = randomized(Wav2Vec2ForCTC(Wav2Vec2Config(**audio_config)).float(), SEEDS["wav2vec2"], 0.1)
    audio_directory = os.path.join(out, WAV2VEC2_DIRECTORY)
    wav2vec2.save_pretrained(audio_directory)
    weights = os.path.join(audio_directory, "model.safetensors")
    state = load_file(weights)
    save_file({released_name(k): v for k, v in state.items()}, weights, metadata={"format": "pt"})
    for name in WAV2VEC2_FILES:
        if os.path.exists(os.path.join(release, WAV2VEC2_DIRECTORY, name)):
            shutil.copyfile(os.path.join(release, WAV2VEC2_DIRECTORY, name), os.path.join(audio_directory, name))
    with open(os.path.join(release, WAV2VEC2_DIRECTORY, "preprocessor_config.json")) as handle:
        processor = json.load(handle)
    processor["processor_class"] = "Wav2Vec2Processor"
    with open(os.path.join(audio_directory, "preprocessor_config.json"), "w") as handle:
        json.dump(processor, handle, indent=2)

    manifest = {}
    for root, _, files in os.walk(out):
        for file in sorted(files):
            if file == "testdev-manifest.json":
                continue
            path = os.path.join(root, file)
            with open(path, "rb") as handle:
                manifest[os.path.relpath(path, out)] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(out, "testdev-manifest.json"), "w") as handle:
        json.dump({"geometry": {"transformer": TRANSFORMER, "text_encoder": TEXT_ENCODER, "wav2vec2": WAV2VEC2},
                   "seeds": SEEDS, "sha256": manifest}, handle, indent=2, sort_keys=True)
    print(json.dumps({k: v[:16] for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    main()
