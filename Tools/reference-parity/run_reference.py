#!/usr/bin/env python3
"""Run a model's reference implementation and record its input and output for InferKit to compare against.

Visual plausibility is not reference quality. A model can load every parameter, run end to end, and still
be wrong — SAM tracked its subject convincingly while missing its entire input normalization, and Depth
Anything produced a correct-looking map because its output is min-max normalized. The only check that
catches that class of bug is running the reference on the same input and comparing numerically.

Protocol: this script writes ONE safetensors file per model containing both

    input_image   the raw RGB plate, [H, W, 3] float32 in 0...1 (no preprocessing applied)
    output        the reference implementation's result

InferKit's `NFKMLXReferenceParityTests` reads the same file, feeds `input_image` to the MLX model, and
compares against `output`. Sharing the input tensor rather than an image file keeps image decoding out of
the comparison, while each side still applies its own preprocessing — so a missing normalization shows up
as a mismatch rather than hiding.

Usage:
    python run_reference.py clip   out/clip-reference.safetensors
    python run_reference.py depth  out/depth-reference.safetensors

A training objective is validated the same way. `zero_dce_losses` records the tensors the reference
scored alongside its four loss values, so both sides score identical inputs:

    python run_reference.py zero_dce_losses out/zero-dce-losses.safetensors --size 64

Requires: torch, safetensors, transformers.
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file


def deterministic_image(height=224, width=224, seed=7):
    """A fixed pseudo-random RGB plate in 0...1. Structured enough to exercise a real forward, and
    identical on both sides of the comparison."""
    generator = np.random.default_rng(seed)
    # Round the block grid UP so the result is exactly height × width — rounding down silently returns a
    # smaller plate, which would reintroduce a resize on the reference side and spoil an isolation test.
    blocks = ((height + 7) // 8, (width + 7) // 8)
    base = generator.random((blocks[0], blocks[1], 3), dtype=np.float32)
    # Smooth blocks give the model spatial structure rather than pixel noise.
    image = np.repeat(np.repeat(base, 8, axis=0), 8, axis=1)
    return np.ascontiguousarray(image[:height, :width, :])


def subject_image(height=320, width=320, seed=7):
    """A plate with an actual subject: a textured ellipse on a smooth background.

    A saliency or matting model has nothing to find in unstructured blocks — the reference's own map
    comes out uniformly near zero, and comparing two near-constant maps measures nothing. This gives
    both sides a foreground to separate.
    """
    generator = np.random.default_rng(seed)
    rows = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    columns = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    background = 0.25 + 0.3 * rows + 0.15 * columns * np.array([1.0, 0.6, 0.2], dtype=np.float32)

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    inside = (((y - height / 2) / (height * 0.3)) ** 2 + ((x - width / 2) / (width * 0.22)) ** 2) <= 1.0
    texture = generator.random((height, width, 3), dtype=np.float32) * 0.2 + 0.75
    image = np.where(inside[..., None], texture, background)
    return np.ascontiguousarray(np.clip(image, 0, 1).astype(np.float32))


def run_clip(image):
    """OpenAI CLIP ViT-B/32 image embedding, L2-normalized, through transformers."""
    from transformers import CLIPModel, CLIPImageProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # The processor expects HWC uint8-like input; hand it the raw plate and let it normalize.
    inputs = processor(images=(image * 255).astype(np.uint8), return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].contiguous()


# The released encoder sizes, as `DepthAnythingV2` itself configures them.
_DEPTH_VARIANTS = {
    "Small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "Base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "Large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _depth_model(checkpoint):
    """Build `DepthAnythingV2` from the authors' own package and load a released `.pth`.

    This drove `transformers` until that package stopped registering the `depth_anything` model type,
    at which point every size raised `KeyError` and the oracle could no longer be re-run at all. The
    original repository is the durable source — and the pattern every other model here already uses.
    IK_REF_SRC holds a `depth_anything_v2/` directory (the real package, so its own relative imports
    resolve); its parent goes on the path rather than being synthesized.
    """
    sys.path.insert(0, _reference_source())
    from depth_anything_v2.dpt import DepthAnythingV2

    variant = os.environ.get("IK_DEPTH_VARIANT", "Small")
    model = DepthAnythingV2(**_DEPTH_VARIANTS[variant])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def _depth_input(image):
    """ImageNet normalization at the plate's own size. The port runs a fixed 518×518, so feeding a
    518×518 plate keeps every resize out of the comparison."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(((image - mean) / deviation).transpose(2, 0, 1)).unsqueeze(0)


def run_depth(image, checkpoint):
    """Depth Anything V2 relative depth, min-max normalized to 0...1 (as InferKit emits it)."""
    model = _depth_model(checkpoint)
    with torch.no_grad():
        depth = model(_depth_input(image))[0]                  # [h, w]
    depth = (depth - depth.min()) / torch.clamp(depth.max() - depth.min(), min=1e-6)
    return depth.contiguous()


def run_depth_encoder(image, checkpoint):
    """The encoder's hooked feature maps, to localize a mismatch to encoder vs DPT head.

    Comparing only final outputs says *that* two implementations differ, not *where*. These are the
    seam between the DINOv2 backbone and the DPT head — and the layer whose missing final LayerNorm
    was the original Depth Anything bug.
    """
    model = _depth_model(checkpoint)
    with torch.no_grad():
        features = model.pretrained.get_intermediate_layers(
            _depth_input(image), model.intermediate_layer_idx[model.encoder], return_class_token=True)
    return features[0][0][0].contiguous()                       # [tokens, C], class token excluded


def run_segformer(image):
    """SegFormer-B0 (ADE20k) class logits at the decode head's native quarter resolution."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).eval()
    inputs = processor(images=(image * 255).astype(np.uint8), return_tensors="pt", do_resize=False)
    with torch.no_grad():
        logits = model(**inputs).logits                         # [1, classes, h/4, w/4]
    return logits[0].permute(1, 2, 0).contiguous()              # [h/4, w/4, classes] to match NHWC


def run_swinir(image, checkpoint):
    """SwinIR classical SR, using the reference `network_swinir.py` from the SwinIR repository.

    IK_SWINIR_SCALE picks the release (4 by default; 3 is the non-power-of-two upsampler).

    Set IK_SWINIR_SRC to the directory holding a downloaded `network_swinir.py` (it is a single
    self-contained file; it needs `timm`).
    """
    import os
    import sys

    sys.path.insert(0, os.environ.get("IK_SWINIR_SRC", "."))
    from network_swinir import SwinIR

    # IK_SWINIR_SCALE selects the release. The upsampler differs by more than a factor: x4 runs two
    # ×2 pixel-shuffle stages, x3 runs one ×3 stage, so a checkpoint fits only its own scale.
    scale = int(os.environ.get("IK_SWINIR_SCALE", "4"))
    # IK_SWINIR_LIGHT selects the lightweight release, which is narrower, shallower, and reconstructs
    # through `pixelshuffledirect` — one convolution and one shuffle, with no surrounding convolutions.
    light = os.environ.get("IK_SWINIR_LIGHT", "") == "1"
    if light:
        model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                       depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                       mlp_ratio=2, upsampler="pixelshuffledirect", resi_connection="1conv").eval()
    else:
        model = SwinIR(upscale=scale, in_chans=3, img_size=48, window_size=8, img_range=1.0,
                       depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv").eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)      # [1, 3, H, W]
    with torch.no_grad():
        output = model(tensor)
    return output[0].permute(1, 2, 0).contiguous()                      # [scale·H, scale·W, 3] NHWC


def run_convtasnet(image, checkpoint):
    """Asteroid Conv-TasNet separation of a deterministic mono waveform, `[speakers, samples]`.

    The plate argument is unused: this is an audio model, so the record's `input_image` carries the
    waveform as a `[1, samples, 1]` tensor that the Swift side reads back.
    """
    from asteroid.models import ConvTasNet

    model = ConvTasNet.from_pretrained(checkpoint).eval()
    samples = 16000
    time = np.arange(samples, dtype=np.float32) / 16000.0
    # Two tones plus a little noise: enough structure for separation to do something measurable.
    generator = np.random.default_rng(3)
    wave = (0.4 * np.sin(2 * np.pi * 220 * time) + 0.3 * np.sin(2 * np.pi * 587 * time)
            + 0.02 * generator.standard_normal(samples).astype(np.float32)).astype(np.float32)
    globals()["_extra"] = {"prompt": torch.tensor(prompt_ids, dtype=torch.int32).contiguous(),
                           "waveform": torch.from_numpy(wave).contiguous()}
    with torch.no_grad():
        estimates = model(torch.from_numpy(wave).reshape(1, 1, -1))
    return estimates[0].contiguous()                            # [speakers, samples]


def _capture_init(init):
    """The reference decorator that records a module's construction arguments."""
    import functools

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)
    return __init__


def _center_trim(tensor, reference):
    """The reference's `center_trim`, trimming a tensor to a length around its middle."""
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def _import_reference(source, package, name, siblings=(), injected=None):
    """Import one reference file from `source` as a member of a synthetic `package`.

    A reference module reaches for its siblings (`from .utils import capture_init`), and those siblings
    import dependencies the model itself never uses — demucs's `utils` pulls in `diffq`. `siblings` names
    files to load from `source` alongside the target; `injected` supplies the rest as ready-made modules,
    so only what the forward pass needs has to exist.
    """
    import importlib.util
    import os
    import types

    shell = types.ModuleType(package)
    shell.__path__ = [source]
    sys.modules[package] = shell
    for member, module in (injected or {}).items():
        sys.modules[f"{package}.{member}"] = module

    def load(member):
        spec = importlib.util.spec_from_file_location(f"{package}.{member}",
                                                      os.path.join(source, f"{member}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package}.{member}"] = module
        spec.loader.exec_module(module)
        return module

    for member in siblings:
        load(member)
    return load(name)


def run_demucs(image, checkpoint):
    """Demucs v2 music separation of a deterministic stereo waveform, `[stems, channels, samples]`.

    Set IK_DEMUCS_SRC to a directory holding `model.py` from the demucs v2 branch (it needs `julius`).
    The record's `waveform` carries the `[channels, samples]` mix the Swift side reads back.

    The reference pads to `valid_length` and center-trims the result, because a `context` of 3 leaves the
    decoder longer than the mix it started from. `NFKMLXDemucsNet.separate` does both internally, so the
    two are compared over the same window.
    """
    import os
    import types
    import torch.nn.functional as F

    utils = types.ModuleType("demucs_v2.utils")
    utils.capture_init, utils.center_trim = _capture_init, _center_trim
    module = _import_reference(os.environ.get("IK_DEMUCS_SRC", "."), "demucs_v2", "model",
                               injected={"utils": utils})
    model = module.Demucs(sources=["drums", "bass", "other", "vocals"], channels=64).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    samples = 22050
    time = np.arange(samples, dtype=np.float32) / 44100.0
    generator = np.random.default_rng(5)
    # A chord plus a periodic transient: harmonic content for the tonal stems, an onset for the drums.
    left = (0.3 * np.sin(2 * np.pi * 110 * time) + 0.2 * np.sin(2 * np.pi * 330 * time)
            + 0.2 * np.sin(2 * np.pi * 440 * time))
    right = (0.3 * np.sin(2 * np.pi * 110 * time + 0.4) + 0.25 * np.sin(2 * np.pi * 554 * time))
    click = ((np.arange(samples) % 5512) < 40).astype(np.float32) * 0.3
    wave = np.stack([left + click, right + click]).astype(np.float32)
    wave += 0.01 * generator.standard_normal(wave.shape).astype(np.float32)

    length = wave.shape[-1]
    padded = F.pad(torch.from_numpy(wave)[None], (0, model.valid_length(length) - length))
    with torch.no_grad():
        estimates = model(padded)
    globals()["_extra"] = {"prompt": torch.tensor(prompt_ids, dtype=torch.int32).contiguous(),
                           "waveform": torch.from_numpy(wave).contiguous()}
    return _center_trim(estimates, length)[0].contiguous()      # [stems, channels, samples]


def run_htdemucs(image, checkpoint):
    """Hybrid Transformer Demucs (Demucs v4) separation of a deterministic stereo clip.

    `demucs` 4.0.1 is installed and `demucs.states.load_model` builds the network straight from the
    released checkpoint, so nothing is vendored here. It predates torch 2.6, so `torch.load` has to be
    patched to `weights_only=False` before the archive will open.

    `use_train_segment` is turned off so the record runs at the clip's own length. It is a padding
    policy, not a weight or a shape: with it on the reference zero-pads every input up to the 7.8-second
    training segment, which would make the record a hundred megabytes and the comparison no sharper.

    The record carries the branch seams as well as the waveform, so a failure says which half is wrong:
    `spectrogram` is the complex-as-channels input, `bottleneck_in`/`bottleneck_out` bracket the
    cross-transformer, and `freq_out`/`time_out` are the two branches' final predictions.
    """
    import functools

    torch.load = functools.partial(torch.load, weights_only=False)
    import demucs.states

    model = demucs.states.load_model(checkpoint).eval()
    model.use_train_segment = False

    samples = 44100
    time = np.arange(samples, dtype=np.float32) / 44100.0
    generator = np.random.default_rng(11)
    left = (0.3 * np.sin(2 * np.pi * 110 * time) + 0.2 * np.sin(2 * np.pi * 330 * time)
            + 0.2 * np.sin(2 * np.pi * 440 * time))
    right = (0.3 * np.sin(2 * np.pi * 110 * time + 0.4) + 0.25 * np.sin(2 * np.pi * 554 * time))
    click = ((np.arange(samples) % 5512) < 40).astype(np.float32) * 0.3
    wave = np.stack([left + click, right + click]).astype(np.float32)
    wave += 0.01 * generator.standard_normal(wave.shape).astype(np.float32)
    mix = torch.from_numpy(wave)[None]

    seams = {}
    handles = [
        model.encoder[3].register_forward_hook(
            lambda m, i, o: seams.__setitem__("bottleneck_in", o.detach())),
        model.crosstransformer.register_forward_hook(
            lambda m, i, o: seams.__setitem__("bottleneck_out", o[0].detach())),
        model.decoder[3].register_forward_hook(
            lambda m, i, o: seams.__setitem__("freq_out", o[0].detach())),
        model.tdecoder[3].register_forward_hook(
            lambda m, i, o: seams.__setitem__("time_out", o[0].detach())),
    ]
    with torch.no_grad():
        estimates = model(mix)
    for handle in handles:
        handle.remove()

    with torch.no_grad():
        spectrogram = model._magnitude(model._spec(mix))
    globals()["_extra"] = {
        "waveform": torch.from_numpy(wave).contiguous(),
        # Channels last, as the port holds them.
        "spectrogram": spectrogram[0].permute(1, 2, 0).contiguous(),
        "bottleneck_in": seams["bottleneck_in"][0].permute(1, 2, 0).contiguous(),
        "bottleneck_out": seams["bottleneck_out"][0].permute(1, 2, 0).contiguous(),
        "freq_out": seams["freq_out"][0].permute(1, 2, 0).contiguous(),
        "time_out": seams["time_out"][0].permute(1, 0).contiguous(),
    }
    return estimates[0].contiguous()                            # [stems, channels, samples]


def run_sd_unet(image, checkpoint):
    """A Stable Diffusion UNet forward, `[height, width, 4]` epsilon.

    diffusers cannot be installed beside the other oracles here — it needs a newer transformers than the
    4.33.3 the Whisper / CLIP / SegFormer records were measured against — so it lives in its own venv and
    this mode runs under that interpreter. Set IK_SD_CONFIG to the model's `unet/config.json`.

    The text conditioning is supplied as a tensor rather than encoded here, and the record carries it:
    the port takes a context and the caller brings the encoder, so the text tower is deliberately not on
    this path.
    """
    import json
    import os
    from diffusers import UNet2DConditionModel

    with open(os.environ["IK_SD_CONFIG"]) as handle:
        config = json.load(handle)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    from safetensors.torch import load_file
    model = UNet2DConditionModel.from_config(config).eval()
    # A half-precision release loads into the float32 model, so both sides compute the same way.
    model.load_state_dict({k: v.float() for k, v in load_file(checkpoint).items()}, strict=True)

    generator = np.random.default_rng(3)
    size = int(os.environ.get("IK_SD_SIZE", "32"))
    latent = torch.from_numpy(
        generator.standard_normal((1, config["in_channels"], size, size)).astype(np.float32))
    context = torch.from_numpy(
        generator.standard_normal((1, 77, config["cross_attention_dim"])).astype(np.float32))
    timestep = torch.tensor([201], dtype=torch.long)
    extra = {"latent": latent[0].permute(1, 2, 0).contiguous(),
             "context": context[0].contiguous(),
             "timestep": timestep.to(torch.float32)}
    kwargs = {}
    if config.get("num_class_embeds"):
        labels = torch.tensor([17], dtype=torch.long)
        kwargs["class_labels"] = labels
        extra["class_label"] = labels.to(torch.float32)
    if config.get("addition_embed_type") == "text_time":
        # The pooled text embedding and the size descriptor SDXL folds into its timestep.
        pooled_width = (config["projection_class_embeddings_input_dim"]
                        - config["addition_time_embed_dim"] * 6)
        pooled = torch.from_numpy(generator.standard_normal((1, pooled_width)).astype(np.float32))
        time_ids = torch.tensor([[size * 8, size * 8, 0, 0, size * 8, size * 8]], dtype=torch.float32)
        kwargs["added_cond_kwargs"] = {"text_embeds": pooled, "time_ids": time_ids}
        extra["pooled"] = pooled[0].contiguous()
        extra["time_ids"] = time_ids[0].contiguous()
    with torch.no_grad():
        out = model(latent, timestep, encoder_hidden_states=context, **kwargs).sample
    globals()["_extra"] = extra
    return out[0].permute(1, 2, 0).contiguous()                 # [height, width, 4]


def run_sd_vae(image, checkpoint):
    """A Stable Diffusion autoencoder round trip, `[height, width, 3]`.

    The record carries the encoder's latent mean as well as the decoded image, so a failure says which
    half diverged. Set IK_SD_CONFIG to the model's `vae/config.json`.
    """
    import json
    import os
    from diffusers import AutoencoderKL

    with open(os.environ["IK_SD_CONFIG"]) as handle:
        config = json.load(handle)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    from safetensors.torch import load_file
    model = AutoencoderKL.from_config(config).eval()
    # The SD 1.5 autoencoders predate the attention rename that `from_pretrained` patches up on the
    # way in. Renaming here keeps the load strict, which is what proves the geometry matches.
    renames = {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.",
               ".proj_attn.": ".to_out.0."}
    state = {}
    for key, value in load_file(checkpoint).items():
        for old_part, new_part in renames.items():
            if old_part in key and "attentions" in key:
                key = key.replace(old_part, new_part)
        state[key] = value
    model.load_state_dict(state, strict=True)

    plate = image                                               # [H, W, 3] in 0...1, from --size/--plate
    x = torch.from_numpy(plate).permute(2, 0, 1)[None] * 2 - 1  # the reference works in -1...1
    with torch.no_grad():
        posterior = model.encode(x).latent_dist
        latent = posterior.mean
        decoded = model.decode(latent).sample
    # The plate is already the record's `input_image`; repeating it here would make two entries share
    # one storage, which safetensors refuses to write.
    globals()["_extra"] = {"latent": latent[0].permute(1, 2, 0).contiguous()}
    return decoded[0].permute(1, 2, 0).contiguous()             # [height, width, 3]


def run_sd_scheduler(image):
    """The DDIM sampler's schedule and per-step update, independent of any network.

    The InferKitMLX networks are at parity, but the loop that iterates them was never measured. This
    drives diffusers' own `DDIMScheduler` over a fixed sequence of latents and model outputs, so a
    comparison isolates the sampler's arithmetic from the UNet entirely.

    The configuration is the released `scheduler/scheduler_config.json` for SD 1.5 inpainting, whose
    `steps_offset` of 1 and `set_alpha_to_one: false` both move the schedule.

    Runs under the diffusers virtual environment, like the other `sd_*` modes.
    """
    import os
    from diffusers import DDIMScheduler

    steps = int(os.environ.get("IK_SD_STEPS", "20"))
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1,
                              prediction_type="epsilon")
    scheduler.set_timesteps(steps)

    generator = np.random.default_rng(7)
    shape = (1, 4, 8, 8)
    latent = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
    latents_in, predictions, latents_out = [], [], []
    for t in scheduler.timesteps:
        prediction = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
        latents_in.append(latent[0].permute(1, 2, 0).contiguous())
        predictions.append(prediction[0].permute(1, 2, 0).contiguous())
        latent = scheduler.step(prediction, t, latent).prev_sample
        latents_out.append(latent[0].permute(1, 2, 0).contiguous())

    clean = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
    noise = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
    noised = scheduler.add_noise(clean, noise, scheduler.timesteps[:1])

    globals()["_extra"] = {
        "timesteps": scheduler.timesteps.to(torch.float32),
        "alphas_cumprod": scheduler.alphas_cumprod.to(torch.float32),
        "latents_in": torch.stack(latents_in),
        "predictions": torch.stack(predictions),
        "clean": clean[0].permute(1, 2, 0).contiguous(),
        "noise": noise[0].permute(1, 2, 0).contiguous(),
        "noised": noised[0].permute(1, 2, 0).contiguous(),
    }
    return torch.stack(latents_out)                             # [steps, height, width, 4]


# The prompts the tokenizer and text-encoder records are measured on. The Swift side declares the same
# list, so a change to either without the other shows up as a mismatch rather than passing quietly.
SD_PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "A PHOTO, of  a   cat!!! (highly detailed), 8k",
    "",
    "  spaced   out  text  ",
    # Written with escapes so the exact code points are visible: byte-level BPE splits a multi-byte
    # character into bytes, which is a real case for the port to reproduce.
    "2024 was a year; na\u00efve caf\u00e9 \u2014 r\u00e9sum\u00e9",
]


def run_sd_tokenizer(image):
    """The ids CLIPTokenizer produces for SD_PROMPTS, one entry per prompt.

    The text encoder embeds ids rather than text, so the tokenizer is a separate seam: a prompt that
    tokenizes differently reaches the model as a different sentence, and the embedding comparison
    alone cannot tell that apart from a wrong weight.

    Set IK_SD_TOKENIZER to a release's `tokenizer/` directory. Runs under the diffusers virtual
    environment, like the other `sd_*` modes.
    """
    import os
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(os.environ["IK_SD_TOKENIZER"])
    extra = {}
    for index, prompt in enumerate(SD_PROMPTS):
        # The ids for the text alone; the start and end markers are the model input's, added below.
        raw = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
        extra[f"prompt_{index}"] = torch.tensor(raw, dtype=torch.int32)
    padded = tokenizer(SD_PROMPTS, padding="max_length", max_length=77, truncation=True,
                       return_tensors="pt")["input_ids"]
    extra["markers"] = torch.tensor([tokenizer.bos_token_id, tokenizer.eos_token_id,
                                     tokenizer.pad_token_id], dtype=torch.int32)
    globals()["_extra"] = extra
    return padded.to(torch.int32).contiguous()                  # [prompts, 77]


def run_sd_text_encoder(image, checkpoint):
    """A Stable Diffusion text encoder's hidden states for SD_PROMPTS[0], `[77, width]`.

    `--checkpoint` is the release's `text_encoder` DIRECTORY, so the reference builds itself from the
    config that ships beside the weights. A tower carrying a projection (SDXL's second) is built as
    `CLIPTextModelWithProjection` and the record adds its pooled embedding.

    The record carries the reference's own token ids: the port embeds ids, not text, so the tokenizer
    is measured separately by `sd_tokenizer` rather than folded into this number.

    Both hidden states are recorded. A release reads one of them — SD 1.x and 2.x take the last, after
    the final layer normalization; SDXL takes the penultimate, before it — and recording both means a
    disagreement says which convention diverged rather than only that something did.

    Set IK_SD_TOKENIZER to the release's `tokenizer/`. Runs under the diffusers virtual environment.
    """
    import json
    import os
    from transformers import CLIPTokenizer

    with open(os.path.join(checkpoint, "config.json")) as handle:
        config = json.load(handle)
    projected = "CLIPTextModelWithProjection" in config.get("architectures", [])
    if projected:
        from transformers import CLIPTextModelWithProjection as Model
    else:
        from transformers import CLIPTextModel as Model
    # A release published only in half precision names its files `.fp16.safetensors`.
    variant = None if os.path.exists(os.path.join(checkpoint, "model.safetensors")) else "fp16"
    model = Model.from_pretrained(checkpoint, torch_dtype=torch.float32, variant=variant).eval()

    tokenizer = CLIPTokenizer.from_pretrained(os.environ["IK_SD_TOKENIZER"])
    encoded = tokenizer(SD_PROMPTS[0], padding="max_length", max_length=77, truncation=True,
                        return_tensors="pt")
    with torch.no_grad():
        out = model(encoded["input_ids"], output_hidden_states=True)

    extra = {"tokens": encoded["input_ids"][0].to(torch.int32).contiguous(),
             "penultimate": out.hidden_states[-2][0].contiguous()}
    if projected:
        extra["pooled"] = out.text_embeds[0].contiguous()
    globals()["_extra"] = extra
    return out.last_hidden_state[0].contiguous()                # [77, width]


def run_sd_text_to_image(image, checkpoint):
    """A whole Stable Diffusion text-to-image run, `[height, width, 3]` in 0...1.

    `--checkpoint` is the release DIRECTORY, so the reference assembles itself from the same files the
    port loads. The sampler is swapped to DDIM, which is the one InferKitMLX implements; the released
    configuration's `steps_offset` and `set_alpha_to_one` carry over through `from_config`.

    The record carries the initial latent. Matching a random source across two implementations proves
    nothing about either, so the port starts from the reference's own noise and the comparison is of
    the text encoder, the UNet, the sampler, and the autoencoder end to end.

    IK_SD_STEPS sets the step count and IK_SD_SIZE the latent side. Runs under the diffusers virtual
    environment, like the other `sd_*` modes.
    """
    import json
    import os
    from diffusers import DDIMScheduler, DiffusionPipeline

    with open(os.path.join(checkpoint, "model_index.json")) as handle:
        index = json.load(handle)
    extra = {} if index["_class_name"].startswith("StableDiffusionXL") else {
        "safety_checker": None, "requires_safety_checker": False}
    variant = None if os.path.exists(
        os.path.join(checkpoint, "unet", "diffusion_pytorch_model.safetensors")) else "fp16"
    pipeline = DiffusionPipeline.from_pretrained(checkpoint, torch_dtype=torch.float32,
                                                 variant=variant, **extra)
    # DDIM is the sampler InferKitMLX implements. A release whose own scheduler counts its schedule
    # down from the end of the training range keeps that spacing through `from_config`.
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    steps = int(os.environ.get("IK_SD_STEPS", "4"))
    side = int(os.environ.get("IK_SD_SIZE", "32"))
    guidance = float(os.environ.get("IK_SD_GUIDANCE", "7.5"))
    # A release with `force_zeros_for_empty_prompt` conditions on zeros when NO negative prompt is
    # supplied, and on the embedding of an empty sentence when one is. IK_SD_NEGATIVE picks which.
    negative = None if os.environ.get("IK_SD_NEGATIVE") == "none" else ""
    generator = np.random.default_rng(11)
    channels = pipeline.unet.config.in_channels
    latent = torch.from_numpy(generator.standard_normal((1, channels, side, side)).astype(np.float32))

    # The latent after each step, so a whole-picture mismatch says WHICH step diverged rather than
    # only that something did.
    trace = []

    def capture(pipe, index, timestep, kwargs):
        trace.append(kwargs["latents"][0].permute(1, 2, 0).contiguous().clone())
        return kwargs

    with torch.no_grad():
        result = pipeline(prompt=SD_PROMPTS[0], negative_prompt=negative, num_inference_steps=steps,
                          guidance_scale=guidance, latents=latent, height=side * 8, width=side * 8,
                          output_type="np", callback_on_step_end=capture)
    # The conditioning and the first guided prediction, so a divergence separates the text tower from
    # the sampler from the UNet without another run.
    extra = {"latent": latent[0].permute(1, 2, 0).contiguous(),
             "latents": torch.stack(trace),
             "settings": torch.tensor([steps, side, guidance, 1 if negative is None else 0], dtype=torch.float32)}
    two_towers = hasattr(pipeline, "text_encoder_2")
    with torch.no_grad():
        if two_towers:
            conditional, unconditional, pooled, unpooled = pipeline.encode_prompt(
                prompt=SD_PROMPTS[0], device="cpu", num_images_per_prompt=1,
                do_classifier_free_guidance=guidance > 1, negative_prompt=negative)
            extra["pooled"] = pooled[0].contiguous()
            added = {"text_embeds": pooled,
                     "time_ids": torch.tensor([[side * 8, side * 8, 0, 0, side * 8, side * 8]],
                                              dtype=torch.float32)}
        else:
            conditional, unconditional = pipeline.encode_prompt(
                SD_PROMPTS[0], device="cpu", num_images_per_prompt=1,
                do_classifier_free_guidance=guidance > 1, negative_prompt=negative)[:2]
            added = None
        pipeline.scheduler.set_timesteps(steps)
        first = pipeline.scheduler.timesteps[0]
        kwargs = {} if added is None else {"added_cond_kwargs": added}
        if guidance > 1:
            both = torch.cat([unconditional, conditional])
            if added is not None:
                kwargs["added_cond_kwargs"] = {"text_embeds": torch.cat([unpooled, pooled]),
                                               "time_ids": torch.cat([added["time_ids"]] * 2)}
            predicted = pipeline.unet(torch.cat([latent] * 2), first,
                                      encoder_hidden_states=both, **kwargs).sample
            prediction = predicted[0] + guidance * (predicted[1] - predicted[0])
        else:
            prediction = pipeline.unet(latent, first, encoder_hidden_states=conditional,
                                       **kwargs).sample[0]

    extra["context"] = conditional[0].contiguous()
    extra["uncontext"] = (unconditional[0].contiguous() if unconditional is not None
                          else torch.zeros_like(conditional[0]))
    extra["first_prediction"] = prediction.permute(1, 2, 0).contiguous()
    globals()["_extra"] = extra
    return torch.from_numpy(result.images[0]).contiguous()      # [height, width, 3] in 0...1


def run_denoiser(image, checkpoint):
    """facebookresearch/denoiser speech enhancement of a deterministic noisy clip, `[samples]`.

    Set IK_DENOISER_SRC to a directory holding `demucs.py` and `resample.py` from the denoiser
    repository. The released files are bare state dicts, so the one thing dns48 and dns64 differ in —
    the hidden width — is read off the first encoder weight rather than guessed.

    This model is `NFKMLXDemucsNet` in its speech configuration — the same encoder, bottleneck, and
    decoder the music model uses — so this record also guards the shared module against a change made
    for one family breaking the other.
    """
    import os
    import types

    utils = types.ModuleType("denoiser.utils")
    utils.capture_init = _capture_init
    module = _import_reference(os.environ.get("IK_DENOISER_SRC", "."), "denoiser", "demucs",
                               siblings=["resample"], injected={"utils": utils})
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state", state)
    model = module.Demucs(hidden=state["encoder.0.0.weight"].shape[0]).eval()
    model.load_state_dict(state, strict=True)

    samples = 16000
    time = np.arange(samples, dtype=np.float32) / 16000.0
    generator = np.random.default_rng(9)
    # A voiced-like harmonic stack under broadband noise: something for the model to actually remove.
    speech = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 140 * (h + 1) * time) for h in range(5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * time)
    wave = (speech * envelope + 0.05 * generator.standard_normal(samples)).astype(np.float32)

    with torch.no_grad():
        cleaned = model(torch.from_numpy(wave).reshape(1, 1, -1))
    globals()["_extra"] = {"prompt": torch.tensor(prompt_ids, dtype=torch.int32).contiguous(),
                           "waveform": torch.from_numpy(wave).contiguous()}
    return cleaned[0, 0].contiguous()                           # [samples]


def _reference_source():
    """The directory holding the single-file reference implementations, from IK_REF_SRC."""
    import os
    return os.environ.get("IK_REF_SRC", ".")


def run_u2net(image, checkpoint):
    """U²-Net saliency, `[H, W]` in 0...1, from the reference `model/u2net.py`.

    The reference's eval script resizes to 320×320 before the network; feeding a 320×320 plate makes
    that resize an identity, so only the network and its normalization are compared.
    """
    module = _import_reference(_reference_source(), "u2net_ref", "u2net")
    # `u2netp` is the light network, a separate class in the reference rather than a configuration.
    light = os.environ.get("IK_U2NET_VARIANT", "full") == "light"
    model = (module.U2NETP(3, 1) if light else module.U2NET(3, 1)).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    # `ToTensorLab`: scale by the plate's own maximum, then normalize with ImageNet statistics.
    scaled = image / max(image.max(), 1e-8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((scaled - mean) / deviation).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        saliency = model(tensor)[0]
    return saliency[0, 0].contiguous()                          # [H, W]


def run_raft(image, checkpoint):
    """RAFT optical flow between a plate and a shifted copy of it, from princeton-vl's own `core/`.

    IK_REF_SRC holds a `raft/` directory with `raft.py`, `corr.py`, `extractor.py`, `update.py`, and
    `utils/utils.py` (the package imports its siblings flatly, so its directory goes on the path rather
    than through `_import_reference`). `--checkpoint` is `raft-things.pth`, whose keys carry the
    `module.` prefix of the training wrapper.

    The record carries `frame1` (the shifted copy both sides read) and both of the reference's outputs:
    `output` is the eighth-resolution flow the recurrent update produces, and `flow_up` the
    convex-mask upsampling of it. Comparing the low-resolution field first separates the network from
    the upsampler.
    """
    import os

    sys.path.insert(0, os.path.join(_reference_source(), "raft"))
    from raft import RAFT

    class Arguments:
        small = False
        dropout = 0
        alternate_corr = False
        mixed_precision = False

        def __contains__(self, name):
            return hasattr(self, name)

    model = RAFT(Arguments())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({name.replace("module.", "", 1): value for name, value in state.items()},
                          strict=True)
    model.eval()

    # A known translation gives the network real motion to find; rolling keeps both frames identical
    # in content, so any disagreement is the implementation rather than the input.
    shift = (5, 7)
    second = np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)
    frames = [torch.from_numpy(np.ascontiguousarray(plate)).permute(2, 0, 1)[None] * 255.0
              for plate in (image, second)]
    with torch.no_grad():
        # Both sides must run the same number of recurrent updates; six is the port's default.
        low, up = model(frames[0], frames[1], iters=6, test_mode=True)
    globals()["_extra"] = {"frame1": torch.from_numpy(np.ascontiguousarray(second)),
                           "flow_up": up[0].permute(1, 2, 0).contiguous()}
    return low[0].permute(1, 2, 0).contiguous()                 # [H/8, W/8, 2] to match NHWC


def run_audio_tagger(image, checkpoint):
    """PANNs Cnn14 AudioSet tag probabilities, `[classes]`, from the reference `models.py`.

    IK_REF_SRC holds `models.py` (saved as `panns_models.py`) and its sibling `pytorch_utils.py` from
    qiuqiangkong/audioset_tagging_cnn; the reference needs `torchlibrosa`. The record's `waveform`
    carries the clip both sides read, `features` the reference's own mel spectrogram, and `embedding`
    the 2048-wide clip vector the classifier reads — three seams, so a front-end mismatch, a network
    mismatch, and a classifier mismatch are told apart rather than summed into one number.
    """
    import os

    source = _reference_source()
    sys.path.insert(0, source)                                  # models.py imports pytorch_utils flatly
    module = _import_reference(source, "panns_ref", "panns_models")
    model = module.Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                         fmin=50, fmax=14000, classes_num=527).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"],
                          strict=True)

    rate, samples = 32000, 64000
    time = np.arange(samples, dtype=np.float32) / rate
    generator = np.random.default_rng(19)
    # A voiced-like harmonic stack, then noise: a clip with real structure in both halves.
    tone = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 220 * (h + 1) * time) for h in range(6))
    wave = np.where(time < 1.0, tone, 0.1 * generator.standard_normal(samples)).astype(np.float32)

    with torch.no_grad():
        batch = torch.from_numpy(wave)[None]
        mel = model.logmel_extractor(model.spectrogram_extractor(batch))    # [1, 1, frames, mels]
        output = model(batch)
    globals()["_extra"] = {"prompt": torch.tensor(prompt_ids, dtype=torch.int32).contiguous(),
                           "waveform": torch.from_numpy(wave).contiguous(),
                           "features": mel[0, 0].contiguous(),
                           "embedding": output["embedding"][0].contiguous()}
    return output["clipwise_output"][0].contiguous()            # [classes]


def run_pose(image, checkpoint):
    """SimpleBaseline joint heatmaps, `[H/4, W/4, joints]`, from the reference `pose_resnet.py`.

    IK_REF_SRC holds `pose_resnet.py` from microsoft/human-pose-estimation.pytorch. `--checkpoint` is the
    mmpose ResNet-50 COCO release, whose keys are the reference's under a `backbone.`/`head.` prefix, so
    the strict load doubles as proof that the two architectures are the same one.

    The network is fully convolutional, so it takes the square plate as it stands; feeding it directly
    keeps the person-crop resize out of the comparison and leaves the network and its normalization.
    """
    import os
    import types as _types

    module = _import_reference(_reference_source(), "pose_ref", "pose_resnet")
    extra = _types.SimpleNamespace(DECONV_WITH_BIAS=False, NUM_DECONV_LAYERS=3,
                                   NUM_DECONV_FILTERS=[256, 256, 256], NUM_DECONV_KERNELS=[4, 4, 4],
                                   FINAL_CONV_KERNEL=1)
    cfg = _types.SimpleNamespace(MODEL=_types.SimpleNamespace(NUM_JOINTS=17, EXTRA=extra))
    model = module.PoseResNet(module.Bottleneck, [3, 4, 6, 3], cfg).eval()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose-to-safetensors"))
    import convert as pose_convert                              # its mmengine stub, not a second copy
    state = pose_convert.extract_state_dict(pose_convert.load_checkpoint(checkpoint))
    trimmed = {name.split(".", 1)[1]: value for name, value in state.items()
               if name.startswith(("backbone.", "head."))}
    model.load_state_dict(trimmed, strict=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((image - mean) / deviation).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensor)
    return heatmaps[0].permute(1, 2, 0).contiguous()            # [H/4, W/4, joints] to match NHWC


def run_rvm(image, checkpoint):
    """Robust Video Matting first-frame alpha, `[H, W]` in 0...1, from the reference `MattingNetwork`.

    IK_REF_SRC holds an `rvm/` directory with the repository's `model/` files (`model.py` and its
    siblings; it needs torchvision). The record's `foreground` carries the composited full-resolution
    foreground, and `alpha_downsampled` / `foreground_downsampled` a second pass at
    `downsample_ratio=0.5`, which routes through the deep-guided-filter refiner the full-resolution
    pass never touches — so the refiner is measured separately from the core network.
    """
    import os

    module = _import_reference(os.path.join(_reference_source(), "rvm"), "rvm_ref", "model",
                               siblings=["mobilenetv3", "resnet", "lraspp", "decoder",
                                         "fast_guided_filter", "deep_guided_filter"])
    model = module.MattingNetwork("mobilenetv3").eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    src = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        fgr, pha, *_ = model(src)
        fgr_small, pha_small, *_ = model(src, downsample_ratio=0.5)
    globals()["_extra"] = {"foreground": fgr[0].permute(1, 2, 0).contiguous(),
                           "alpha_downsampled": pha_small[0, 0].contiguous(),
                           "foreground_downsampled": fgr_small[0].permute(1, 2, 0).contiguous()}
    return pha[0, 0].contiguous()


def run_yolo(image, checkpoint):
    """YOLOv8 pre-suppression predictions, `[anchors, 4 + classes]`, from ultralytics' own model.

    The decoded tensor is the right seam: box centers and sizes in pixels plus sigmoid class
    probabilities for every anchor across the three strides, before any thresholding or NMS — so a
    wrong DFL decode or anchor grid cannot hide behind a lucky suppression. Feed a square plate
    (`--size 640 --image <photo>`) so the reference's letterboxing is an identity.
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    net = model.model.float().eval()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        predictions = net(tensor)
    decoded = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
    return decoded[0].transpose(0, 1).contiguous()              # [anchors, 4 + classes]


def run_codeformer(image, checkpoint):
    """CodeFormer restoration, `[H, W, 3]` in 0...1, from sczhou/CodeFormer's own architecture files.

    IK_REF_SRC holds a `codeformer/` directory with `codeformer_arch.py` and `vqgan_arch.py`; the
    basicsr registry/logger they import are stubbed. Runs the real inference settings (`w=0.5`,
    `adain=True`). The record's `logits` carries the code-prediction seam `[tokens, codebook]` — the
    continuous tensor that localizes a mismatch to encoder+transformer vs generator, robust to
    argmax near-ties the final image is not.
    """
    import os
    import types

    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = type("Registry", (), {"register": staticmethod(lambda: (lambda cls: cls))})()
    utils = types.ModuleType("basicsr.utils")
    utils.get_root_logger = lambda *args, **kwargs: None
    sys.modules.update({"basicsr": types.ModuleType("basicsr"), "basicsr.utils": utils,
                        "basicsr.utils.registry": registry})

    source = os.path.join(_reference_source(), "codeformer")
    module = _import_reference(source, "basicsr.archs", "codeformer_arch", siblings=["vqgan_arch"])
    model = module.CodeFormer(fix_modules=None).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        out, logits, _ = model(tensor, w=0.5, adain=True)
    globals()["_extra"] = {"logits": logits[0].contiguous()}
    return torch.clamp((out[0].permute(1, 2, 0) + 1) / 2, 0, 1).contiguous()


def run_yolo_detections(image, checkpoint):
    """YOLOv8 detections on a NON-SQUARE frame, through ultralytics' own `predict` — letterboxing,
    suppression, and the mapping back to original-image coordinates included.

    `run_yolo` compares the raw prediction tensor on a square plate, where letterboxing is an
    identity. This mode is the other half: a 16:9 frame, where the reference scales by the smaller
    ratio, pads to a stride multiple with gray, and then undoes both when it reports boxes. The record
    carries `plate` (the non-square frame, `[H, W, 3]`), `output` (normalized xyxy boxes), `classes`,
    and `confidences`.

    Ultralytics treats a numpy array as BGR and flips it internally, so the plate is handed over
    reversed — passing RGB would silently detect on colour-swapped pixels.
    """
    from PIL import Image
    from ultralytics import YOLO

    source = os.environ.get("IK_YOLO_IMAGE")
    if not source:
        raise SystemExit("yolo_detections needs IK_YOLO_IMAGE (a photo to letterbox)")
    photo = Image.open(source).convert("RGB").resize((640, 360), Image.BILINEAR)
    plate = np.ascontiguousarray(np.asarray(photo).astype(np.float32) / 255.0)

    model = YOLO(checkpoint)
    results = model.predict(np.ascontiguousarray((plate * 255).astype(np.uint8)[..., ::-1]),
                            verbose=False)[0]
    globals()["_extra"] = {"plate": torch.from_numpy(plate).contiguous(),
                           "classes": results.boxes.cls.to(torch.int32).contiguous(),
                           "confidences": results.boxes.conf.contiguous()}
    return results.boxes.xyxyn.contiguous()                     # [detections, 4] normalized xyxy


def run_lama(image, checkpoint):
    """LaMa (big-lama) inpainting, `[H, W, 3]` in 0...1, from advimman's own `ffc.py`.

    IK_REF_SRC holds a `lama/` directory with `ffc.py`. Its three `saicinpainting` siblings are
    stubbed: `get_activation` is the only one the generator actually reaches (big-lama ends in a
    sigmoid), while the spatial-transform wrapper and squeeze-excitation are unused at this config.
    The geometry comes from big-lama's own `config.yaml`: 18 blocks, ratio 0.75 through the trunk and
    at the last downsample only.

    The record carries `mask` (1 where the region is regenerated), `raw` (the generator's own output,
    before compositing — the seam that isolates the network from the paste-back) and `output` (the
    composited result the port returns).
    """
    import types
    from torch import nn

    base = types.ModuleType("saicinpainting.training.modules.base")

    def get_activation(kind="tanh"):
        return {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), False: nn.Identity()}[kind]

    base.get_activation = get_activation
    base.BaseDiscriminator = nn.Module
    spatial = types.ModuleType("saicinpainting.training.modules.spatial_transform")
    spatial.LearnableSpatialTransformWrapper = nn.Module
    squeeze = types.ModuleType("saicinpainting.training.modules.squeeze_excitation")
    squeeze.SELayer = nn.Module
    for name in ["saicinpainting", "saicinpainting.training", "saicinpainting.training.modules"]:
        package = types.ModuleType(name)
        package.__path__ = []                                   # a plain module cannot host submodules
        sys.modules[name] = package
    utils = types.ModuleType("saicinpainting.utils")
    utils.get_shape = lambda t: tuple(t.shape)
    sys.modules["saicinpainting.utils"] = utils
    sys.modules["saicinpainting.training.modules.base"] = base
    sys.modules["saicinpainting.training.modules.spatial_transform"] = spatial
    sys.modules["saicinpainting.training.modules.squeeze_excitation"] = squeeze

    module = _import_reference(os.path.join(_reference_source(), "lama"), "lama_ref", "ffc")
    model = module.FFCResNetGenerator(
        input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18, add_out_act="sigmoid",
        init_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        downsample_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        resnet_conv_kwargs={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False}).eval()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict({name[len("generator."):]: value for name, value in state.items()
                           if name.startswith("generator.")}, strict=True)

    # A rectangle over the middle of the frame: a real hole, not a token one.
    height, width = image.shape[:2]
    mask = np.zeros((height, width, 1), dtype=np.float32)
    mask[height // 3: height * 2 // 3, width // 3: width * 2 // 3] = 1.0

    plate = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0)
    stacked = torch.cat([plate * (1 - mask_tensor), mask_tensor], dim=1)
    with torch.no_grad():
        predicted = model(stacked)
    composited = mask_tensor * predicted + (1 - mask_tensor) * plate
    globals()["_extra"] = {"mask": torch.from_numpy(mask[..., 0]).contiguous(),
                           "raw": predicted[0].permute(1, 2, 0).contiguous()}
    return composited[0].permute(1, 2, 0).contiguous()


def run_nafnet(image, checkpoint):
    """NAFNet restoration, `[H, W, 3]` in 0...1, from megvii-research's own `NAFNet_arch.py`.

    IK_REF_SRC holds a `nafnet/` directory with `NAFNet_arch.py` (saved as `nafnet_arch.py`). Its two
    basicsr imports are supplied here: `LayerNorm2d` is the reference's own channel-wise normalization
    (a LayerNorm over the channel axis of an NCHW tensor), and `Local_Base` is only used by the
    TLSC variant, so a bare class is enough. The released SIDD width-32 checkpoint is the denoising
    model, which is what the module's default configuration sizes.
    """
    import types
    from torch import nn

    class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
            self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
            self.eps = eps

        def forward(self, x):
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / (var + self.eps).sqrt()
            return self.weight[None, :, None, None] * y + self.bias[None, :, None, None]

    arch_util = types.ModuleType("basicsr.models.archs.arch_util")
    arch_util.LayerNorm2d = LayerNorm2d
    local_arch = types.ModuleType("basicsr.models.archs.local_arch")
    local_arch.Local_Base = type("Local_Base", (), {})
    for name in ["basicsr", "basicsr.models", "basicsr.models.archs"]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules["basicsr.models.archs.arch_util"] = arch_util
    sys.modules["basicsr.models.archs.local_arch"] = local_arch

    module = _import_reference(os.path.join(_reference_source(), "nafnet"), "nafnet_ref", "nafnet_arch")
    # The released geometries, from the repository's own option files. SIDD (denoise) spreads its
    # blocks through the middle; GoPro (deblur) puts twenty-eight of them in the last encoder stage.
    geometries = {
        "SIDD": {"width": 32, "middle_blk_num": 12, "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2]},
        "GoPro": {"width": 32, "middle_blk_num": 1, "enc_blk_nums": [1, 1, 1, 28], "dec_blk_nums": [1, 1, 1, 1]},
        "REDS": {"width": 64, "middle_blk_num": 1, "enc_blk_nums": [1, 1, 1, 28], "dec_blk_nums": [1, 1, 1, 1]},
    }
    model = module.NAFNet(img_channel=3, **geometries[os.environ.get("IK_NAFNET_VARIANT", "SIDD")]).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("params", state.get("state_dict", state)), strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        restored = model(tensor)
    return restored[0].permute(1, 2, 0).contiguous()


def run_rife(image, checkpoint):
    """RIFE frame interpolation, `[H, W, 3]` in 0...1, from the released HDv3 `IFNet_HDv3.py`.

    IK_REF_SRC holds a `rife/` directory with `ifnet_hdv3.py` and `warplayer.py` (the reference
    imports `warp` from `model.warplayer`, injected here). The two frames are the plate and a shifted
    copy, so the network has real motion to find; the record's `frame1` carries the shifted copy the
    Swift side reads back.
    """
    import types

    warplayer = _import_reference(os.path.join(_reference_source(), "rife"), "rife_model", "warplayer")
    model_package = types.ModuleType("model")
    model_package.__path__ = []
    sys.modules["model"] = model_package
    sys.modules["model.warplayer"] = warplayer

    module = _import_reference(os.path.join(_reference_source(), "rife"), "rife_ref", "ifnet_hdv3")
    model = module.IFNet().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({name.replace("module.", "", 1): value for name, value in state.items()},
                          strict=True)

    shift = (4, 6)
    second = np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)

    # The reference network has no internal padding and simply fails on a size that is not a multiple
    # of the coarsest stride — its own inference script pads outside the model. The port pads by edge
    # replication to a multiple of 32 and crops back, so the oracle does the same here: what is then
    # compared is the network on identical input, not a padding convention the reference never fixes.
    height, width = image.shape[:2]
    padded = [np.pad(plate, ((0, -height % 32), (0, -width % 32), (0, 0)), mode="edge")
              for plate in (image, second)]
    frames = [torch.from_numpy(np.ascontiguousarray(plate)).permute(2, 0, 1)[None] for plate in padded]
    with torch.no_grad():
        _, _, merged = model(torch.cat(frames, dim=1))
    globals()["_extra"] = {"frame1": torch.from_numpy(np.ascontiguousarray(second)).contiguous()}
    return merged[2][0].permute(1, 2, 0)[:height, :width].contiguous()   # finest scale, cropped back


def run_modnet(image, checkpoint):
    """MODNet portrait matte, `[H, W]` in 0...1, from ZHKKKe's own `modnet.py`.

    IK_REF_SRC holds a `modnet/` directory with `modnet.py`, `mobilenetv2.py`, and `wrapper.py`. The
    backbone wrapper would fetch ImageNet weights on construction, so `backbone_pretrained=False`
    skips that — the released checkpoint supplies everything. Inference normalizes to -1...1 (the
    demo's `Normalize(0.5, 0.5)`), and the network's strides need sides that are multiples of 32, so
    feed a plate that already is one.
    """
    import types
    from torch import nn

    source = os.path.join(_reference_source(), "modnet")
    # `wrapper.py` imports the network from `.mobilenetv2`, and `modnet.py` reaches for
    # `.backbones.SUPPORTED_BACKBONES` — a package `__init__` that is not one of the three files, so
    # it is synthesized here from the wrapper's own class.
    mobilenet = _import_reference(source, "modnet_ref", "mobilenetv2")
    sys.modules["modnet_ref.backbones.mobilenetv2"] = mobilenet
    wrapper = _import_reference(source, "modnet_ref", "wrapper",
                                injected={"mobilenetv2": mobilenet})
    backbones = types.ModuleType("modnet_ref.backbones")
    backbones.__path__ = []
    backbones.SUPPORTED_BACKBONES = {"mobilenetv2": wrapper.MobileNetV2Backbone}
    sys.modules["modnet_ref.backbones"] = backbones

    module = _import_reference(source, "modnet_ref", "modnet")
    model = module.MODNet(backbone_pretrained=False).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict({name.replace("module.", "", 1): value for name, value in state.items()},
                          strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        _, _, matte = model(tensor, inference=True)
    return matte[0, 0].contiguous()                             # [H, W] alpha


def run_bisenet(image, checkpoint):
    """BiSeNetV1 class logits, `[H, W, classes]`, from CoinCheung's own `bisenetv1.py`.

    IK_REF_SRC holds a `bisenet/` directory with `bisenetv1.py` and `resnet.py`. The ResNet-18 would
    fetch torchvision's ImageNet weights on construction, so its loader is neutralized — the released
    checkpoint supplies everything. Comparing logits rather than the label map keeps the check
    sensitive: an argmax hides every difference too small to flip a pixel.
    """
    import types
    from torch import nn

    modules = types.ModuleType("lib.models.resnet")
    for name in ["lib", "lib.models"]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    source = os.path.join(_reference_source(), "bisenet")
    resnet = _import_reference(source, "bisenet_ref", "resnet")
    resnet.Resnet18.init_weight = lambda self: None             # skip the torchvision download
    sys.modules["lib.models.resnet"] = resnet
    module = _import_reference(source, "bisenet_ref", "bisenetv1", injected={"resnet": resnet})

    # `aux_mode="eval"` omits the two auxiliary heads, which the released checkpoint carries, so the
    # model is built in the mode that has them and simply run in eval; the first output is the one
    # inference uses.
    model = module.BiSeNetV1(n_classes=19, aux_mode="train").eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((torch.from_numpy(image).permute(2, 0, 1) - mean) / deviation).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
    return logits[0].permute(1, 2, 0).contiguous()              # [H, W, classes] to match NHWC


def run_whisper(image, checkpoint):
    """Whisper tiny transcription, from openai-whisper itself.

    `--checkpoint` is the released `tiny.pt`. The record carries the `waveform` both sides read, the
    reference's own **log-mel** (the seam between the front end and the encoder — a mel mismatch and a
    network mismatch look identical at the token level), and `output` as the generated token ids. The
    clip is a deterministic harmonic sweep rather than speech: the tokens it produces are arbitrary,
    but both implementations must produce the SAME arbitrary tokens.
    """
    import whisper

    model = whisper.load_model(checkpoint) if os.path.exists(checkpoint) else whisper.load_model("tiny")
    model.eval()
    # The prompt and the suppression boundary come from the model's OWN tokenizer: large-v3 carries one
    # more language token than every earlier size, which shifts `<|transcribe|>` and `<|notimestamps|>`.
    reference_tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=getattr(model, "num_languages", 99),
        language="en", task="transcribe")
    prompt_ids = list(reference_tokenizer.sot_sequence_including_notimestamps)

    rate, seconds = 16000, 4.0
    time = np.arange(int(rate * seconds), dtype=np.float32) / rate
    generator = np.random.default_rng(29)
    # A voiced-like stack with a slow sweep, then noise: enough structure to move the decoder off its
    # prompt without depending on a real recording.
    sweep = sum(0.3 / (h + 1) * np.sin(2 * np.pi * (140 + 40 * time) * (h + 1) * time) for h in range(4))
    wave = np.where(time < 3.0, sweep, 0.05 * generator.standard_normal(time.shape)).astype(np.float32)

    padded = whisper.pad_or_trim(torch.from_numpy(wave))
    # large-v3 produces 128 mel bands where every earlier size produces 80, and its encoder's first
    # convolution takes that many channels; the default would feed it the wrong shape.
    mel = whisper.log_mel_spectrogram(padded, n_mels=model.dims.n_mels)

    # `model.decode` is not plain greedy: it also applies SuppressBlank, a curated non-speech token
    # list, and timestamp rules. The port suppresses the special/timestamp range wholesale, so running
    # the reference's own decoder under the PORT's rule is what isolates the network — otherwise a
    # decoding-policy difference is indistinguishable from a weights or attention bug. The policy gap
    # itself is real and recorded separately.
    prompt = prompt_ids                                         # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    tokens = list(prompt)
    with torch.no_grad():
        audio = model.encoder(mel.unsqueeze(0))
        first_logits = None
        for _ in range(32):
            logits = model.decoder(torch.tensor([tokens]), audio)[0, -1].clone()
            if first_logits is None:
                first_logits = logits.clone()
            logits[reference_tokenizer.sot:] = float("-inf")     # the port's suppression, applied here
            nxt = int(torch.argmax(logits).item())
            if nxt == 50257:                                    # <|endoftext|>
                break
            tokens.append(nxt)

    # The reference's OWN rules, which the port now implements: its curated non-speech set masked at
    # every step, and SuppressBlank (a space and an immediate end of text) at the first sampled
    # position only. Recorded beside the port-rule tokens so the two policies are compared directly
    # rather than described.
    non_speech = list(reference_tokenizer.non_speech_tokens)
    space_token = reference_tokenizer.encode(" ")[0]

    ruled = list(prompt)
    with torch.no_grad():
        for step in range(32):
            logits = model.decoder(torch.tensor([ruled]), audio)[0, -1].clone()
            logits[reference_tokenizer.sot:] = float("-inf")     # specials and timestamps
            logits[non_speech] = float("-inf")                  # SuppressTokens, the "-1" default
            if step == 0:                                       # SuppressBlank
                logits[space_token] = float("-inf")
                logits[reference_tokenizer.eot] = float("-inf")
            nxt = int(torch.argmax(logits).item())
            if nxt == reference_tokenizer.eot:
                break
            ruled.append(nxt)

    # The TIMESTAMPED decode, which is a different prompt as well as a different rule set: the segment
    # times only exist when `<|notimestamps|>` is left OUT of the prompt, and the timestamp range then
    # has to stay unmasked. `ApplyTimestampRules` is what orders the result — timestamps in pairs,
    # never decreasing, and a timestamp forced at the opening position.
    timestamp_begin = reference_tokenizer.timestamp_begin
    no_timestamps = reference_tokenizer.no_timestamps
    timed_prompt = [t for t in prompt_ids if t != no_timestamps]
    sample_begin = len(timed_prompt)
    max_initial_index = round(1.0 / (30.0 / model.dims.n_audio_ctx))     # max_initial_timestamp 1.0 s

    timed = list(timed_prompt)
    with torch.no_grad():
        for _ in range(32):
            logits = model.decoder(torch.tensor([timed]), audio)[0, -1].clone()
            logits[no_timestamps] = float("-inf")
            logits[reference_tokenizer.sot: timestamp_begin] = float("-inf")   # specials, not times
            logits[non_speech] = float("-inf")
            if len(timed) == sample_begin:
                logits[space_token] = float("-inf")
                logits[reference_tokenizer.eot] = float("-inf")

            seq = timed[sample_begin:]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    logits[timestamp_begin:] = float("-inf")
                else:
                    logits[: reference_tokenizer.eot] = float("-inf")
            stamps = [t for t in seq if t >= timestamp_begin]
            if stamps:
                last = stamps[-1] if (last_was_timestamp and not penultimate_was_timestamp) else stamps[-1] + 1
                logits[timestamp_begin: last] = float("-inf")
            if len(timed) == sample_begin:
                logits[: timestamp_begin] = float("-inf")
                logits[timestamp_begin + max_initial_index + 1:] = float("-inf")
            # If the timestamps hold more probability together than any single text token, take one.
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            if logprobs[timestamp_begin:].logsumexp(dim=-1) > logprobs[:timestamp_begin].max():
                logits[:timestamp_begin] = float("-inf")

            nxt = int(torch.argmax(logits).item())
            if nxt == reference_tokenizer.eot:
                break
            timed.append(nxt)

    globals()["_extra"] = {"prompt": torch.tensor(prompt_ids, dtype=torch.int32).contiguous(),
                           "waveform": torch.from_numpy(wave).contiguous(),
                           "features": mel.contiguous(),                 # [mels, frames]
                           "first_logits": first_logits.contiguous(),    # the decoder seam
                           "non_speech_tokens": torch.tensor(sorted(non_speech), dtype=torch.int32).contiguous(),
                           "space_token": torch.tensor([space_token], dtype=torch.int32).contiguous(),
                           "ruled_tokens": torch.tensor(ruled[len(prompt):], dtype=torch.int32).contiguous(),
                           "timestamp_prompt": torch.tensor(timed_prompt, dtype=torch.int32).contiguous(),
                           "timestamp_begin": torch.tensor([timestamp_begin], dtype=torch.int32).contiguous(),
                           "timed_tokens": torch.tensor(timed[sample_begin:], dtype=torch.int32).contiguous()}
    return torch.tensor(tokens[len(prompt):], dtype=torch.int32).contiguous()



def run_retinaface(image, checkpoint):
    """RetinaFace mobile0.25 detection, from facexlib's own RetinaFace.

    This is the detector the CodeFormer reference pipeline uses, so matching it is what makes an
    aligned crop here the same crop the reference produces.

    The record carries the network's PRE-SUPPRESSION outputs — `output` is the box offsets, with
    `scores` and `landmark_offsets` beside them — because a detection-level comparison alone cannot
    separate a network mismatch from a decoding or threshold difference. `detections` carries the
    reference's own decoded, suppressed result for the end-to-end check.
    """
    import numpy as np
    import torch
    from facexlib.detection.retinaface import RetinaFace

    model = RetinaFace(network_name="mobile0.25", device=torch.device("cpu"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # The reference reads BGR 0...255 through OpenCV and subtracts a per-channel mean.
    rgb = (image * 255.0).astype(np.float32)
    bgr = rgb[:, :, ::-1].copy()
    tensor = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0) - model.mean_tensor

    with torch.no_grad():
        loc, conf, landmarks, priors = model._RetinaFace__detect_faces(tensor)
        detections = model.detect_faces((bgr).astype(np.float32), conf_threshold=0.8, nms_threshold=0.4)

    globals()["_extra"] = {
        "scores": conf.squeeze(0).contiguous(),
        "landmark_offsets": landmarks.squeeze(0).contiguous(),
        "priors": priors.contiguous(),
        "detections": torch.from_numpy(np.asarray(detections, dtype=np.float32)).contiguous(),
    }
    return loc.squeeze(0).contiguous()



def run_qwen3(image, checkpoint):
    """Qwen3 dense decoder logits, from transformers' own Qwen3ForCausalLM.

    `--checkpoint` is the released model DIRECTORY. Needs transformers >= 4.51, which is newer than the
    one the vision oracles run under; `oracle_environments` in the validation manifest records where
    that interpreter lives.

    The record carries the token ids both sides read, the logits for every position of the prompt (the
    prefill, which is where an attention or normalization mistake shows), and the greedy continuation
    the model produces — a token-level match is the end-to-end check and the logits say where a
    mismatch came from.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # E2B fits at float32; E4B's 16 GB of bf16 weights would double past this machine's RAM, so
    # IK_GEMMA_DTYPE=bfloat16 runs the oracle at the released precision instead. The Swift side then
    # loads at `.checkpoint` precision, and the parity threshold is the half-precision one.
    dtype = getattr(torch, os.environ.get("IK_GEMMA_DTYPE", "float32"))
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype).eval()

    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        logits = model(ids).logits[0]                       # [tokens, vocabulary]
        generated = model.generate(ids, max_new_tokens=16, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    continuation = generated[0, ids.shape[1]:]

    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    return logits.contiguous()


# The query/document pair the Qwen3-Embedding record is measured on. The Swift side declares the same
# strings, so the tokenizer-agreement check reproduces the reference's ids from the text.
_EMBED_TASK = "Given a web search query, retrieve relevant passages that answer the query"
_EMBED_QUERY = f"Instruct: {_EMBED_TASK}\nQuery:What is the capital of China?"
_EMBED_DOCUMENT = "The capital of China is Beijing."


def run_qwen3_embedding(image, checkpoint):
    """Qwen3-Embedding embeddings, from the model card's own transformers recipe.

    `--checkpoint` is the released `Qwen/Qwen3-Embedding-0.6B` directory. The embedder is the base
    `Qwen3Model` (AutoModel, no lm_head) read at its last hidden state, pooled at the LAST token — the
    tokenizer appends `<|endoftext|>`, whose position is what the pooling reads — and L2-normalized. It
    needs transformers >= 4.51, the same interpreter the qwen3 mode uses.

    The record carries the query embedding as `output` and the document embedding beside it, plus the
    reference's own token ids for both: the network check feeds those ids and compares the embedding,
    while a separate tokenizer-agreement check reproduces the ids from the shared text.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    model = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).eval()

    def embed(text):
        batch = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state          # [1, tokens, hidden]
        pooled = hidden[:, -1]                                  # last-token pooling, single sequence
        embedding = F.normalize(pooled, p=2, dim=1)[0]
        return embedding, batch["input_ids"][0]

    query_embedding, query_ids = embed(_EMBED_QUERY)
    document_embedding, document_ids = embed(_EMBED_DOCUMENT)

    globals()["_extra"] = {
        "query_tokens": query_ids.to(torch.int32).contiguous(),
        "document_tokens": document_ids.to(torch.int32).contiguous(),
        "document_embedding": document_embedding.contiguous(),
        # The retrieval score the two embeddings produce, for an end-to-end check that survives a
        # per-embedding drift the cosine would still call a match.
        "score": torch.dot(query_embedding, document_embedding).reshape(1).contiguous(),
    }
    return query_embedding.contiguous()


# The query/document pair the EmbeddingGemma record is measured on, each with the model's own task
# prompt prepended. The Swift side declares the same strings.
_EG_QUERY = "task: search result | query: What is the capital of China?"
_EG_DOCUMENT = "title: none | text: The capital of China is Beijing."


def run_embeddinggemma(image, checkpoint):
    """EmbeddingGemma-300M embeddings, from the sentence-transformers pipeline over transformers' own
    Gemma3TextModel.

    `--checkpoint` is the released directory (the ungated `unsloth/embeddinggemma-300m` mirror). The
    backbone is Gemma3TextModel with `use_bidirectional_attention` (no causal mask), read at its last
    hidden state, mean-pooled over every token, run through the two Dense projections (768 -> 3072 ->
    768, no bias, Identity activation) the sentence-transformers head carries, and L2-normalized. Needs
    transformers >= 4.56 (bidirectional Gemma3); the `llm` oracle interpreter has it.

    The record carries the query embedding as `output`, the document embedding beside it, both token
    id sequences, the retrieval score, and the backbone's per-layer hidden states for the query so a
    divergence is located to a layer rather than guessed from the embedding.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).eval()
    dense2 = load_file(os.path.join(checkpoint, "2_Dense", "model.safetensors"))["linear.weight"]
    dense3 = load_file(os.path.join(checkpoint, "3_Dense", "model.safetensors"))["linear.weight"]

    def embed(text, record_layers=False):
        batch = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**batch, output_hidden_states=record_layers)
        pooled = out.last_hidden_state.mean(dim=1)          # mean pooling, single unpadded sequence
        projected = (pooled @ dense2.T) @ dense3.T          # the Dense bottleneck, no bias
        embedding = F.normalize(projected, p=2, dim=1)[0]
        return embedding, batch["input_ids"][0], (out.hidden_states if record_layers else None)

    query_embedding, query_ids, layers = embed(_EG_QUERY, record_layers=True)
    document_embedding, document_ids, _ = embed(_EG_DOCUMENT)

    extra = {
        "query_tokens": query_ids.to(torch.int32).contiguous(),
        "document_tokens": document_ids.to(torch.int32).contiguous(),
        "document_embedding": document_embedding.contiguous(),
        "score": torch.dot(query_embedding, document_embedding).reshape(1).contiguous(),
    }
    for index, hidden in enumerate(layers):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    globals()["_extra"] = extra
    return query_embedding.contiguous()


# The query and the two documents the reranker record is measured on: one relevant, one not, so the
# test checks the scores match AND that the reranker orders them.
_RERANK_QUERY = "What is the capital of France?"
# Long enough that the pair exceeds the 128-token local attention window (64 either side), so the
# sliding-window layers are exercised by the isolation harness rather than degenerating to global.
_RERANK_RELEVANT = ("Paris is the capital and most populous city of France. Situated on the river Seine "
                    "in the north of the country, it has been a major European centre of finance, "
                    "diplomacy, commerce, fashion, and art for centuries. The city is home to landmarks "
                    "such as the Eiffel Tower, the Louvre, and the cathedral of Notre-Dame, and its "
                    "metropolitan area is among the largest in Europe.")
_RERANK_IRRELEVANT = "The Great Barrier Reef is the world's largest coral reef system, off Australia."


def run_modernbert_reranker(image, checkpoint):
    """A ModernBERT cross-encoder reranker's relevance score, from transformers' own
    ModernBertForSequenceClassification.

    `--checkpoint` is the released `Alibaba-NLP/gte-reranker-modernbert-base` directory. The model reads
    a `[CLS] query [SEP] document [SEP]` pair through the bidirectional encoder, mean-pools, runs a
    prediction head (dense + gelu + LayerNorm), and a single-logit classifier gives the score.

    The record carries the token ids of both pairs, the relevant pair's per-layer hidden states for the
    isolation harness, and both scores, so the test checks the arithmetic and the ordering.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, dtype=torch.float32).eval()

    def score(document, record_layers=False):
        inputs = tokenizer(_RERANK_QUERY, document, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=record_layers)
        return out.logits[0], inputs["input_ids"][0], (out.hidden_states if record_layers else None)

    relevant, relevant_ids, layers = score(_RERANK_RELEVANT, record_layers=True)
    irrelevant, irrelevant_ids, _ = score(_RERANK_IRRELEVANT)

    extra = {
        "relevant_tokens": relevant_ids.to(torch.int32).contiguous(),
        "irrelevant_tokens": irrelevant_ids.to(torch.int32).contiguous(),
        "irrelevant_score": irrelevant.contiguous(),
    }
    for index, hidden in enumerate(layers):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    globals()["_extra"] = extra
    return relevant.contiguous()


def run_smolvlm(image, checkpoint):
    """A SmolVLM2 vision-language forward, from transformers' own SmolVLMForConditionalGeneration.

    `--checkpoint` is the released `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` directory. The image is
    the shared plate, resized to the processor's own tiling; the SigLIP vision encoder embeds each tile,
    the pixel-shuffle connector projects them to the decoder width, and the Llama decoder reads the text
    with the projected vision tokens spliced in at the image-token positions.

    The record carries the input ids, the processor's pixel values (so the Swift side skips the image
    processor and measures the network), the vision encoder's per-tile last hidden state and the
    connector's output for the isolation harness, the logits, and the greedy continuation.

    Needs Pillow, torchvision, and num2words for the processor; the `llm` oracle interpreter has them.
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()

    pil = Image.fromarray((image * 255).astype(np.uint8))
    prompt = "User:<image>What is in this image?<end_of_utterance>\nAssistant:"
    inputs = processor(text=prompt, images=[pil], return_tensors="pt")

    pixel_values = inputs["pixel_values"]                       # [1, tiles, 3, 512, 512]
    flat = pixel_values.view(-1, *pixel_values.shape[2:])       # [tiles, 3, 512, 512]
    patch = model.config.vision_config.patch_size
    grid = flat.shape[-1] // patch
    patch_attention_mask = torch.ones(flat.shape[0], grid, grid, dtype=torch.bool)

    vm = model.model.vision_model
    with torch.no_grad():
        embeddings = vm.embeddings(pixel_values=flat, patch_attention_mask=patch_attention_mask)
        layer0 = vm.encoder.layers[0](embeddings, attention_mask=None)[0]
        vision = vm(pixel_values=flat, patch_attention_mask=patch_attention_mask).last_hidden_state
        features = model.model.connector(vision)               # [tiles, 64, decoder width]
        logits = model(**inputs).logits[0]                     # [sequence, vocabulary]
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1]:]

    globals()["_extra"] = {
        "input_ids": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "pixel_values": flat.contiguous(),
        "vision_embeddings": embeddings.contiguous(),
        "vision_layer0": layer0.contiguous(),
        "vision_hidden": vision.contiguous(),
        "image_features": features.contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
        "image_token_id": torch.tensor([model.config.image_token_id], dtype=torch.int32),
    }
    return logits.contiguous()


def run_qwen3vl(image, checkpoint):
    """Qwen3-VL's vision tower, from transformers' own Qwen3VLForConditionalGeneration.

    `--checkpoint` is the released `Qwen/Qwen3-VL-2B-Instruct` directory. The 2D-rotary ViT embeds each
    patch, a merger folds 2×2 patches to the decoder width, and the deepstack heads (layers 5/11/17)
    produce three feature maps. The record carries the processor's pixel values and grid, the vision
    seams for the isolation harness (patch embedding, the added position embedding, the 2D rotary table),
    and the merged output plus the three deepstack features, plus the input ids and logits.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()

    pil = Image.fromarray((image * 255).astype(np.uint8))
    prompt = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
              "<|im_end|>\n<|im_start|>assistant\n")
    inputs = processor(text=[prompt], images=[pil], return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    visual = model.model.visual
    with torch.no_grad():
        patch = visual.patch_embed(pixel_values)
        positions = visual.fast_pos_embed_interpolate(grid_thw)
        rotary = visual.rot_pos_emb(grid_thw)
        embeds, deepstack = visual(pixel_values, grid_thw=grid_thw)
        logits = model(**inputs).logits[0]
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1]:]

    extra = {
        "pixel_values": pixel_values.contiguous(),
        "image_grid_thw": grid_thw.to(torch.int32).contiguous(),
        "patch_embed": patch.contiguous(),
        "pos_embeds": positions.contiguous(),
        "rotary_pos_emb": rotary.contiguous(),
        "vision_output": embeds.contiguous(),
        "input_ids": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    for index, feature in enumerate(deepstack):
        extra[f"deepstack_{index}"] = feature.contiguous()
    globals()["_extra"] = extra
    return logits.contiguous()


def run_gguf(image, checkpoint):
    """Reference dequantization of a GGUF model's tensors, from the `gguf` package.

    `--checkpoint` is a `.gguf` file. For the first tensor of each block-quant type the native reader
    implements, this records the dequantized values (capped so the record stays small) under the type's
    name. The Swift side reads the same GGUF, picks the first tensor of each type in file order — the
    same one — dequantizes it, and compares. `gguf` is the ground truth, the way llama.cpp is for the
    format.
    """
    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(checkpoint)
    cap = 262_144
    seen = set()
    extra = {}
    for tensor in reader.tensors:
        name = tensor.tensor_type.name
        if name in ("Q4_K", "Q6_K", "Q8_0", "Q5_0", "F32") and name not in seen:
            seen.add(name)
            values = dequantize(tensor.data, tensor.tensor_type).flatten()[:cap].astype(np.float32)
            extra[name] = torch.from_numpy(np.ascontiguousarray(values))
    globals()["_extra"] = extra
    return torch.zeros(1)


def _tiny_decoder_record(model, tokens, release_name=lambda key: key):
    """Logits, every hidden state, and the weights in release naming, for a tiny random model."""
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    for key, value in model.state_dict().items():
        extra[f"w::{release_name(key)}"] = (value.float() if value.is_floating_point()
                                            else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def _randomized(model, seed=11, scale=0.05):
    torch.manual_seed(seed)
    state = model.state_dict()
    for key in sorted(state):
        if state[key].is_floating_point():
            state[key] = torch.randn(state[key].shape) * scale
    model.load_state_dict(state)
    return model.eval().float()


def run_qwen3_moe(image, checkpoint):
    """The Qwen3-MoE decoder's arithmetic, from transformers' own Qwen3MoeForCausalLM, at a tiny
    random configuration.

    The released sizes (30B-A3B and up) do not fit this machine at any precision the oracle can run,
    so the mixture-of-experts feed-forward — router softmax over every expert, top-k selection with
    the selected weights renormalized (`norm_topk_prob`), the experts' SwiGLU, the weighted sum — is
    measured at a size that does, with the dense attention around it. Every hidden state is recorded
    so a divergence is located to a layer. The weights are saved under the release's own names
    (`mlp.experts.N.gate_proj.weight`), which the Swift loader stacks. `checkpoint` is unused.
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        hidden_size=64, num_hidden_layers=3, vocab_size=128, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=96, moe_intermediate_size=32,
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
        mlp_only_layers=[], rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        max_position_embeddings=64, attention_bias=False)
    model = _randomized(Qwen3MoeForCausalLM(config))
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_mixtral(image, checkpoint):
    """The Mixtral decoder's arithmetic, from transformers' own MixtralForCausalLM, at a tiny random
    configuration.

    Mixtral's block takes the softmax over the SELECTED experts' logits, which equals Qwen3-MoE's
    softmax-over-all followed by renormalization; this record is what holds the Swift module, which
    computes the latter form for both, to Mixtral's own arithmetic. Its tensor names differ
    (`block_sparse_moe.experts.N.w1/w3/w2`), so the record also measures the loader's rename. No
    sliding window, as the released 8x7B has none. `checkpoint` is unused.
    """
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(
        hidden_size=64, num_hidden_layers=3, vocab_size=128, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=32, num_local_experts=4,
        num_experts_per_tok=2, rms_norm_eps=1e-6, rope_theta=10000.0, sliding_window=None,
        max_position_embeddings=64, tie_word_embeddings=False)
    model = _randomized(MixtralForCausalLM(config), seed=13)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_gemma4(image, checkpoint):
    """Gemma 4 text-decoder logits, from transformers' own Gemma4 implementation.

    `--checkpoint` is the released model DIRECTORY. Gemma 4 is in no released transformers, so this
    runs under its own interpreter — see `oracle_environments` in the validation manifest.

    The record carries the token ids both sides read, the decoder's logits for the whole prompt, and
    the greedy continuation. The decoder is what this port implements; the release's vision and audio
    towers are not.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()

    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(ids, max_new_tokens=12, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    continuation = generated[0, ids.shape[1]:]

    # PER-LAYER ISOLATION. `hidden_states` is the embedding output followed by each layer's, so
    # `hidden.0` is what enters layer 0 and `hidden.N+1` is what layer N produced. Comparing them one
    # at a time says WHICH layer first diverges, which a whole-model cosine cannot.
    extra = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.float().contiguous()



def run_gemma4_moe(image, checkpoint):
    """The Gemma 4 mixture-of-experts block (the 26B-A4B family) at a tiny random configuration,
    from transformers' own Gemma4ForCausalLM.

    The released mixture does not fit this machine, so the routed branch — the scale-free router norm,
    the learned per-channel and per-expert scales, the softmax and top-k, and the fused-projection
    experts, all summed BESIDE the dense feed-forward — is measured at a size that does, over the full
    Gemma 4 block (per-layer input embeddings, the sandwich norms, dual rotary). Every hidden state is
    recorded so a divergence is located to a layer. The per-layer input vocabulary is shrunk to the
    token vocabulary here (`vocab_size_per_layer_input`) so the record stays small; the released model
    uses the 262144 default. `checkpoint` is unused.
    """
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    config = Gemma4TextConfig(
        hidden_size=64, num_hidden_layers=2, vocab_size=131, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=96,
        hidden_size_per_layer_input=32, vocab_size_per_layer_input=131,
        num_kv_shared_layers=0, sliding_window=64,
        layer_types=["sliding_attention", "full_attention"], rms_norm_eps=1e-6,
        max_position_embeddings=64, tie_word_embeddings=True,
        enable_moe_block=True, num_experts=6, top_k_experts=2, moe_intermediate_size=48,
        hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=None)
    model = _randomized(Gemma4ForCausalLM(config), seed=17)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    # `lm_head.weight` is tied to `embed_tokens.weight` (shared storage), which safetensors refuses to
    # save twice; the Swift net ties, so the head is dropped and reconstructed from the embedding.
    for key, value in model.state_dict().items():
        if key == "lm_head.weight":
            continue
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def run_gguf_lm(image, checkpoint):
    """End-to-end GGUF language-model logits, from transformers loading the GGUF directly.

    `--checkpoint` is the GGUF FILE. transformers dequantizes the GGUF and runs its own llama/qwen
    decoder, which is the reference the MLX GGUF loader is held to end to end: the tokenizer's ids
    (recorded so the Swift side feeds the identical sequence and its own tokenizer is checked
    separately), the logits over the prompt, and the greedy continuation. The dequantization is the
    canonical one, so a quantized model still has a single correct answer here.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = os.path.dirname(checkpoint)
    filename = os.path.basename(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(directory, gguf_file=filename)
    model = AutoModelForCausalLM.from_pretrained(directory, gguf_file=filename, dtype=torch.float32).eval()

    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids)
        generated = model.generate(ids, max_new_tokens=10, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": generated[0, ids.shape[1]:].to(torch.int32).contiguous(),
    }
    return out.logits[0].float().contiguous()


def run_gemma4_unified(image, checkpoint):
    """The Gemma 4 unified text decoder (the 12B `gemma4_unified_text`) at a tiny random configuration,
    from transformers' own Gemma4UnifiedForCausalLM.

    A DIFFERENT decoder from the E-series `gemma4_text`: no per-layer input embeddings and no mixture,
    but a scale-free VALUE norm beside the query and key norms, per-layer head widths (a full layer runs
    a 512-wide head where a sliding one runs its own), attention scaling of 1.0, and the sandwich norms
    with a per-layer scalar. Every hidden state is recorded for the isolation harness; the weights are
    saved in the release naming (`model.layers.N.self_attn.v_proj.weight`), and `lm_head` is dropped as
    tied. `checkpoint` is unused.
    """
    import torch
    from transformers import Gemma4UnifiedTextConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM

    config = Gemma4UnifiedTextConfig(
        hidden_size=64, num_hidden_layers=3, vocab_size=131, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=96, sliding_window=64,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        rms_norm_eps=1e-6, max_position_embeddings=64, tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=None)
    model = _randomized(Gemma4UnifiedForCausalLM(config), seed=19)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    for key, value in model.state_dict().items():
        if key == "lm_head.weight":
            continue
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def run_gemma4_vision(image, checkpoint):
    """The Gemma 4 vision encoder at a tiny random configuration, from transformers' own
    Gemma4VisionModel (patch embedder + bidirectional sandwich encoder, before the pooler).

    The vision tower embeds flattened patches through one linear projection, adds a learned 2-D
    position embedding (an x-table and a y-table summed per patch), and runs the same sandwich block
    the text decoder does but bidirectional and with NO rotary — the positions are the learned
    embeddings. `pixel_values` are the flattened patches `[batch, patches, 3·patch²]`, `pixel_position_ids`
    their `(x, y)` grid. The record carries both inputs, the encoder's last hidden state, and the
    weights in the release naming (the projections live under `.linear`, a `Gemma4ClippableLinear`).
    `checkpoint` is unused.
    """
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    config = Gemma4VisionConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=8, intermediate_size=48, patch_size=4, position_embedding_size=16,
        pooling_kernel_size=2, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
        use_clipped_linears=False, standardize=False)
    model = _randomized(Gemma4VisionModel(config), seed=23)

    torch.manual_seed(3)
    side = 4
    pixel_values = torch.randn(1, side * side, 3 * config.patch_size ** 2)
    grid = [[x, y] for y in range(side) for x in range(side)]
    position_ids = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
    padding = (position_ids == -1).all(dim=-1)
    with torch.no_grad():
        embeds = model.patch_embedder(pixel_values, position_ids, padding)
        encoded = model.encoder(inputs_embeds=embeds, attention_mask=~padding,
                                pixel_position_ids=position_ids)
        # The full tower: encoder, then the position-based average pooler and the sqrt(hidden) scaling.
        pooled = model(pixel_values, pixel_position_ids=position_ids).last_hidden_state
    extra = {
        "pixel_values": pixel_values[0].float().contiguous(),
        "position_ids": position_ids[0].to(torch.int32).contiguous(),
        "pooled": pooled.float().contiguous(),
    }
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return encoded.last_hidden_state[0].float().contiguous()


def run_gemma4_vision_real(image, checkpoint):
    """The Gemma 4 vision path on the RELEASED weights: the vision tower and its multimodal embedder,
    loaded selectively from the tri-modal `Gemma4ForConditionalGeneration` checkpoint (the E2B release
    carries a vision tower, an audio tower, and the decoder together).

    `--checkpoint` is the release directory. Random pixel values over a real patch grid are fed to both
    sides, so the tower and embedder are compared on real weights without the resize entering. Records
    the pixel values, the position grid, the pooled soft tokens, and the embedder's projection.
    """
    import json
    import os
    import torch
    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    vision_config = Gemma4VisionConfig(**config["vision_config"])
    text_config = Gemma4TextConfig(**config["text_config"])
    vision = Gemma4VisionModel(vision_config).eval()
    embedder = Gemma4MultimodalEmbedder(vision_config, text_config).eval()

    vision_state, embed_state = {}, {}
    with safe_open(os.path.join(checkpoint, "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("model.vision_tower."):
                vision_state[key[len("model.vision_tower."):]] = handle.get_tensor(key).float()
            elif key.startswith("model.embed_vision."):
                embed_state[key[len("model.embed_vision."):]] = handle.get_tensor(key).float()
    vision.load_state_dict(vision_state, strict=True)
    embedder.load_state_dict(embed_state, strict=True)

    side = 6                                                     # 36 patches, 4 soft tokens at pooling 3
    torch.manual_seed(3)
    pixel_values = torch.randn(1, side * side, 3 * vision_config.patch_size ** 2)
    grid = [[x, y] for y in range(side) for x in range(side)]
    position_ids = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
    padding = (position_ids == -1).all(dim=-1)
    seams = {}
    vision.encoder.layers[0].register_forward_hook(
        lambda m, i, o: seams.__setitem__("layer0", (o[0] if isinstance(o, tuple) else o).detach()))
    vision.encoder.layers[0].self_attn.register_forward_hook(
        lambda m, i, o: seams.__setitem__("attn0", (o[0] if isinstance(o, tuple) else o).detach()))
    vision.encoder.layers[0].mlp.register_forward_hook(
        lambda m, i, o: seams.__setitem__("mlp0", o.detach()))
    vision.encoder.layers[0].input_layernorm.register_forward_hook(
        lambda m, i, o: seams.__setitem__("inorm0", o.detach()))
    with torch.no_grad():
        embeds = vision.patch_embedder(pixel_values, position_ids, padding)
        encoded = vision.encoder(inputs_embeds=embeds, attention_mask=~padding,
                                 pixel_position_ids=position_ids).last_hidden_state
        pooled = vision(pixel_values, pixel_position_ids=position_ids).last_hidden_state
        projected = embedder(pooled)

    globals()["_extra"] = {
        "pixel_values": pixel_values[0].float().contiguous(),
        "position_ids": position_ids[0].to(torch.int32).contiguous(),
        "patch_embed": embeds[0].float().contiguous(),
        "inorm0": seams["inorm0"][0].float().contiguous(),
        "attn0": seams["attn0"][0].float().contiguous(),
        "mlp0": seams["mlp0"][0].float().contiguous(),
        "layer0": seams["layer0"][0].float().contiguous(),
        "encoded": encoded[0].float().contiguous(),
        "pooled": pooled.float().contiguous(),
    }
    return projected.float().contiguous()


def run_gemma4_audio_real(image, checkpoint):
    """The Gemma 4 audio path on the RELEASED weights: the audio Conformer and its multimodal embedder,
    loaded selectively from the tri-modal checkpoint. Random mel features are fed to both sides, so the
    tower and the projection into the decoder's space are compared on real weights. `--checkpoint` is
    the release directory.
    """
    import json
    import os
    import torch
    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    audio_config = Gemma4AudioConfig(**config["audio_config"])
    text_config = Gemma4TextConfig(**config["text_config"])
    audio = Gemma4AudioModel(audio_config).eval()
    embedder = Gemma4MultimodalEmbedder(audio_config, text_config).eval()

    audio_state, embed_state = {}, {}
    with safe_open(os.path.join(checkpoint, "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("model.audio_tower."):
                audio_state[key[len("model.audio_tower."):]] = handle.get_tensor(key).float()
            elif key.startswith("model.embed_audio."):
                embed_state[key[len("model.embed_audio."):]] = handle.get_tensor(key).float()
    audio.load_state_dict(audio_state, strict=True)
    embedder.load_state_dict(embed_state, strict=True)

    torch.manual_seed(4)
    features = torch.randn(1, 64, 128)
    mask = torch.ones(1, 64, dtype=torch.float32)
    with torch.no_grad():
        output = audio(features, mask, return_dict=True)
        projected = embedder(output.last_hidden_state)

    globals()["_extra"] = {
        "features": features[0].float().contiguous(),
        "encoded": output.last_hidden_state[0].float().contiguous(),
    }
    return projected[0].float().contiguous()


def run_gemma4_conditional_real(image, checkpoint):
    """The FULL Gemma4ForConditionalGeneration on the released E2B weights: an image and a
    placeholder-carrying prompt through the vision tower, the embedder, the splice, and the decoder,
    end to end. `--checkpoint` is the release directory.

    Random pixel values over a 6x6 grid (36 patches -> 4 soft tokens at pooling 3) fill four image
    placeholder tokens. Records the token ids, the pixel values, the position grid, and the decoder's
    logits over the fused sequence.
    """
    import json
    import os
    import torch
    from transformers import AutoModelForImageTextToText

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    image_token = config["image_token_id"]
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()

    torch.manual_seed(3)
    side = 6
    pixel_values = torch.randn(1, side * side, 3 * config["vision_config"]["patch_size"] ** 2)
    grid = [[x, y] for y in range(side) for x in range(side)]
    position_ids = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor([[2] + [image_token] * 4 + [1234, 5678, 91011]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids, pixel_values=pixel_values,
                       image_position_ids=position_ids).logits[0]

    globals()["_extra"] = {
        "tokens": input_ids[0].to(torch.int32).contiguous(),
        "pixel_values": pixel_values[0].float().contiguous(),
        "position_ids": position_ids[0].to(torch.int32).contiguous(),
    }
    return logits.float().contiguous()


def run_gemma4_embedder(image, checkpoint):
    """The Gemma 4 multimodal embedder — a scale-free RMS norm and a projection into the language
    model's space — from transformers' own Gemma4MultimodalEmbedder at a tiny configuration.

    This is the piece that turns a tower's soft tokens into embeddings the decoder splices in. Records
    the input soft tokens, the projected output, and the weights in the release naming. `checkpoint`
    is unused.
    """
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as G
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig

    vision = Gemma4VisionConfig(hidden_size=32, rms_norm_eps=1e-6)
    text = Gemma4TextConfig(hidden_size=48)
    embedder = _randomized(G.Gemma4MultimodalEmbedder(vision, text), seed=31)
    soft = torch.randn(1, 5, 32)
    with torch.no_grad():
        out = embedder(soft)
    extra = {"soft": soft[0].float().contiguous()}
    for key, value in embedder.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return out[0].float().contiguous()


def run_gemma4_mel(image, checkpoint):
    """The Gemma 4 audio front end: raw audio to log-mel features, from transformers' own
    Gemma4AudioFeatureExtractor. `checkpoint` is unused.

    A deterministic waveform is recorded alongside its features so the Swift front end runs the exact
    same samples. This is a pure preprocessing step (framing, Hann window, magnitude FFT, HTK mel bank,
    log), so it matches to floating precision.
    """
    import numpy as np
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    extractor = Gemma4AudioFeatureExtractor()
    rng = np.random.RandomState(7)
    waveform = (0.3 * np.sin(np.linspace(0, 80, 4000)) + 0.05 * rng.randn(4000)).astype(np.float32)
    mask = np.ones(waveform.shape[0], dtype=np.float32)
    features, _ = extractor._extract_spectrogram(waveform[None, :], mask)

    import torch
    globals()["_extra"] = {"waveform": torch.tensor(waveform, dtype=torch.float32).contiguous()}
    return torch.tensor(np.asarray(features), dtype=torch.float32).contiguous()


def run_gemma4_audio(image, checkpoint):
    """The Gemma 4 audio Conformer at a tiny random configuration, from transformers' own
    Gemma4AudioModel.

    The most involved Gemma 4 tower: a 2-D convolutional subsampler, then Conformer layers — a macaron
    feed-forward, a blocked relative-position attention (Transformer-XL rel-shift, a per-dimension
    softplus scale, a logit softcap), a light depthwise convolution, a second macaron feed-forward, and
    sandwich norms — and an output projection. The record carries the mel input, the post-subsample
    hidden state, the parameter-free relative position encoding, the blocked attention mask, and the
    final output, so the subsampler and the Conformer are each isolated. The blocked mask is fed to the
    Swift side rather than reconstructed there, which isolates the attention arithmetic from the mask
    construction. `checkpoint` is unused.
    """
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as G
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig

    config = Gemma4AudioConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, conv_kernel_size=3,
        attention_chunk_size=4, attention_context_left=3, attention_context_right=0, hidden_act="silu",
        residual_weight=0.5, gradient_clipping=1e10, attention_logit_cap=50.0, rms_norm_eps=1e-6,
        subsampling_conv_channels=[8, 16], output_proj_dims=48)
    model = _randomized(G.Gemma4AudioModel(config), seed=29)
    # `_randomized` also perturbs the clippable-linear clamp buffers; the released model ships them at
    # ±inf (identity), so reset them, matching the port which does not model the clamps.
    for name, buffer in model.named_buffers():
        if name.endswith(("input_min", "output_min")):
            buffer.fill_(-float("inf"))
        elif name.endswith(("input_max", "output_max")):
            buffer.fill_(float("inf"))

    torch.manual_seed(5)
    features = torch.randn(1, 32, config.subsampling_conv_channels[0])
    # Capture the blocked attention mask the model builds by wrapping its own converter.
    captured = {}
    original = model._convert_4d_mask_to_blocked_5d
    def wrapped(mask_4d):
        result = original(mask_4d)
        captured["mask"] = result
        return result
    model._convert_4d_mask_to_blocked_5d = wrapped
    # Hooks to capture the first layer's sub-component outputs, for the isolation harness.
    seams = {}
    handles = []
    handles.append(model.layers[0].feed_forward1.register_forward_hook(
        lambda m, i, o: seams.__setitem__("ff1", o.detach())))
    handles.append(model.layers[0].self_attn.register_forward_hook(
        lambda m, i, o: seams.__setitem__("attn", o[0].detach())))
    handles.append(model.layers[0].lconv1d.register_forward_hook(
        lambda m, i, o: seams.__setitem__("lconv", o.detach())))
    handles.append(model.layers[0].register_forward_hook(
        lambda m, i, o: seams.__setitem__("layer0", o.detach())))
    with torch.no_grad():
        subsampled, _ = model.subsample_conv_projection(features, None)
        position_embeddings = model.rel_pos_enc(subsampled)
        out = model(features)
    for h in handles:
        h.remove()
    mask_5d = captured.get("mask")

    extra = {
        "input_features": features[0].float().contiguous(),
        "subsampled": subsampled[0].float().contiguous(),
        "position_embeddings": position_embeddings.float().contiguous(),
    }
    if mask_5d is not None:
        extra["attention_mask"] = mask_5d[0].to(torch.int32).contiguous()   # [1, num_blocks, chunk, context]
    for name, value in seams.items():
        extra[f"seam_{name}"] = value[0].float().contiguous()
    for key, value in model.state_dict().items():
        # The clippable-linear clamp buffers are identity (±inf); skip them.
        if key.endswith(("input_min", "input_max", "output_min", "output_max")):
            continue
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.last_hidden_state[0].float().contiguous()


def run_qwen3_5(image, checkpoint):
    """Qwen3.5 hybrid-decoder logits, from transformers' own Qwen3_5 implementation.

    The family Qwen3.5, Qwen3.6, and Qwen3.8 share: three gated delta-rule recurrence layers then one
    gated full-attention layer. 4B is the smallest release, and the only one that fits here.

    Records the per-layer hidden states as well as the logits, so the isolation harness can locate a
    divergence rather than only detect one.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()

    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(ids, max_new_tokens=12, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)

    extra = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": generated[0, ids.shape[1]:].to(torch.int32).contiguous(),
    }
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.float().contiguous()


def run_clip_text(image):
    """CLIP ViT-B/32 text embedding, L2-normalized, through transformers.

    The record carries `tokens` (the ids the reference tokenized) so the Swift side embeds exactly the
    same sequence — the port takes ids rather than text, because a byte-level BPE vocabulary is a
    load-time artifact rather than part of the network.
    """
    from transformers import CLIPModel, CLIPTokenizer

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    tokenizer = CLIPTokenizer.from_pretrained(name)
    encoded = tokenizer(["a photograph of a dog running on grass"], padding="max_length",
                        max_length=77, return_tensors="pt")
    with torch.no_grad():
        features = model.get_text_features(**encoded)
    features = features / features.norm(dim=-1, keepdim=True)
    globals()["_extra"] = {"tokens": encoded["input_ids"][0].to(torch.int32).contiguous()}
    return features[0].contiguous()


def run_siggraph17(image, checkpoint):
    """siggraph17 colorization, `[H, W, 3]` sRGB in 0...1, from richzhang's own `siggraph17.py`.

    IK_REF_SRC holds `siggraph17.py` and `base_color.py`. The hint and its mask are left empty, which
    is how the reference colorizes automatically. The record's `ab` carries the network's own output
    before the lightness goes back, so a mismatch says network or Lab conversion rather than both.
    """
    import types
    from skimage import color

    ipython = types.ModuleType("IPython")
    ipython.embed = lambda *args, **kwargs: None
    sys.modules["IPython"] = ipython

    _import_reference(_reference_source(), "siggraph_ref", "base_color")
    module = _import_reference(_reference_source(), "siggraph_ref", "siggraph17")
    model = module.SIGGRAPHGenerator().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    lab = color.rgb2lab(image).astype(np.float32)
    lightness = torch.from_numpy(lab[:, :, :1]).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        ab = model(lightness)                                   # already unnormalized by the model
    colorized = color.lab2rgb(np.concatenate([lab[:, :, :1], ab[0].permute(1, 2, 0).numpy()], axis=2))

    globals()["_extra"] = {"lightness": torch.from_numpy(lab[:, :, 0]).contiguous(),
                           "ab": ab[0].permute(1, 2, 0).contiguous()}
    return torch.from_numpy(colorized.astype(np.float32)).contiguous()


def run_bisenetv2(image, checkpoint):
    """BiSeNetV2 class logits, `[H, W, classes]`, from CoinCheung's own `bisenetv2.py`.

    The released `model_final_v2.pth` predates the repository's current `SegmentHead`: it emits
    `classes × upFactor²` channels and pixel-shuffles, where master now emits `classes` and
    interpolates. Everything before the heads is unchanged, so the head is replaced here rather than
    hunting a historical revision — and the substitution is visible instead of buried in a pinned
    commit. The auxiliary heads are training-only and are not built.
    """
    import types
    from torch import nn

    source = os.path.join(_reference_source(), "bisenet")
    module = _import_reference(source, "bisenetv2_ref", "bisenetv2")

    class ShuffleHead(nn.Module):
        """The head the checkpoint was trained with."""

        def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
            super().__init__()
            self.conv = module.ConvBNReLU(in_chan, mid_chan, 3, stride=1)
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_chan, n_classes * up_factor ** 2, 1, 1, 0, bias=True),
                nn.PixelShuffle(up_factor))

        def forward(self, x):
            return self.conv_out(self.conv(x))

    module.SegmentHead = ShuffleHead
    model = module.BiSeNetV2(n_classes=19, aux_mode="eval")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("aux")}, strict=True)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((torch.from_numpy(image).permute(2, 0, 1) - mean) / deviation).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
    return logits[0].permute(1, 2, 0).contiguous()               # [H, W, classes] to match NHWC


def run_rife_v4(image, checkpoint):
    """RIFE v4 interpolation, `[H, W, 3]` in 0...1.

    IK_REF_SRC holds a `rife4/` directory with `rife_arch.py` — a vendored architecture covering the
    v4 generations, used because the version's own `IFNet.py` ships inside the model zip rather than
    in the repository. The released `rife-flownet-4.13.2` weights match its `4.17` branch: four
    blocks, a `Head_417` encoder, ResConv trunk entries, and a shuffled upsampling convolution. Both
    sides run four scales `[8, 4, 2, 1]` and pad to a multiple of 64.
    """
    import os

    import types

    # The vendored file imports one ComfyUI helper at module scope for device selection; nothing on
    # the forward path uses it, so a stub is enough to import the architecture.
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    management = types.ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    sys.modules["comfy"], sys.modules["comfy.model_management"] = comfy, management

    sys.path.insert(0, os.path.join(_reference_source(), "rife4"))
    from rife_arch import IFNet
    from safetensors.torch import load_file

    model = IFNet(arch_ver="4.17")
    model.load_state_dict(load_file(checkpoint), strict=True)
    model.eval()

    shift = (4, 6)
    second = np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)
    frames = [torch.from_numpy(np.ascontiguousarray(plate)).permute(2, 0, 1)[None]
              for plate in (image, second)]
    with torch.no_grad():
        merged = model(frames[0], frames[1], timestep=0.5, scale_list=[8, 4, 2, 1],
                       training=False, fastmode=True, ensemble=False)
    result = merged if torch.is_tensor(merged) else merged[-1]
    globals()["_extra"] = {"frame1": torch.from_numpy(np.ascontiguousarray(second)).contiguous()}
    return result[0].permute(1, 2, 0).contiguous()


def run_sam2_encoder(image, checkpoint):
    """SAM 2's Hiera image encoder, from facebookresearch's own sources.

    The `sam2` package cannot be installed here (it requires Python >= 3.10), so IK_REF_SRC holds a
    `sam2/` directory with the six files the image encoder needs, which do parse under 3.9;
    `iopath` and `sam2.utils.misc` are stubbed because only the video path and an optional checkpoint
    loader reach them. The record carries the finest two FPN levels plus `output` as the vision
    features the mask decoder reads — three seams, so a trunk mismatch and a neck mismatch are told
    apart.
    """
    import types

    for name in ["iopath", "iopath.common"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    file_io = types.ModuleType("iopath.common.file_io")
    file_io.g_pathmgr = None
    sys.modules["iopath.common.file_io"] = file_io
    misc = types.ModuleType("sam2.utils.misc")
    misc.mask_to_box = lambda mask: mask
    sys.modules["sam2.utils.misc"] = misc

    sys.path.insert(0, _reference_source())
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    # IK_SAM2_VARIANT selects the released size. Each one's numbers come from its own config: the
    # base_plus config sets only the width and heads, so the rest are the Hiera defaults, and large
    # overrides every axis.
    variant = os.environ.get("IK_SAM2_VARIANT", "tiny")
    geometry = {
        "tiny":      dict(embed_dim=96,  num_heads=1, stages=(1, 2, 7, 2),
                          global_att_blocks=(5, 7, 9), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
        "base_plus": dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3),
                          global_att_blocks=(12, 16, 20), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(14, 14)),
        "large":     dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4),
                          global_att_blocks=(23, 33, 43), window_spec=(8, 4, 16, 8),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
    }[variant]
    channels = {"tiny": [768, 384, 192, 96], "base_plus": [896, 448, 224, 112],
                "large": [1152, 576, 288, 144]}[variant]
    trunk = Hiera(**geometry)
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256), d_model=256,
                   backbone_channel_list=channels, fpn_top_down_levels=[2, 3],
                   fpn_interp_model="nearest")
    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    encoder.load_state_dict({k[len("image_encoder."):]: v for k, v in state.items()
                             if k.startswith("image_encoder.")}, strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = encoder(tensor)
    levels = out["backbone_fpn"]
    globals()["_extra"] = {"level0": levels[0][0].permute(1, 2, 0).contiguous(),
                           "level1": levels[1][0].permute(1, 2, 0).contiguous()}
    return out["vision_features"][0].permute(1, 2, 0).contiguous()      # [64, 64, 256] NHWC


def run_sam2_decoder(image, checkpoint):
    """SAM 2's prompt encoder and mask decoder, from facebookresearch's own sources.

    Runs the encoder first, then a single positive click at the frame's centre, and dumps the raw
    mask logits for all four tokens plus the IoU and object scores. Comparing every token rather than
    the selected mask keeps a token permutation distinguishable from a genuine drift — the diagnostic
    that resolved SAM 1's decoder.
    """
    import types

    for name in ["iopath", "iopath.common"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    file_io = types.ModuleType("iopath.common.file_io")
    file_io.g_pathmgr = None
    sys.modules["iopath.common.file_io"] = file_io
    misc = types.ModuleType("sam2.utils.misc")
    misc.mask_to_box = lambda mask: mask
    sys.modules["sam2.utils.misc"] = misc

    sys.path.insert(0, _reference_source())
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.prompt_encoder import PromptEncoder
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    from sam2.modeling.sam.transformer import TwoWayTransformer

    # IK_SAM2_VARIANT selects the released size. Each one's numbers come from its own config: the
    # base_plus config sets only the width and heads, so the rest are the Hiera defaults, and large
    # overrides every axis.
    variant = os.environ.get("IK_SAM2_VARIANT", "tiny")
    geometry = {
        "tiny":      dict(embed_dim=96,  num_heads=1, stages=(1, 2, 7, 2),
                          global_att_blocks=(5, 7, 9), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
        "base_plus": dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3),
                          global_att_blocks=(12, 16, 20), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(14, 14)),
        "large":     dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4),
                          global_att_blocks=(23, 33, 43), window_spec=(8, 4, 16, 8),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
    }[variant]
    channels = {"tiny": [768, 384, 192, 96], "base_plus": [896, 448, 224, 112],
                "large": [1152, 576, 288, 144]}[variant]
    trunk = Hiera(**geometry)
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256), d_model=256,
                   backbone_channel_list=channels, fpn_top_down_levels=[2, 3],
                   fpn_interp_model="nearest")
    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1).eval()
    prompt = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                           input_image_size=(1024, 1024), mask_in_chans=16).eval()
    decoder = MaskDecoder(num_multimask_outputs=3, transformer_dim=256,
                          transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048,
                                                        num_heads=8),
                          use_high_res_features=True, pred_obj_scores=True,
                          pred_obj_scores_mlp=True, use_multimask_token_for_obj_ptr=True).eval()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    for module, prefix in [(encoder, "image_encoder."), (prompt, "sam_prompt_encoder."),
                           (decoder, "sam_mask_decoder.")]:
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = encoder(tensor)
        levels = out["backbone_fpn"]
        height, width = image.shape[:2]
        point = torch.tensor([[[width / 2.0, height / 3.0]]], dtype=torch.float32)
        labels = torch.tensor([[1]], dtype=torch.int64)
        sparse, dense = prompt(points=(point, labels), boxes=None, masks=None)
        masks, iou, _, object_score = decoder(
            image_embeddings=out["vision_features"], image_pe=prompt.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True, repeat_image=False,
            # The reference's base model applies these projections before calling the decoder, even
            # though `conv_s0`/`conv_s1` are the decoder's own parameters.
            high_res_features=[decoder.conv_s0(levels[0]), decoder.conv_s1(levels[1])])
    globals()["_extra"] = {"iou": iou[0].contiguous(),
                           "object_score": object_score[0].contiguous(),
                           "sparse": sparse[0].contiguous()}
    return masks[0].contiguous()                                # [tokens, H, W] raw logits


def run_sam2_memory(image, checkpoint):
    """SAM 2's memory encoder and memory attention, from facebookresearch's own sources.

    These are what make SAM 2 a tracker: the encoder folds a frame's features and its predicted mask
    into a compact memory, and the attention conditions the next frame on it. Both are driven here
    with deterministic tensors rather than a real second frame, so the comparison isolates the two
    modules from the tracking loop that would sequence them. The record carries the memory the encoder
    produced (`memory`) and the attention's output (`output`).
    """
    import types

    for name in ["iopath", "iopath.common"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    file_io = types.ModuleType("iopath.common.file_io")
    file_io.g_pathmgr = None
    sys.modules["iopath.common.file_io"] = file_io
    misc = types.ModuleType("sam2.utils.misc")
    misc.mask_to_box = lambda mask: mask
    sys.modules["sam2.utils.misc"] = misc

    sys.path.insert(0, _reference_source())
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.transformer import RoPEAttention

    encoder = MemoryEncoder(out_dim=64,
                            mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
                            fuser=Fuser(CXBlock(dim=256), num_layers=2),
                            position_encoding=PositionEmbeddingSine(num_pos_feats=64)).eval()
    layer = MemoryAttentionLayer(
        activation="relu", dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        self_attention=RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1),
        cross_attention=RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1,
                                      kv_in_dim=64, rope_k_repeat=True, feat_sizes=(32, 32)),
        d_model=256)
    attention = MemoryAttention(d_model=256, pos_enc_at_input=True, layer=layer, num_layers=4).eval()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    for module, prefix in [(encoder, "memory_encoder."), (attention, "memory_attention.")]:
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)

    generator = np.random.default_rng(31)
    features = torch.from_numpy(generator.standard_normal((1, 256, 64, 64)).astype(np.float32))
    # The tracker upsamples the decoder's low-resolution mask to the full frame before encoding it;
    # the downsampler's total stride of 16 is what lands it back on the feature grid.
    mask = torch.from_numpy(generator.standard_normal((1, 1, 1024, 1024)).astype(np.float32))
    with torch.no_grad():
        encoded = encoder(features, mask)
        memory = encoded["vision_features"]                     # [1, 64, 64, 64]
        # The attention works on flattened tokens, memory first as the tracker feeds it.
        current = torch.from_numpy(generator.standard_normal((1, 256, 64, 64)).astype(np.float32))
        # `batch_first=True` describes what the LAYERS want, so `MemoryAttention` takes its inputs
        # SEQUENCE-first and transposes them itself. Handing it batch-first tensors silently makes the
        # tokens the batch, and every token then attends only to itself.
        current_tokens = current.flatten(2).permute(2, 0, 1)     # [tokens, batch, C]
        memory_tokens = memory.flatten(2).permute(2, 0, 1)
        current_pos = torch.from_numpy(generator.standard_normal(current_tokens.shape).astype(np.float32))
        memory_pos = torch.from_numpy(generator.standard_normal(memory_tokens.shape).astype(np.float32))
        out = attention(curr=current_tokens, memory=memory_tokens,
                        curr_pos=current_pos, memory_pos=memory_pos, num_obj_ptr_tokens=0)

    globals()["_extra"] = {"features": features[0].permute(1, 2, 0).contiguous(),
                           "mask": mask[0].permute(1, 2, 0).contiguous(),
                           "memory": memory[0].permute(1, 2, 0).contiguous(),
                           "current": current_tokens[:, 0].contiguous(),
                           "current_pos": current_pos[:, 0].contiguous(),
                           "memory_pos": memory_pos[:, 0].contiguous()}
    return out[:, 0].contiguous()                               # [tokens, 256]


def run_videosr(image, checkpoint):
    """BasicVSR ×4 super-resolution of a three-frame clip, `[T, 4H, 4W, 3]`, from mmediting's own
    `basicvsr_net.py`.

    IK_REF_SRC holds a `basicvsr/` directory with `basicvsr_net.py` and the real common files it draws
    on (`sr_backbone_utils.py`, `flow_warp.py`, `upsample.py`); only mmcv's `ConvModule` (a plain
    convolution + ReLU at SPyNet's settings) and the registry/logger shells are stubbed. The clip is
    three shifted crops of a larger plate, so SPyNet sees genuine translation; the record's `frames`
    carries all three `[T, H, W, 3]` for the Swift side.
    """
    import os
    import types
    from torch import nn

    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     norm_cfg=None, act_cfg=dict(type="ReLU")):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.activate = nn.ReLU() if act_cfg else None

        def forward(self, x):
            x = self.conv(x)
            return self.activate(x) if self.activate else x

    mmcv_cnn = types.ModuleType("mmcv.cnn")
    mmcv_cnn.ConvModule = ConvModule
    mmcv_cnn.constant_init = lambda *args, **kwargs: None
    mmcv_cnn.kaiming_init = lambda *args, **kwargs: None
    mmcv_runner = types.ModuleType("mmcv.runner")
    mmcv_runner.load_checkpoint = lambda *args, **kwargs: None
    parrots = types.ModuleType("mmcv.utils.parrots_wrapper")
    parrots._BatchNorm = nn.BatchNorm2d
    registry = types.ModuleType("mmedit.models.registry")
    registry.BACKBONES = type("Registry", (), {"register_module": staticmethod(lambda: (lambda cls: cls))})()
    utils_module = types.ModuleType("mmedit.utils")
    utils_module.get_root_logger = lambda *args, **kwargs: None
    sys.modules.update({"mmcv": types.ModuleType("mmcv"), "mmcv.cnn": mmcv_cnn,
                        "mmcv.runner": mmcv_runner, "mmcv.utils": types.ModuleType("mmcv.utils"),
                        "mmcv.utils.parrots_wrapper": parrots,
                        "mmedit": types.ModuleType("mmedit"), "mmedit.models": types.ModuleType("mmedit.models"),
                        "mmedit.models.registry": registry, "mmedit.utils": utils_module})

    source = os.path.join(_reference_source(), "basicvsr")
    upsample = _import_reference(source, "mmedit.models.common", "upsample",
                                 siblings=["sr_backbone_utils", "flow_warp"])
    common = sys.modules["mmedit.models.common"]
    common.PixelShufflePack = upsample.PixelShufflePack
    common.ResidualBlockNoBN = sys.modules["mmedit.models.common.sr_backbone_utils"].ResidualBlockNoBN
    common.make_layer = sys.modules["mmedit.models.common.sr_backbone_utils"].make_layer
    common.flow_warp = sys.modules["mmedit.models.common.flow_warp"].flow_warp

    module = _import_reference(source, "basicvsr_ref", "basicvsr_net")
    model = module.BasicVSRNet().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({name[len("generator."):]: value for name, value in state.items()
                           if name.startswith("generator.")}, strict=True)

    # Three 64×64 crops sliding across an 80×80 plate: real translation for SPyNet to find.
    plate = subject_image(80, 80)
    offsets = [(0, 0), (2, 3), (4, 6)]
    frames = np.stack([plate[dy:dy + 64, dx:dx + 64] for dy, dx in offsets])
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)     # [1, T, 3, H, W]
    with torch.no_grad():
        upscaled = model(clip)                                           # [1, T, 3, 4H, 4W]
    globals()["_extra"] = {"frames": torch.from_numpy(frames).contiguous()}
    return upscaled[0].permute(0, 2, 3, 1).contiguous()                  # [T, 4H, 4W, 3]


def run_zero_dce(image, checkpoint):
    """Zero-DCE low-light enhancement, `[H, W, 3]` in 0...1, from the reference `model.py`."""
    module = _import_reference(_reference_source(), "zero_dce_ref", "zerodce")
    model = module.enhance_net_nopool().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        enhanced = model(tensor)[1]
    return enhanced[0].permute(1, 2, 0).contiguous()


def run_style_transfer(image, checkpoint):
    """Fast style transfer, `[H, W, 3]`, from the reference `transformer_net.py`.

    Johnson's network is trained on a 0...255 scale, so the plate is scaled up on the way in and the
    result divided back down — matching what the port does internally.
    """
    module = _import_reference(_reference_source(), "style_ref", "transformer_net")
    model = module.TransformerNet()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # The released weights carry InstanceNorm running statistics that the current layer no longer keeps.
    for name in [k for k in state if k.endswith(("running_mean", "running_var"))]:
        del state[name]
    model.load_state_dict(state, strict=True)
    model.eval()

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0) * 255.0
    with torch.no_grad():
        stylized = model(tensor)
    return (stylized[0].permute(1, 2, 0) / 255.0).contiguous()


def run_realesrgan(image, checkpoint):
    """Real-ESRGAN ×4, `[4H, 4W, 3]` in 0...1, from BasicSR's `rrdbnet_arch.py`.

    That file reaches for BasicSR's registry and layer helpers; the registry decorator is a no-op here
    and the two helpers are the reference's own one-liners, so nothing beyond the architecture is
    supplied locally.
    """
    import types
    from torch import nn

    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = type("Registry", (), {"register": staticmethod(lambda: (lambda cls: cls))})()
    basicsr = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    sys.modules["basicsr"], sys.modules["basicsr.utils"] = basicsr, utils
    sys.modules["basicsr.utils.registry"] = registry

    arch = types.ModuleType("realesrgan_ref.arch_util")
    arch.default_init_weights = lambda *args, **kwargs: None
    arch.make_layer = lambda block, count, **kwargs: nn.Sequential(*[block(**kwargs) for _ in range(count)])

    def pixel_unshuffle(x, scale):
        b, c, h, w = x.size()
        view = x.view(b, c, h // scale, scale, w // scale, scale)
        return view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h // scale, w // scale)
    arch.pixel_unshuffle = pixel_unshuffle

    module = _import_reference(_reference_source(), "realesrgan_ref", "rrdbnet_arch",
                               injected={"arch_util": arch})
    # The anime release is a six-block generator; the general one is twenty-three.
    blocks = int(os.environ.get("IK_ESRGAN_BLOCKS", "23"))
    model = module.RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=blocks, num_grow_ch=32)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)
    model.eval()

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        upscaled = model(tensor)
    return upscaled[0].permute(1, 2, 0).contiguous()


def run_colorizer(image, checkpoint):
    """Colorization (Zhang et al. ECCV-16), `[H, W, 3]` sRGB in 0...1, from the reference `eccv16.py`.

    Feed a plate at the model's own 256×256 so the reference's input resize is an identity and the only
    resampling left in the comparison is the network's own ×4 upsample of the ab prediction.

    The record's `lightness` carries the L channel the reference used, so the Swift side can score the
    network alone as well as the whole chain.
    """
    import types
    from skimage import color

    # `eccv16.py` imports IPython's debugger at module scope and never calls it.
    ipython = types.ModuleType("IPython")
    ipython.embed = lambda *args, **kwargs: None
    sys.modules["IPython"] = ipython

    _import_reference(_reference_source(), "colorizer_ref", "base_color")
    module = _import_reference(_reference_source(), "colorizer_ref", "eccv16")
    model = module.ECCVGenerator().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    lab = color.rgb2lab(image).astype(np.float32)
    lightness = torch.from_numpy(lab[:, :, :1]).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        ab = model(lightness)                                   # [1, 2, H, W], unnormalized
    colorized = color.lab2rgb(np.concatenate([lab[:, :, :1], ab[0].permute(1, 2, 0).numpy()], axis=2))

    globals()["_extra"] = {"lightness": torch.from_numpy(lab[:, :, 0]).contiguous(),
                           "ab": ab[0].permute(1, 2, 0).contiguous()}
    return torch.from_numpy(colorized.astype(np.float32)).contiguous()


def run_deeplab(image, checkpoint):
    """torchvision DeepLabV3-ResNet50 class logits at the head's native stride-8 resolution.

    Comparing logits rather than the label map keeps the check sensitive: argmax hides everything except
    a difference big enough to flip a pixel's winner.
    """
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=21, aux_loss=True).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((torch.from_numpy(image).permute(2, 0, 1) - mean) / deviation).unsqueeze(0)
    with torch.no_grad():
        logits = model.classifier(model.backbone(tensor)["out"])
    return logits[0].permute(1, 2, 0).contiguous()              # [h/8, w/8, classes] to match NHWC


def run_vad(image, checkpoint):
    """NVIDIA NeMo frame-level MarbleNet speech probability per frame, `[frames]`.

    `--checkpoint` is the released `.nemo` archive. The record's `waveform` carries the `[samples]` clip
    the Swift side reads back; both sides run their own mel front end over it, so a preprocessing
    difference surfaces rather than hides.
    """
    import huggingface_hub

    # NeMo imports names the Hub dropped after its pin; they are only used by its model-search helpers,
    # not by the restore path, so a placeholder is enough to let the package import.
    for name in ["ModelFilter", "DatasetFilter"]:
        if not hasattr(huggingface_hub, name):
            setattr(huggingface_hub, name, type(name, (), {}))
    import nemo.collections.asr as nemo_asr

    # The installed NeMo builds a loss with a weight buffer the released archive predates; nothing on
    # the inference path reads it, so the restore is not strict about it.
    model = nemo_asr.models.EncDecFrameClassificationModel.restore_from(checkpoint, strict=False).eval()
    # Dither is training-time noise; zero it so the record is reproducible whichever NeMo gates on.
    model.preprocessor.featurizer.dither = 0.0

    samples = 16000
    time = np.arange(samples, dtype=np.float32) / 16000.0
    generator = np.random.default_rng(13)
    # Half a second of a voiced-like harmonic stack, then noise: a clip with a real boundary in it.
    speech = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 150 * (h + 1) * time) for h in range(6))
    wave = np.where(time < 0.5, speech, 0.05 * generator.standard_normal(samples)).astype(np.float32)

    with torch.no_grad():
        features, lengths = model.preprocessor(input_signal=torch.from_numpy(wave)[None],
                                               length=torch.tensor([samples]))
        logits = model(input_signal=torch.from_numpy(wave)[None],
                       input_signal_length=torch.tensor([samples]))
    # The mel spectrogram is the seam between the front end and the encoder: agreeing here and
    # disagreeing at the output isolates the network, and vice versa.
    globals()["_extra"] = {"prompt": torch.tensor(prompt_ids, dtype=torch.int32).contiguous(),
                           "waveform": torch.from_numpy(wave).contiguous(),
                           "features": features[0].transpose(0, 1).contiguous()}   # [frames, mels]
    return torch.softmax(logits, dim=-1)[0, :, 1].contiguous()  # [frames]


def run_silero_vad(image):
    """Silero VAD v6 (snakers4, PyPI `silero_vad` 6.2.1) per-chunk speech probability, `[chunks]`.

    The clip is exactly 32 chunks of 512 samples (1.024 s at 16 kHz) with a boundary at 0.5 s, so no
    padding enters and both sides produce the same chunk count. The record's `waveform` is the clip the
    Swift side reads back; both sides stream 512-sample chunks with a 64-sample look-back and thread the
    LSTM state, so a preprocessing or state-threading difference surfaces rather than hides.

    The reference bundles the model, so this mode needs no `--checkpoint`. Requires: `silero_vad`,
    `torchaudio` (pip install silero-vad torchaudio).
    """
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    model.reset_states()
    rate, chunk, chunks = 16000, 512, 32
    samples = chunks * chunk
    time = np.arange(samples, dtype=np.float32) / rate
    generator = np.random.default_rng(13)
    speech = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 150 * (h + 1) * time) for h in range(6))
    wave = np.where(time < 0.5, speech, 0.05 * generator.standard_normal(samples)).astype(np.float32)

    probabilities = []
    with torch.no_grad():
        for index in range(chunks):
            block = torch.from_numpy(wave[index * chunk:(index + 1) * chunk])[None]   # [1, 512]
            probabilities.append(float(np.asarray(model(block, rate)).reshape(-1)[0]))
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous()}
    return torch.tensor(probabilities, dtype=torch.float32)                            # [chunks]


def run_dac(image):
    """Descript Audio Codec (44 kHz) round trip: the codebook tokens and the waveform reconstructed from
    them. The record's `waveform` is a clip an exact multiple of the hop long (so no padding enters and
    the frame counts agree); `codes` is `[n_codebooks, frames]` and `reconstruction` is the decode of
    those codes. Comparing codes exercises the encoder + RVQ; decoding the recorded codes isolates the
    decoder from a single code flipping on a codebook near-tie.

    Needs no `--checkpoint`; the `dac` package downloads the released weights. Requires: descript-audio-codec.
    """
    import dac

    model = dac.DAC.load(dac.utils.download(model_type="44khz")).eval()
    hop = model.hop_length
    frames = 87
    samples = frames * hop
    time = np.arange(samples, dtype=np.float32) / model.sample_rate
    generator = np.random.default_rng(19)
    wave = (0.3 * np.sin(2 * np.pi * 220 * time) + 0.2 * np.sin(2 * np.pi * 440 * time)
            + 0.1 * np.sin(2 * np.pi * 880 * time) + 0.02 * generator.standard_normal(samples)).astype(np.float32)
    wave = np.clip(wave, -1, 1)

    with torch.no_grad():
        audio = model.preprocess(torch.from_numpy(wave)[None, None], model.sample_rate)
        _, codes, _, _, _ = model.encode(audio)
        reconstruction = model.decode(model.quantizer.from_codes(codes)[0])

    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous(),
                           "codes": codes[0].to(torch.int32).contiguous()}          # [n_codebooks, frames]
    return reconstruction.reshape(-1)[:samples].contiguous()                         # [samples]


def run_snac(image):
    """SNAC (24 kHz multi-scale codec) round trip: the per-codebook tokens (`codes0`, `codes1`, … at each
    codebook's own temporal rate) and the waveform reconstructed from them. The decoder's noise block is
    DISABLED so the reconstruction is deterministic; its expected contribution is zero, and the Swift side
    decodes with the same noise-off path.

    Needs no `--checkpoint`; `SNAC.from_pretrained` fetches the released weights. Requires: snac.
    """
    import snac.layers as layers
    from snac import SNAC

    layers.NoiseBlock.forward = lambda self, x: x                      # deterministic decode
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    hop = int(np.prod(model.encoder_rates)) if hasattr(model, "encoder_rates") else 512
    frames = 24
    samples = frames * hop
    time = np.arange(samples, dtype=np.float32) / model.sampling_rate
    generator = np.random.default_rng(23)
    wave = (0.3 * np.sin(2 * np.pi * 180 * time) + 0.2 * np.sin(2 * np.pi * 360 * time)
            + 0.02 * generator.standard_normal(samples)).astype(np.float32)
    wave = np.clip(wave, -1, 1)

    with torch.no_grad():
        codes = model.encode(torch.from_numpy(wave)[None, None])
        reconstruction = model.decode(codes)

    extra = {"waveform": torch.from_numpy(wave).contiguous()}
    for index, stream in enumerate(codes):
        extra[f"codes{index}"] = stream[0].to(torch.int32).contiguous()
    globals()["_extra"] = extra
    return reconstruction.reshape(-1)[:samples].contiguous()


def run_siglip2(image):
    """SigLIP 2 (base-patch16-224) image-text model: the L2-normalized image embedding, plus the text
    embeddings, token ids, and sigmoid logits for a few captions. The image pixel values are the plate
    normalized to -1…1 (SigLIP's normalization), fed directly so the processor's resize is out of the
    comparison; the plate must be 224×224 (`--size 224`).

    Requires: transformers >= 4.51 (SigLIP 2), which the llm oracle env carries.
    """
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    pixel_values = torch.from_numpy((image * 2 - 1).transpose(2, 0, 1))[None].float()
    texts = ["a photo of a cat", "a photo of two cats", "a city street at night"]
    tokens = processor(text=texts, return_tensors="pt", padding="max_length", max_length=64).input_ids

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixel_values)
        text_features = model.get_text_features(input_ids=tokens)
        # Isolation seams: the embeddings output and the post-layernorm hidden (pre-head).
        vision = model.vision_model
        patch_embeds = vision.embeddings(pixel_values)
        vision_last = vision(pixel_values=pixel_values).last_hidden_state
    image_embeds = image_features / image_features.norm(dim=-1, keepdim=True)
    text_embeds = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = text_embeds @ image_embeds.t() * model.logit_scale.exp() + model.logit_bias

    globals()["_extra"] = {"text_embeds": text_embeds.contiguous(),
                           "tokens": tokens.to(torch.int32).contiguous(),
                           "logits": logits.reshape(-1).contiguous(),
                           "patch_embeds": patch_embeds[0].contiguous(),
                           "vision_last": vision_last[0].contiguous()}
    return image_embeds[0].contiguous()                                    # [embed], L2-normalized


def run_taesd(image):
    """TAESD (tiny SD autoencoder) round trip: the encoder's latent and the decoder's reconstruction.
    Feed a plate whose side is a multiple of 8 (`--size 256`), so the latent is `side/8` square. The
    reference is madebyollin's own `taesd.py` (vendored under the source root) loaded with the released
    `.pth` weights. Requires: huggingface_hub, and `taesd.py` in the source root.
    """
    import os
    import sys
    import tempfile
    import urllib.request

    sys.path.insert(0, os.environ.get("IK_REF_SRC", os.path.expanduser("~/.inferkit-validation/sources")))
    from taesd import TAESD

    def weights(name):
        # The .pth live in the GitHub repo, not the HF repo (which carries the diffusers safetensors).
        path = os.path.join(tempfile.gettempdir(), name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"https://github.com/madebyollin/taesd/raw/main/{name}", path)
        return path

    model = TAESD(weights("taesd_encoder.pth"), weights("taesd_decoder.pth")).eval()
    with torch.no_grad():
        latent = model.encoder(torch.from_numpy(image.transpose(2, 0, 1))[None].float())
        reconstruction = model.decoder(latent)
    globals()["_extra"] = {"latent": latent[0].permute(1, 2, 0).contiguous()}       # [h, w, 4] NHWC
    return reconstruction[0].permute(1, 2, 0).contiguous()                          # [H, W, 3]


def run_ltx_vae(image):
    """LTX-Video VAE round trip: the deterministic latent (posterior mean) and the decoded video, plus
    the encoder seams (conv_in, first down block, mid block). Uses a fixed random video, so the plate is
    ignored. The VAE is loaded from `IK_LTX_VAE_DIR` (default `~/.inferkit-validation/raw/ltx-vae`).

    Runs under the `ltx` oracle env (diffusers >= 0.32). Tensors are returned in NDHWC to match the port.
    """
    import os
    from diffusers import AutoencoderKLLTXVideo

    directory = os.environ.get("IK_LTX_VAE_DIR", os.path.expanduser("~/.inferkit-validation/raw/ltx-vae"))
    vae = AutoencoderKLLTXVideo.from_pretrained(directory, torch_dtype=torch.float32).eval()
    generator = np.random.default_rng(11)
    video = (generator.random((1, 3, 9, 64, 64), dtype=np.float32) * 2 - 1)

    seams = {}
    vae.encoder.conv_in.register_forward_hook(lambda m, i, o: seams.__setitem__("enc_conv_in", o.detach()))
    vae.encoder.down_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("enc_down0", o.detach()))
    vae.encoder.mid_block.register_forward_hook(lambda m, i, o: seams.__setitem__("enc_mid", o.detach()))
    with torch.no_grad():
        mean = vae.encode(torch.from_numpy(video)).latent_dist.mean
        decoded = vae.decode(mean).sample

    def ndhwc(t):
        return t[0].permute(1, 2, 3, 0).contiguous()

    globals()["_extra"] = {"input_video": ndhwc(torch.from_numpy(video)), "latent": ndhwc(mean),
                           "enc_conv_in": ndhwc(seams["enc_conv_in"]), "enc_down0": ndhwc(seams["enc_down0"]),
                           "enc_mid": ndhwc(seams["enc_mid"])}
    return ndhwc(decoded)                                                   # [T, H, W, 3]


def run_ltx_transformer(image):
    """LTX-Video DiT velocity prediction from random latent tokens, a random text embedding, and a
    timestep, plus the rope / proj_in / first-block seams. The text embedding is fed directly, so the DiT
    is verified in isolation (no T5). The model loads from `IK_LTX_TF_DIR`
    (default `~/.inferkit-validation/raw/ltx-transformer`). Runs under the `ltx` oracle env.
    """
    import os
    from diffusers import LTXVideoTransformer3DModel

    directory = os.environ.get("IK_LTX_TF_DIR", os.path.expanduser("~/.inferkit-validation/raw/ltx-transformer"))
    model = LTXVideoTransformer3DModel.from_pretrained(directory, torch_dtype=torch.float32).eval()
    generator = torch.Generator().manual_seed(7)
    frames, height, width = 2, 2, 2
    latent = torch.randn(1, frames * height * width, 128, generator=generator)
    text = torch.randn(1, 4, 4096, generator=generator)
    timestep = torch.tensor([500.0])
    scale = (1.0, 1.0, 1.0)

    seams = {}
    model.proj_in.register_forward_hook(lambda m, i, o: seams.__setitem__("proj_in", o.detach()))
    model.transformer_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o.detach()))
    original_rope = model.rope.forward

    def rope_hook(*a, **k):
        out = original_rope(*a, **k)
        seams["rope_cos"], seams["rope_sin"] = out[0].detach(), out[1].detach()
        return out

    model.rope.forward = rope_hook
    with torch.no_grad():
        output = model(hidden_states=latent, encoder_hidden_states=text, timestep=timestep,
                       encoder_attention_mask=torch.ones(1, 4), num_frames=frames, height=height, width=width,
                       rope_interpolation_scale=scale, return_dict=False)[0]

    globals()["_extra"] = {"latent": latent[0].contiguous(), "text": text[0].contiguous(),
                           "timestep": timestep.contiguous(), "proj_in": seams["proj_in"][0].contiguous(),
                           "block0": seams["block0"][0].contiguous(), "rope_cos": seams["rope_cos"][0].contiguous(),
                           "rope_sin": seams["rope_sin"][0].contiguous(),
                           "grid": torch.tensor([frames, height, width], dtype=torch.int32)}
    return output[0].contiguous()                                          # [S, 128]


def run_ltx_t5(image):
    """LTX-Video T5-XXL text encoder output for a fixed padded token sequence, plus the embedding and
    first-block seams. Loads from `IK_LTX_T5_DIR` (default `~/.inferkit-validation/raw/ltx-t5`). Runs
    under the `ltx` oracle env (transformers with T5EncoderModel).
    """
    import os
    from transformers import T5EncoderModel

    directory = os.environ.get("IK_LTX_T5_DIR", os.path.expanduser("~/.inferkit-validation/raw/ltx-t5"))
    model = T5EncoderModel.from_pretrained(directory, torch_dtype=torch.float32).eval()
    ids = torch.tensor([[3, 19, 2523, 40, 8, 1946, 55, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

    seams = {}
    model.shared.register_forward_hook(lambda m, i, o: seams.__setitem__("embed", o.detach()))
    model.encoder.block[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o[0].detach()))
    with torch.no_grad():
        output = model(input_ids=ids).last_hidden_state

    globals()["_extra"] = {"tokens": ids.to(torch.int32).contiguous(), "embed": seams["embed"][0].contiguous(),
                           "block0": seams["block0"][0].contiguous()}
    return output[0].contiguous()                                          # [S, 4096]


def run_z_image(image):
    """The Z-Image S3-DiT velocity prediction at a tiny random configuration, from diffusers' own
    ZImageTransformer2DModel, plus the timestep, noise-refiner, and first unified-layer seams.

    Single-stream: the image latent and the caption features are concatenated and every layer's
    self-attention runs over the join. The caption features are supplied directly (no Qwen3), so the
    DiT is verified in isolation, as the LTX DiT is. Sequence lengths are deliberately NOT a multiple
    of 32, so the learned pad tokens (`x_pad_token`, `cap_pad_token`) and the (0,0,0) pad positions are
    exercised. Runs under the `ltx` oracle env (diffusers with ZImageTransformer2DModel). `image` unused.
    """
    from diffusers import ZImageTransformer2DModel

    model = ZImageTransformer2DModel(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=32, n_layers=2,
        n_refiner_layers=1, n_heads=2, n_kv_heads=2, norm_eps=1e-5, qk_norm=True, cap_feat_dim=24,
        rope_theta=256.0, t_scale=1000.0, axes_dims=[4, 6, 6], axes_lens=[1024, 512, 512])
    model = _randomized(model, seed=19)

    generator = torch.Generator().manual_seed(5)
    latent = torch.randn(4, 1, 8, 8, generator=generator)                  # [C, F, H, W] -> 16 tokens
    cap = torch.randn(20, 24, generator=generator)                         # 20 caption tokens
    t = torch.tensor([0.3])

    seams = {}
    model.t_embedder.register_forward_hook(lambda m, i, o: seams.__setitem__("t_emb", o.detach()))
    model.noise_refiner[-1].register_forward_hook(lambda m, i, o: seams.__setitem__("noise_out", o.detach()))
    model.layers[0].register_forward_hook(lambda m, i, o: seams.__setitem__("layer0", o.detach()))
    with torch.no_grad():
        output = model([latent], t, [cap], patch_size=2, f_patch_size=1, return_dict=False)[0][0]

    extra = {"latent": latent.contiguous(), "cap_feats": cap.contiguous(), "timestep": t.contiguous(),
             "t_emb": seams["t_emb"][0].contiguous(), "noise_out": seams["noise_out"][0].contiguous(),
             "layer0": seams["layer0"][0].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [C, F, H, W]


def run_sana(image):
    """The SANA linear-attention DiT velocity at a tiny random configuration, from diffusers'
    SanaTransformer2DModel, plus the patch-embed, first-block, and embedded-timestep seams.

    SANA replaces softmax self-attention with ReLU LINEAR attention and the MLP with a GLUMBConv
    (gated depthwise convolution over the token grid); each block cross-attends to the caption. The
    caption embedding is supplied directly (no Gemma), so the DiT is verified in isolation. Runs under
    the `ltx` oracle env (diffusers with SanaTransformer2DModel). `image` unused.
    """
    from diffusers import SanaTransformer2DModel

    model = SanaTransformer2DModel(
        in_channels=8, out_channels=8, num_attention_heads=2, attention_head_dim=8, num_layers=2,
        num_cross_attention_heads=2, cross_attention_head_dim=8, cross_attention_dim=16,
        caption_channels=12, mlp_ratio=2.0, patch_size=1, attention_bias=False,
        norm_elementwise_affine=False, norm_eps=1e-6, qk_norm=None, guidance_embeds=False,
        timestep_scale=1.0, sample_size=8)
    model = _randomized(model, seed=23)

    generator = torch.Generator().manual_seed(6)
    latent = torch.randn(1, 8, 4, 4, generator=generator)                  # [B, C, H, W]
    cap = torch.randn(1, 6, 12, generator=generator)                       # [B, Lc, caption_channels]
    t = torch.tensor([0.4])

    seams = {}
    model.patch_embed.register_forward_hook(lambda m, i, o: seams.__setitem__("patch", o.detach()))
    model.time_embed.register_forward_hook(lambda m, i, o: seams.__setitem__("embedded", o[1].detach()))
    model.transformer_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o.detach()))
    with torch.no_grad():
        output = model(hidden_states=latent, encoder_hidden_states=cap, timestep=t, return_dict=False)[0][0]

    extra = {"latent": latent[0].contiguous(), "cap_feats": cap[0].contiguous(), "timestep": t.contiguous(),
             "patch": seams["patch"][0].contiguous(), "embedded": seams["embedded"][0].contiguous(),
             "block0": seams["block0"][0].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [C, H, W]


def run_wan(image):
    """The Wan text-to-video DiT velocity at a tiny random configuration, from diffusers'
    WanTransformer3DModel, plus the patch-embed and first-block seams.

    A 3-D sequence transformer over a Conv3d-patchified video latent, with a 3-axis interleaved rotary
    over the (frame, height, width) grid, self + cross attention, and an across-heads RMS q/k norm. The
    text embedding is supplied directly (no umT5), so the DiT is verified in isolation. Runs under the
    `ltx` oracle env (diffusers with WanTransformer3DModel). `image` unused.
    """
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=4,
        out_channels=4, text_dim=10, freq_dim=256, ffn_dim=48, num_layers=2, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None,
        rope_max_seq_len=1024)
    model = _randomized(model, seed=29)

    generator = torch.Generator().manual_seed(8)
    latent = torch.randn(1, 4, 2, 4, 4, generator=generator)               # [B, C, T, H, W]
    text = torch.randn(1, 6, 10, generator=generator)                      # [B, Lc, text_dim]
    t = torch.tensor([0.35])

    seams = {}
    model.patch_embedding.register_forward_hook(lambda m, i, o: seams.__setitem__("patch", o.detach()))
    model.blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o.detach()))
    with torch.no_grad():
        output = model(hidden_states=latent, timestep=t, encoder_hidden_states=text, return_dict=False)[0][0]

    extra = {"latent": latent[0].contiguous(), "text": text[0].contiguous(), "timestep": t.contiguous(),
             "patch": seams["patch"][0].flatten(1).transpose(0, 1).contiguous(),
             "block0": seams["block0"][0].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [C, F, H, W]


def run_umt5(image):
    """The umT5 text encoder (`UMT5EncoderModel`, Wan's text encoder) at a tiny random configuration,
    from transformers. umT5 differs from plain T5 in giving EVERY layer its own relative-position bias;
    everything else (T5LayerNorm, the gated FFN, the unscaled attention) is shared with T5. Records the
    last hidden state and the weights. Runs under the `llm` oracle env. `image` unused.
    """
    from transformers import UMT5Config, UMT5EncoderModel

    config = UMT5Config(
        d_model=32, num_layers=3, num_heads=2, d_kv=16, d_ff=64, vocab_size=128,
        relative_attention_num_buckets=16, relative_attention_max_distance=32, dropout_rate=0.0)
    model = _randomized(UMT5EncoderModel(config), seed=47)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=tokens).last_hidden_state[0]

    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for key, value in model.state_dict().items():
        if key == "encoder.embed_tokens.weight":                          # tied to shared.weight
            continue
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [N, d_model]


def run_gemma2(image):
    """The Gemma-2 text decoder (`Gemma2Model`, SANA's text encoder) at a tiny random configuration,
    from transformers. Records the last hidden state and the weights. Exercises the sandwich norms, the
    (1+w) RMS, the attention logit soft-cap, GQA, and the alternating sliding-window / full attention
    (a small window makes the sliding layers differ from the full ones). Runs under the `llm` oracle env
    (transformers with Gemma2Model). `image` unused.
    """
    from transformers import Gemma2Config, Gemma2Model

    config = Gemma2Config(
        hidden_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        intermediate_size=64, vocab_size=128, query_pre_attn_scalar=8, attn_logit_softcapping=50.0,
        sliding_window=3, rope_theta=10000.0, hidden_activation="gelu_pytorch_tanh", max_position_embeddings=64)
    model = _randomized(Gemma2Model(config), seed=43)
    print("layer_types:", config.layer_types)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5, 88, 30]], dtype=torch.long)
    with torch.no_grad():
        output = model(tokens).last_hidden_state[0]

    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [N, hidden]


def run_dpm_solver(image):
    """SANA's released sampler, `DPMSolverMultistepScheduler` in its flow-prediction configuration, run
    over a FIXED sequence of velocities so the scheduler math is verified with no model. Records the
    sigma schedule, the timesteps, and the sample trajectory. Runs under the `ltx` oracle env.
    """
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, algorithm_type="dpmsolver++", solver_order=2,
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0,
        final_sigmas_type="zero", solver_type="midpoint", lower_order_final=True)
    steps = 8
    scheduler.set_timesteps(steps)

    generator = torch.Generator().manual_seed(3)
    initial = torch.randn(4, 4, 4, generator=generator)
    velocities = [torch.randn(4, 4, 4, generator=generator) for _ in range(steps)]

    sample = initial.clone()
    trajectory = []
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.step(velocities[i], t, sample, return_dict=False)[0]
        trajectory.append(sample.clone())

    extra = {"sigmas": scheduler.sigmas.float().contiguous(), "timesteps": scheduler.timesteps.float().contiguous(),
             "initial": initial.contiguous(), "velocities": torch.stack(velocities).contiguous(),
             "trajectory": torch.stack(trajectory).contiguous()}
    globals()["_extra"] = extra
    return trajectory[-1].contiguous()


def run_unipc(image):
    """Wan's released sampler, `UniPCMultistepScheduler` in its flow-prediction configuration, run over a
    FIXED sequence of velocities so the predictor-corrector math is verified with no model. Records the
    sigma schedule, timesteps, and the sample trajectory. Runs under the `ltx` oracle env.
    """
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, solver_order=2, solver_type="bh2", predict_x0=True,
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0, lower_order_final=True)
    steps = 8
    scheduler.set_timesteps(steps)

    generator = torch.Generator().manual_seed(4)
    initial = torch.randn(4, 4, 4, generator=generator)
    velocities = [torch.randn(4, 4, 4, generator=generator) for _ in range(steps)]

    sample = initial.clone()
    trajectory = []
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.step(velocities[i], t, sample, return_dict=False)[0]
        trajectory.append(sample.clone())

    extra = {"sigmas": scheduler.sigmas.float().contiguous(), "timesteps": scheduler.timesteps.float().contiguous(),
             "initial": initial.contiguous(), "velocities": torch.stack(velocities).contiguous(),
             "trajectory": torch.stack(trajectory).contiguous()}
    globals()["_extra"] = extra
    return trajectory[-1].contiguous()


def run_wan_vae(image):
    """The Wan 3D causal VAE (`AutoencoderKLWan`, Wan 2.2 residual path) at a tiny random configuration,
    from diffusers' own class. Exercises the stateful feat_cache streaming loop — the encoder consumes
    frames in chunks (1 then 4) and the decoder emits one latent frame at a time, threading the causal
    convolutions' temporal cache — plus the residual AvgDown3D / DupUp3D shortcuts, the temporal
    resample time_convs, and the patchify (2×). Records the encoded latent mean and the decode. Runs
    under the `ltx` oracle env. `image` unused.
    """
    from diffusers import AutoencoderKLWan

    model = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[2, 2], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[True], is_residual=True, patch_size=2, in_channels=12, out_channels=12)
    model = _randomized(model, seed=41)

    generator = torch.Generator().manual_seed(13)
    video = torch.randn(1, 3, 5, 16, 16, generator=generator)              # [B, C, T, H, W], 5 frames
    seams = {}
    model.decoder.conv_in.register_forward_hook(lambda m, i, o: seams.__setitem__("dec_conv_in", o.detach()))
    model.decoder.mid_block.register_forward_hook(lambda m, i, o: seams.__setitem__("dec_mid", o.detach()))
    model.decoder.up_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("dec_up0", o.detach()))
    with torch.no_grad():
        latent = model.encode(video).latent_dist.mean
        enc_raw = model._encode(video)                                     # pre-quant-split moments
        enc_1frame = model._encode(video[:, :, 0:1])                       # single-chunk encode
        decoded = model.decode(latent).sample
        decoded_f0 = model.decode(latent[:, :, 0:1]).sample                # first latent frame alone (hooks fire here)

    def ndhwc(t):
        return t[0].permute(1, 2, 3, 0).contiguous()                       # [C,T,H,W] -> [T,H,W,C]

    extra = {"video": ndhwc(video), "latent": ndhwc(latent),
             "decoded_f0": ndhwc(decoded_f0), "enc_raw": ndhwc(enc_raw), "enc_1frame": ndhwc(enc_1frame),
             "dec_conv_in": ndhwc(seams["dec_conv_in"]), "dec_mid": ndhwc(seams["dec_mid"]),
             "dec_up0": ndhwc(seams["dec_up0"])}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return ndhwc(decoded)                                                  # [T, H, W, 3]


def run_wan_vae_21(image):
    """The Wan 2.1 autoencoder (`AutoencoderKLWan`, the NON-residual path) at a tiny random configuration.
    Wan 2.1 differs from 2.2 in a flat down-block list (no AvgDown3D / DupUp3D shortcuts), a halving
    upsampler, no patchify, and 16 latent channels. Records the encoded latent mean and the decode. Runs
    under the `ltx` oracle env. `image` unused.
    """
    from diffusers import AutoencoderKLWan

    model = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[True], is_residual=False, patch_size=None, in_channels=3, out_channels=3)
    model = _randomized(model, seed=53)

    generator = torch.Generator().manual_seed(17)
    video = torch.randn(1, 3, 5, 16, 16, generator=generator)
    with torch.no_grad():
        latent = model.encode(video).latent_dist.mean
        decoded = model.decode(latent).sample

    def ndhwc(t):
        return t[0].permute(1, 2, 3, 0).contiguous()

    extra = {"video": ndhwc(video), "latent": ndhwc(latent)}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return ndhwc(decoded)


def run_dc_ae(image):
    """The Deep-Compression Autoencoder (SANA's VAE) at a tiny random configuration, from diffusers'
    own AutoencoderDC. Deterministic (encode returns one latent). Exercises a ResBlock stage, an
    EfficientViTBlock stage (multiscale ReLU linear attention + GLUMBConv), the Conv downsample /
    interpolate upsample with their channel-average / channel-repeat shortcuts, and the in/out
    shortcuts. Records the latent and the decode. Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers import AutoencoderDC

    model = AutoencoderDC(
        in_channels=3, latent_channels=8, attention_head_dim=4,
        encoder_block_types=["ResBlock", "EfficientViTBlock"],
        decoder_block_types=["ResBlock", "EfficientViTBlock"],
        encoder_block_out_channels=[16, 32], decoder_block_out_channels=[16, 32],
        encoder_layers_per_block=[1, 1], decoder_layers_per_block=[1, 1],
        encoder_qkv_multiscales=[(), (3,)], decoder_qkv_multiscales=[(), (3,)],
        downsample_block_type="Conv", upsample_block_type="interpolate")
    model = _randomized(model, seed=37)

    generator = torch.Generator().manual_seed(11)
    sample = torch.randn(1, 3, 8, 8, generator=generator)
    with torch.no_grad():
        latent = model.encode(sample).latent
        decoded = model.decode(latent).sample

    extra = {"sample": sample[0].permute(1, 2, 0).contiguous(),           # [H, W, 3] NHWC
             "latent": latent[0].permute(1, 2, 0).contiguous()}            # [h, w, latent] NHWC
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return decoded[0].permute(1, 2, 0).contiguous()                        # [H, W, 3] NHWC


def run_ip_adapter(image):
    """IP-Adapter's two pieces at a tiny random configuration, from diffusers: the ImageProjection (CLIP
    image embedding → image-text tokens) and the decoupled cross-attention (`IPAdapterAttnProcessor2_0`:
    the text cross-attention plus a scaled image-conditioned attention sharing the query). Records both
    seams. Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers.models.embeddings import ImageProjection
    from diffusers.models.attention_processor import Attention, IPAdapterAttnProcessor2_0

    torch.manual_seed(59)
    # 1. Image projection.
    proj = ImageProjection(image_embed_dim=16, cross_attention_dim=12, num_image_text_embeds=4)
    proj = _randomized(proj, seed=59)
    image_embeds = torch.randn(1, 16)
    with torch.no_grad():
        ip_tokens = proj(image_embeds)                                     # [1, 4, 12]

    # 2. Decoupled cross-attention.
    attn = Attention(query_dim=12, cross_attention_dim=12, heads=2, dim_head=6)
    processor = IPAdapterAttnProcessor2_0(hidden_size=12, cross_attention_dim=12, num_tokens=[4], scale=0.7)
    attn.set_processor(processor)
    attn = _randomized(attn, seed=61)
    hidden = torch.randn(1, 8, 12)
    text = torch.randn(1, 5, 12)
    with torch.no_grad():
        output = attn(hidden, encoder_hidden_states=(text, [ip_tokens]))   # [1, 8, 12]

    extra = {"image_embeds": image_embeds[0].contiguous(), "ip_tokens": ip_tokens[0].contiguous(),
             "hidden": hidden[0].contiguous(), "text": text[0].contiguous()}
    for key, value in proj.state_dict().items():
        extra[f"wp::{key}"] = value.float().contiguous()
    for key, value in attn.state_dict().items():
        extra[f"wa::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output[0].contiguous()


def run_dc_ae_real(image):
    """The Deep-Compression Autoencoder on the RELEASED SANA weights, from diffusers' AutoencoderDC. A
    real-weights end-to-end validation: it loads the actual `Sana_600M` VAE, encodes the plate, and
    decodes, so the released config, the sharded/real checkpoint, and the loader are exercised — not just
    the architecture at a tiny config. Loads from `IK_VAL_DCAE` (default `~/.inferkit-validation/raw/
    sana-dcae`). Runs under the `ltx` oracle env.
    """
    import os
    from diffusers import AutoencoderDC

    directory = os.environ.get("IK_VAL_DCAE", os.path.expanduser("~/.inferkit-validation/raw/sana-dcae"))
    model = AutoencoderDC.from_pretrained(directory, torch_dtype=torch.float32).eval()
    sample = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) * 2 - 1   # [1, 3, H, W] in -1..1
    with torch.no_grad():
        latent = model.encode(sample).latent
        decoded = model.decode(latent).sample

    globals()["_extra"] = {"latent": latent[0].permute(1, 2, 0).contiguous()} # [h, w, 32] NHWC
    return decoded[0].permute(1, 2, 0).contiguous()                          # [H, W, 3] NHWC (input_image is the plate)


def run_flux_vae(image):
    """The Flux autoencoder (the one Z-Image encodes into) at a tiny random configuration, from
    diffusers' own AutoencoderKL. It is the same class Stable Diffusion uses, differing only in the
    16 latent channels and in dropping the quant convolutions (`use_quant_conv=False`), which is the
    path this record exercises. Records the encoded mean latent and the decode. Runs under the `ltx`
    oracle env. `image` unused.
    """
    from diffusers import AutoencoderKL

    model = AutoencoderKL(
        in_channels=3, out_channels=3, block_out_channels=[8, 16], layers_per_block=1,
        latent_channels=4, norm_num_groups=4, use_quant_conv=False, use_post_quant_conv=False,
        mid_block_add_attention=True,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"])
    model = _randomized(model, seed=31)

    generator = torch.Generator().manual_seed(9)
    sample = torch.randn(1, 3, 16, 16, generator=generator)
    with torch.no_grad():
        mean = model.encode(sample).latent_dist.mean
        decoded = model.decode(mean).sample

    extra = {"sample": sample[0].permute(1, 2, 0).contiguous(),           # [H, W, 3] NHWC
             "mean": mean[0].permute(1, 2, 0).contiguous()}                # [h, w, latent] NHWC
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return decoded[0].permute(1, 2, 0).contiguous()                        # [H, W, 3] NHWC


def run_sam_encoder(image, checkpoint):
    """The official SAM image encoder's neck output — the seam between the ViT and the mask decoder.

    Feed a 1024×1024 plate so `ResizeLongestSide` and the square padding are both identities; then the
    only work on either side is normalization + encoder, and a mismatch localizes to the ViT itself.
    """
    from segment_anything import sam_model_registry, SamPredictor

    predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=checkpoint).eval())
    predictor.set_image((image * 255).astype(np.uint8))
    features = predictor.features[0]                            # [256, 64, 64]
    return features.permute(1, 2, 0).contiguous()               # [64, 64, 256], matching MLX's NHWC


def run_sam(image, checkpoint):
    """The official SAM mask from a single positive point at (w/2, h/3), as probabilities in 0...1."""
    from segment_anything import sam_model_registry, SamPredictor

    predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=checkpoint).eval())
    predictor.set_image((image * 255).astype(np.uint8))
    height, width = image.shape[:2]
    point = np.array([[width / 2.0, height / 3.0]])
    masks, scores, _ = predictor.predict(point_coords=point, point_labels=np.array([1]),
                                         multimask_output=True, return_logits=True)
    best = int(np.argmax(scores))
    return torch.sigmoid(torch.from_numpy(masks[best]))         # [H, W] probabilities


def run_sam_decoder(image, checkpoint):
    """All four raw mask-token logits and IoU predictions from the official mask decoder.

    Comparing the selected mask conflates three failure modes: token permutation, logit inversion, and a
    genuine decoder divergence. The full `[4, 256, 256]` token set (`output`) plus `scores` `[4]`
    disentangles them — the Swift side prints the pairwise cosine matrix.
    """
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint).eval()
    predictor = SamPredictor(sam)
    predictor.set_image((image * 255).astype(np.uint8))
    height, width = image.shape[:2]
    coords = torch.tensor([[[width / 2.0 + 0.5, height / 3.0 + 0.5]]], dtype=torch.float32)
    labels = torch.tensor([[1]], dtype=torch.int64)
    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        masks, scores = sam.mask_decoder.predict_masks(
            image_embeddings=predictor.features, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
    globals()["_extra"] = {"scores": scores[0].contiguous()}
    return masks[0].contiguous()                                # [4, 256, 256] logits


def run_zero_dce_losses(image):
    """The four zero-reference losses Zero-DCE trains against.

    Transcribed from the reference implementation (Li-Chongyi/Zero-DCE, `Zero-DCE_code/Myloss.py`,
    Guo et al. CVPR 2020) with the `.cuda()` calls removed and the training script's settings applied
    (`L_exp(16, 0.6)`). Several lines look like slips and are reproduced exactly anyway: the released
    weights were trained with these, so matching them is what makes a fine-tune behave like the
    published method. Each is marked below.

    This record isolates the loss arithmetic, so the enhanced image and the curve maps are synthesized
    here and written into the record rather than produced by a network. Both sides then score the SAME
    tensors, and any difference can only come from the losses themselves.
    """
    import torch.nn.functional as F

    def l_exp(x, patch_size=16, mean_val=0.6):
        x = torch.mean(x, 1, keepdim=True)
        mean = F.avg_pool2d(x, patch_size)
        # Squared deviation, not absolute.
        return torch.mean(torch.pow(mean - mean_val, 2))

    def l_color(x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        # The differences are squared, then squared AGAIN under the square root: the result is the
        # 4th powers, not the squares. Reproduced as written.
        k = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        return torch.mean(k)

    def l_tv(x):
        batch_size, _, h_x, w_x = x.size()
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        # The sums run over channels too, but the counts do not, so the result carries a factor of
        # the channel count on top of the leading 2.
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : w_x - 1], 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def l_spa(org, enhance):
        kernels = [
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],                 # left
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],                 # right
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],                 # up
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],                 # down
        ]
        pool = torch.nn.AvgPool2d(4)
        org_pool = pool(torch.mean(org, 1, keepdim=True))
        enhance_pool = pool(torch.mean(enhance, 1, keepdim=True))
        # padding=1 means the border regions difference against zero, so the four directions are not
        # redundant there even though squaring makes the interior pairs symmetric.
        total = 0
        for kernel in kernels:
            weight = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
            d_org = F.conv2d(org_pool, weight, padding=1)
            d_enhance = F.conv2d(enhance_pool, weight, padding=1)
            total = total + torch.pow(d_org - d_enhance, 2)
        return torch.mean(total)

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)     # [1, 3, H, W]
    enhanced = torch.clamp(original * 1.8 + 0.05, 0.0, 1.0)

    generator = np.random.default_rng(11)
    height, width = image.shape[0], image.shape[1]
    blocks = generator.random(((height + 3) // 4, (width + 3) // 4, 24), dtype=np.float32)
    maps = np.repeat(np.repeat(blocks, 4, axis=0), 4, axis=1)[:height, :width, :] * 2 - 1
    maps = np.ascontiguousarray(maps)
    curve_maps = torch.from_numpy(maps).permute(2, 0, 1).unsqueeze(0)    # [1, 24, H, W]

    globals()["_extra"] = {
        "enhanced": enhanced[0].permute(1, 2, 0).contiguous(),           # [H, W, 3]
        "curve_maps": curve_maps[0].permute(1, 2, 0).contiguous(),       # [H, W, 24]
    }
    return torch.stack([
        l_exp(enhanced),
        l_color(enhanced),
        l_tv(curve_maps),
        l_spa(original, enhanced),
    ]).contiguous()


def run_segformer_loss(image):
    """SegFormer's semantic-segmentation training loss.

    Transcribed from `SegformerForSemanticSegmentation.forward` in transformers: the logits are
    bilinearly upsampled to the label resolution (`align_corners=False`) and scored with
    `CrossEntropyLoss`. Downsampling the labels instead would be cheaper and would throw away the thin
    structures segmentation is judged on.

    As with `zero_dce_losses`, this isolates the loss: the logits and labels are synthesized here and
    written into the record, so both sides score identical tensors and the network is factored out.
    """
    import torch.nn.functional as F

    height, width = image.shape[0], image.shape[1]
    classes = 5
    generator = np.random.default_rng(23)
    # Logits at the decode head's own quarter resolution, as the model produces them.
    logits = generator.normal(size=(1, classes, height // 4, width // 4)).astype(np.float32)
    labels = generator.integers(0, classes, size=(1, height, width)).astype(np.int64)

    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    upsampled = F.interpolate(logits_tensor, size=(height, width), mode="bilinear", align_corners=False)
    loss = torch.nn.CrossEntropyLoss()(upsampled, labels_tensor)

    globals()["_extra"] = {
        "logits": logits_tensor[0].permute(1, 2, 0).contiguous(),        # [h/4, w/4, classes] NHWC
        "labels": labels_tensor[0].to(torch.int32).contiguous(),         # [H, W]
    }
    return loss.reshape(1).contiguous()


def run_fastspeech2(image, checkpoint):
    """The FastSpeech2 conformer acoustic model, from transformers' own implementation, on the
    released espnet/fastspeech2_conformer LJSpeech weights.

    `--checkpoint` is the release DIRECTORY (config.json + pytorch_model.bin). The input is a fixed
    phoneme-id sequence — the ids come from the release's own vocabulary, so no grapheme-to-phoneme
    dependency enters the oracle. The record carries every seam: the encoder output (where a conformer
    mistake shows), the predicted durations (which gate everything downstream — one frame off and the
    mel comparison is meaningless), pitch and energy, and `output` as the post-postnet mel.
    """
    from transformers import FastSpeech2ConformerModel

    model = FastSpeech2ConformerModel.from_pretrained(checkpoint).eval()
    ids = torch.tensor([[13, 5, 30, 22, 17, 41, 9, 25, 33, 4]])
    with torch.no_grad():
        out = model(input_ids=ids, return_dict=True)

    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "encoder_hidden": out.encoder_last_hidden_state[0].float().contiguous(),
        "durations": out.duration_outputs[0].to(torch.int32).contiguous(),
        "pitch": out.pitch_outputs[0].float().contiguous(),
        "energy": out.energy_outputs[0].float().contiguous(),
    }
    return out.spectrogram[0].float().contiguous()


def run_hifigan(image, checkpoint):
    """The HiFi-GAN generator (jik876's own models.py) on the released UNIVERSAL_V1 weights.

    `--checkpoint` is the released `g_*` file; the config beside it is the UNIVERSAL_V1 geometry,
    which is also this port's default. The input is a deterministic synthetic mel — the vocoder is a
    pure function of it, so nothing about speech is assumed — recorded beside the waveform it
    produces. The reference fuses its weight norm (`remove_weight_norm`) before running, which is the
    arithmetic the converter bakes into the safetensors.
    """
    import numpy as np, types, json, os

    source = os.path.join(_reference_source(), "hifigan")
    sys.path.insert(0, source)
    # utils.py imports matplotlib for a plotting helper the generator never calls.
    for name in ["matplotlib", "matplotlib.pylab"]:
        module = types.ModuleType(name)
        module.use = lambda *a, **k: None
        sys.modules.setdefault(name, module)
    from models import Generator
    from env import AttrDict

    config_path = os.path.join(os.path.dirname(checkpoint), "hifigan_universal_v1_config.json")
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    generator = Generator(h)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    generator.load_state_dict(state["generator"], strict=True)
    generator.eval()
    generator.remove_weight_norm()

    frames = 60
    grid_m, grid_t = np.meshgrid(np.arange(h.num_mels), np.arange(frames), indexing="ij")
    mel = (np.sin(grid_m * 0.31 + grid_t * 0.17) * 1.5 - 2.0).astype(np.float32)   # [80, frames]

    with torch.no_grad():
        wave = generator(torch.from_numpy(mel).unsqueeze(0))[0, 0]

    globals()["_extra"] = {"mel": torch.from_numpy(mel).contiguous()}
    return wave.float().contiguous()


def run_music_vocoder(image, checkpoint):
    """The MiniMax Music 3 Flow-VAE decoder (diffusers' own MiniMaxMusic3Vocoder) on the released
    vocoder component.

    `--checkpoint` is the release's `vocoder/` DIRECTORY (config.json + the weight-normed float32
    safetensors); diffusers applies the weight norm itself, which is the arithmetic the Swift loader
    fuses at load. The input is a deterministic standard-normal latent `[1, 128, T]` — the vocoder is
    a pure function of it, so nothing about music is assumed — recorded beside the stereo waveform it
    produces. Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    from diffusers import MiniMaxMusic3Vocoder

    vocoder = MiniMaxMusic3Vocoder.from_pretrained(checkpoint).eval()
    generator = np.random.default_rng(11)
    latents = torch.from_numpy(generator.standard_normal((1, 128, 64)).astype(np.float32))
    with torch.no_grad():
        wave = vocoder(latents)[0]

    globals()["_extra"] = {"latents": latents[0].contiguous()}
    return wave.float().contiguous()


def run_music_depth(image, checkpoint):
    """The MiniMax Music 3 RVQ depth decoder (diffusers' own MiniMaxMusic3RVQDepthDecoder) on the
    released component.

    `--checkpoint` is the release's `rvq_depth_decoder/` DIRECTORY. The record covers every parameter
    family: the transformer forward on a deterministic depth sequence (position embedding, the four
    causal blocks, the final norm), all seven codebook heads on the last step, the shared projection,
    and the offset-packed residual embedding table. The release ships bf16; both sides run float32.
    Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    from diffusers import MiniMaxMusic3RVQDepthDecoder

    decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    generator = np.random.default_rng(13)
    inputs = torch.from_numpy(generator.standard_normal((2, 8, 4096)).astype(np.float32))
    projection_input = torch.from_numpy(generator.standard_normal((2, 4096)).astype(np.float32))
    ids = torch.from_numpy(generator.integers(0, 1024 * 7, size=(2, 7)))
    with torch.no_grad():
        hidden = decoder(inputs)
        head_logits = torch.stack([head(hidden[:, -1]) for head in decoder.audio_heads])
        projected = decoder.projection(projection_input)
        embedded = decoder.audio_embeddings(ids)

    globals()["_extra"] = {
        "inputs_embeds": inputs.contiguous(),
        "head_logits": head_logits.float().contiguous(),
        "projection_input": projection_input.contiguous(),
        "projected": projected.float().contiguous(),
        "embedding_ids": ids.to(torch.int32).contiguous(),
        "embedded": embedded.float().contiguous(),
    }
    return hidden.float().contiguous()


def run_music_condition(image, checkpoint):
    """The MiniMax Music 3 condition encoder (diffusers' own MiniMaxMusic3ConditionEncoder) on the
    released component.

    `--checkpoint` is the release's `condition_encoder/` DIRECTORY. The input is a deterministic
    stand-in for the fused per-frame hidden states `[1, frames, 8 * 4096]`; the output is the
    latent-aligned conditioning, which pins the learned softmax blend, the projection, and the exact
    nearest-neighbor resample from 13 frames to int(13 * 44100/24000 * 960/512) = 44 latents.
    Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    from diffusers import MiniMaxMusic3ConditionEncoder

    encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    generator = np.random.default_rng(17)
    hidden = torch.from_numpy(generator.standard_normal((1, 13, 8 * 4096)).astype(np.float32))
    with torch.no_grad():
        condition = encoder(hidden)

    globals()["_extra"] = {"hidden_states": hidden[0].contiguous()}
    return condition[0].float().contiguous()


def run_music_dit(image, checkpoint):
    """The MiniMax Music 3 flow-matching DiT (diffusers' own MiniMaxMusic3Transformer1DModel) on the
    released component, plus the sigma schedule from diffusers' own FlowMatchEulerDiscreteScheduler.

    `--checkpoint` is the RELEASE ROOT (the transformer loads from its `transformer/` subfolder and
    the scheduler config from `scheduler/`). Velocities are recorded at three timesteps and for the
    zero-condition unconditional branch, because the CFG path runs both; the sigma record pins the
    `invert_sigmas` schedule the pipeline drives the sampler with. The component ships float32.
    Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    import os
    from diffusers import FlowMatchEulerDiscreteScheduler, MiniMaxMusic3Transformer1DModel

    model = MiniMaxMusic3Transformer1DModel.from_pretrained(
        os.path.join(checkpoint, "transformer"), torch_dtype=torch.float32).eval()
    generator = np.random.default_rng(19)
    length = 24
    latents = torch.from_numpy(generator.standard_normal((1, 128, length)).astype(np.float32))
    condition = torch.from_numpy((generator.standard_normal((1, length, 2048)) * 0.5).astype(np.float32))

    extra = {"latents": latents[0].contiguous(), "condition": condition[0].contiguous()}
    with torch.no_grad():
        for name, t in (("t0", 0.0), ("tmid", 0.5), ("tlate", 1.0 - 1.0 / 30.0)):
            velocity = model(latents, torch.tensor([t]), condition, return_dict=False)[0]
            extra[f"velocity_{name}"] = velocity[0].float().contiguous()
        unconditional = model(latents, torch.tensor([0.5]), torch.zeros_like(condition), return_dict=False)[0]
        extra["velocity_unconditional"] = unconditional[0].float().contiguous()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(checkpoint, "scheduler"))
    steps = 30
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps))
    extra["sigmas"] = scheduler.sigmas.float().contiguous()
    extra["timesteps"] = scheduler.timesteps.float().contiguous()

    globals()["_extra"] = extra
    # `output` must not alias an extra: safetensors refuses tensors that share memory.
    return extra["velocity_tmid"].clone()


def run_music_ar(image, checkpoint):
    """The MiniMax Music 3 autoregressive stage: the released Qwen3-8B language model and the depth
    decoder driven by the Diffusers pipeline's OWN helper functions (`_sample_top_k`,
    `_generate_depth_codes`, `_embed_audio_frame`, the prompt cleaners), teacher-forceably.

    `--checkpoint` is the RELEASE ROOT. The record carries the assembled prompt pair, every sampled
    code per loop iteration (so the Swift side replays the same choices and the comparison measures
    the networks rather than two random streams), the prompt prefill's last hidden state, the first
    step's raw and guided logits, and the fused per-frame hidden states that condition synthesis.
    Both sides run bf16 — the model does not fit this machine at float32.
    Runs under the `music` oracle environment (diffusers >= 0.40.0, transformers >= 5)."""
    import os

    from diffusers import MiniMaxMusic3RVQDepthDecoder
    from diffusers.modular_pipelines.minimax_music3 import encoders as ref
    from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

    lm = Qwen3ForCausalLM.from_pretrained(
        os.path.join(checkpoint, "language_model"), torch_dtype=torch.bfloat16).eval()
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        os.path.join(checkpoint, "rvq_depth_decoder"), torch_dtype=torch.bfloat16).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(checkpoint, "tokenizer"))

    # Hoisted out of the f-string: a backslash inside an f-string expression parses only from
    # Python 3.12, and the LLM oracle environment is 3.9, so this file has to stay parseable there.
    lyrics = "[verse]\\nHello world"
    text = (
        f"{ref._IM_START}{ref._CAPTION_START}{ref._clean_caption('Dreamy synth-pop, female vocals')}"
        f"{ref._CAPTION_END}{ref._LYRICS_START}{ref._normalize_lyrics(lyrics)}"
        f"{ref._LYRICS_END}{ref._IM_END}{ref._AUDIO_START}"
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    unconditional = input_ids.clone()
    unconditional[:, 1:-2] = ref._AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((input_ids, unconditional), dim=0)

    generator = torch.Generator().manual_seed(11)
    max_frames = 3

    with torch.no_grad():
        text_embeds = lm.model.embed_tokens(text_ids)
        output = lm.model(inputs_embeds=text_embeds, use_cache=True)
        past = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]
        prefill_hidden = last_hidden.float().clone()

        vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool)
        vocab_mask[ref._AUDIO_CODE_OFFSET : ref._AUDIO_CODE_OFFSET + ref._SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[ref._AUDIO_END_TOKEN_ID] = False

        first_logits = None
        first_guided = None
        frame_hiddens = []
        frame_codes_list = []
        for frame_index in range(max_frames + 1):
            raw = lm.lm_head(last_hidden).float()
            logits = raw.masked_fill(vocab_mask, -float("inf"))
            conditional, uncond_row = logits[0:1], logits[1:2]
            guided = uncond_row + (conditional - uncond_row) * ref._AR_CFG_SCALE
            threshold = torch.topk(conditional, ref._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            if first_logits is None:
                first_logits = raw.clone()
                first_guided = guided.clone()
            sampled = ref._sample_top_k(guided, generator)
            if int(sampled.item()) == ref._AUDIO_END_TOKEN_ID:
                break
            semantic_code = sampled - ref._AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = ref._generate_depth_codes(
                lm, depth, last_hidden, semantic_code.repeat(2), generator)
            frame_codes_list.append(frame_codes[0].tolist())
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1).float())
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = ref._embed_audio_frame(lm, depth, frame_codes)
            output = lm.model(inputs_embeds=feedback, past_key_values=past, use_cache=True)
            past = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

    stacked = torch.stack(frame_hiddens, dim=1)[0]
    globals()["_extra"] = {
        "text_ids": text_ids.to(torch.int32).contiguous(),
        "codes": torch.tensor(frame_codes_list, dtype=torch.int32),
        "prefill_hidden": prefill_hidden.contiguous(),
        "first_logits": first_logits.contiguous(),
        "first_guided": torch.nan_to_num(first_guided, neginf=-1e9)[0].contiguous(),
        "frame_hiddens": stacked.contiguous(),
    }
    return stacked.clone()


# The music prompt cases, mirrored verbatim by NFKMLXMusic3Tests: markdown stripping, the
# <|tag value|> rewrite, structure-tag normalization (text on a tag line is dropped, inline tags
# split onto their own lines, uppercase tags lowercase), multi-byte text, and whitespace forms.
MUSIC_PROMPTS = [
    ("Dreamy synth-pop, female vocals", "[verse]\nHello world"),
    ("# Epic Rock\n- **loud** guitars\n* driving *rhythm*\n---\n• four    spaces",
     "[Verse] ignored text\nFirst line [Chorus] second"),
    ("<|bpm 128|> J-ポップ \U0001f3b5 vocals", "[intro]\nこんにちは ^ 世界"),
    ("  jazz,  with\n\n\nswing!  ", "[verse]\nline one  \n[bridge]\nend"),
]


def run_music_tokenizer(image, checkpoint):
    """The MiniMax Music 3 prompt contract: the Diffusers pipeline's OWN cleaners
    (`_clean_caption`, `_normalize_lyrics`), the special-token template, the release tokenizer, and
    the CFG-row substitution, over MUSIC_PROMPTS.

    `--checkpoint` is the RELEASE ROOT (the tokenizer loads from `tokenizer/`). Each case records
    the `[2, L]` conditional/unconditional id pair; the Swift side rebuilds the same prompts through
    the core byte-level BPE and must match token for token — even whitespace-level changes to the
    assembled prompt change the generated audio.
    Runs under the `music` oracle environment (diffusers >= 0.40.0, transformers >= 5)."""
    import os

    from diffusers.modular_pipelines.minimax_music3 import encoders as ref
    from transformers import Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(checkpoint, "tokenizer"))
    extra = {}
    for index, (caption, lyrics) in enumerate(MUSIC_PROMPTS):
        text = (
            f"{ref._IM_START}{ref._CAPTION_START}{ref._clean_caption(caption)}{ref._CAPTION_END}"
            f"{ref._LYRICS_START}{ref._normalize_lyrics(lyrics)}{ref._LYRICS_END}"
            f"{ref._IM_END}{ref._AUDIO_START}"
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        unconditional = input_ids.clone()
        unconditional[:, 1:-2] = ref._AUDIO_CFG_TOKEN_ID
        extra[f"case{index}_ids"] = torch.cat((input_ids, unconditional), dim=0).to(torch.int32)
    extra["case_count"] = torch.tensor([len(MUSIC_PROMPTS)], dtype=torch.int32)
    globals()["_extra"] = extra
    return extra["case0_ids"][0].clone()


def run_deepseek_v4(image, checkpoint):
    """The DeepSeek V4 decoder's arithmetic, from transformers' own implementation, at a tiny
    configuration with every layer sliding attention.

    The released weights cannot run on any machine here, and DeepSeek's own `inference/model.py`
    imports GPU-only tilelang kernels — but transformers main now carries a complete plain-PyTorch
    implementation (compressor, indexer, hyper-connections, both routers), which is a genuine
    third-party oracle for the arithmetic at a size that fits.

    Every layer is `sliding_attention` and the sequence is shorter than the window, so the reference's
    attention degenerates to exactly the dense-with-sink path the port computes and no compressed KV
    entry exists (only a CLOSED window emits one). That measures the MLA projections, the per-head
    query norm, the trailing interleaved rope, the sink softmax, the output de-rotation, the grouped
    output projection, both routers, the clamped SwiGLU experts, and the hyper-connections.

    The record's weights are saved in the RELEASE naming (transformers' own rename table inverted,
    the fused per-expert tensors split back), so the Swift module — whose keys are the release's —
    loads them strictly with no new loading code. `checkpoint` is unused; pass anything.
    """
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    config = DeepseekV4Config(
        hidden_size=64, num_hidden_layers=4, vocab_size=128,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=16, o_lora_rank=16, o_groups=2, sliding_window=16,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        moe_intermediate_size=32, intermediate_size=32, num_hash_layers=1,
        index_n_heads=4, index_head_dim=8, index_topk=4,
        hc_mult=4, first_k_dense_replace=0, max_position_embeddings=64,
        layer_types=["sliding_attention"] * 4,
        rms_norm_eps=1e-6, routed_scaling_factor=1.5, swiglu_limit=10.0,
        scoring_func="sqrtsoftplus", rope_theta=10000.0,
        hc_sinkhorn_iters=20, hc_eps=1e-6)

    torch.manual_seed(11)
    model = DeepseekV4ForCausalLM(config).eval().float()
    # `nn.Parameter(torch.empty(...))` fields depend on _init_weights; randomize EVERYTHING
    # explicitly so the record does not depend on transformers' init policy of the week.
    state = model.state_dict()
    for key in sorted(state):
        if key.endswith("tid2eid"):
            state[key] = torch.randint(0, config.n_routed_experts,
                                       state[key].shape, dtype=state[key].dtype)
        elif not state[key].is_floating_point():
            continue
        else:
            state[key] = torch.randn(state[key].shape) * 0.05
    model.load_state_dict(state)

    tokens = torch.tensor([[3, 17, 42, 99, 7, 61]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)

    # transformers names -> the release's, which are the Swift module's own keys.
    import re
    def release_name(key):
        rules = [
            (r"^model\.embed_tokens\.weight$", "embed.weight"),
            (r"^lm_head\.weight$", "head.weight"),
            (r"^model\.hc_head\.hc_fn$", "hc_head_fn"),
            (r"^model\.hc_head\.hc_base$", "hc_head_base"),
            (r"^model\.hc_head\.hc_scale$", "hc_head_scale"),
            (r"^model\.norm\.weight$", "norm.weight"),
            (r"^model\.", ""),
        ]
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        key = key.replace(".self_attn.", ".attn.")
        key = key.replace(".mlp.", ".ffn.")
        key = key.replace(".attn_hc.fn", ".hc_attn_fn")
        key = key.replace(".attn_hc.base", ".hc_attn_base")
        key = key.replace(".attn_hc.scale", ".hc_attn_scale")
        key = key.replace(".ffn_hc.fn", ".hc_ffn_fn")
        key = key.replace(".ffn_hc.base", ".hc_ffn_base")
        key = key.replace(".ffn_hc.scale", ".hc_ffn_scale")
        key = key.replace(".input_layernorm.", ".attn_norm.")
        key = key.replace(".post_attention_layernorm.", ".ffn_norm.")
        key = key.replace(".sinks", ".attn_sink")
        key = key.replace(".q_a_proj.", ".wq_a.")
        key = key.replace(".q_a_norm.", ".q_norm.")
        key = key.replace(".q_b_proj.", ".wq_b.")
        key = key.replace(".kv_proj.", ".wkv.")
        key = key.replace(".kv_norm.", ".norm.")
        key = key.replace(".o_a_proj.", ".wo_a.")
        key = key.replace(".o_b_proj.", ".wo_b.")
        key = key.replace(".gate.e_score_correction_bias", ".gate.bias")
        key = key.replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
        key = key.replace(".shared_experts.down_proj.", ".shared_experts.w2.")
        key = key.replace(".shared_experts.up_proj.", ".shared_experts.w3.")
        return key

    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()      # [S, hc_mult, D]
    for key, value in model.state_dict().items():
        name = release_name(key)
        if name.endswith(".experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            gate, up = value.chunk(2, dim=1)                # [E, 2I, H] -> two [E, I, H]
            for expert in range(value.shape[0]):
                extra[f"w::{base}{expert}.w1.weight"] = gate[expert].contiguous()
                extra[f"w::{base}{expert}.w3.weight"] = up[expert].contiguous()
        elif name.endswith(".experts.down_proj"):
            base = name[: -len("down_proj")]
            for expert in range(value.shape[0]):
                extra[f"w::{base}{expert}.w2.weight"] = value[expert].contiguous()
        else:
            extra[f"w::{name}"] = (value.float() if value.is_floating_point()
                                   else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def run_deepseek_quant(image, checkpoint):
    """The dequantization of a real DeepSeek V4 weight, both formats the release stores.

    `--checkpoint` is a safetensors SHARD URL of the release. Only the two tensors this reads are
    fetched, through HTTP range requests, so the record costs a few megabytes rather than the
    checkpoint's hundreds of gigabytes.

    torch decodes the fp8 side: it has `float8_e4m3fn` and `float8_e8m0fnu` on the CPU, so the
    expectation is the reference's own arithmetic. It has NO CPU kernel for `float4_e2m1fn_x2`, so the
    4-bit side decodes through `ml_dtypes`, whose `float4_e2m1fn` is the same format from a different
    vendor. Needs the Python 3.12 interpreter that `oracle_environments` records for gemma; the vision
    oracles' torch predates both dtypes.
    """
    import json, struct, subprocess
    import numpy as np, ml_dtypes

    def fetch(rng):
        done = subprocess.run(["curl", "-sSL", "--fail", "-m", "600", "-A", "InferKit/0.1",
                               "-H", f"Range: bytes={rng}", checkpoint], capture_output=True)
        if done.returncode != 0:
            raise SystemExit(f"range request failed: {done.stderr.decode()[:200]}")
        return done.stdout

    length = struct.unpack("<Q", fetch("0-7"))[0]
    header = json.loads(fetch(f"8-{8 + length - 1}"))
    base = 8 + length

    def tensor(key):
        entry = header[key]
        start, end = entry["data_offsets"]
        return fetch(f"{base + start}-{base + end - 1}"), entry

    extra = {}

    raw, meta = tensor("layers.0.attn.wkv.weight")
    scale_raw, scale_meta = tensor("layers.0.attn.wkv.scale")
    weight = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(*meta["shape"])
    scale = torch.frombuffer(bytearray(scale_raw), dtype=torch.uint8).view(*scale_meta["shape"])
    values = weight.view(torch.float8_e4m3fn).float()
    scales = scale.view(torch.float8_e8m0fnu).float()
    spread = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[: values.shape[0], : values.shape[1]]
    extra["fp8_bytes"] = weight.clone()
    extra["fp8_scale_bytes"] = scale.clone()
    fp8_expected = (values * spread).contiguous()

    raw, meta = tensor("layers.0.ffn.experts.0.w1.weight")
    scale_raw, scale_meta = tensor("layers.0.ffn.experts.0.w1.scale")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(*meta["shape"])
    scale = np.frombuffer(scale_raw, dtype=np.uint8).reshape(*scale_meta["shape"])
    # A byte holds the earlier value in its low nibble.
    low = (packed & 0x0F).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    high = (packed >> 4).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    values = np.stack([low, high], axis=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    scales = scale.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    spread = np.repeat(scales, 32, axis=1)[:, : values.shape[1]]
    extra["fp4_bytes"] = torch.from_numpy(packed.copy())
    extra["fp4_scale_bytes"] = torch.from_numpy(scale.copy())
    extra["fp4_expected"] = torch.from_numpy(np.ascontiguousarray(values * spread))

    globals()["_extra"] = extra
    return fp8_expected



def run_rope_scaling(image):
    """RoPE frequency scaling: the inverse frequencies and attention factor transformers computes.

    Weight-free and architecture-free — scaling is a function of the rotary geometry and the config's
    `rope_scaling` block alone, so this isolates it completely. `ROPE_INIT_FUNCTIONS` is the same
    dispatch every transformers decoder uses, so matching it here matches every model that declares
    one of these.

    Each case writes the parameters it used alongside its result, so the Swift side reads the
    configuration from the record rather than repeating literals that could drift apart from it.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    class Config:
        def __init__(self, dim, base, max_positions, scaling):
            self.rope_theta = base
            self.head_dim = dim
            self.hidden_size = dim
            self.num_attention_heads = 1
            self.max_position_embeddings = max_positions
            self.rope_scaling = scaling
            self.partial_rotary_factor = 1.0
            # transformers 5.x reads `rope_parameters` (the scaling block plus `rope_theta`) and calls
            # `standardize_rope_params`; older releases read `rope_scaling`. Both are served.
            self.rope_parameters = dict(scaling, rope_theta=base)

        def standardize_rope_params(self):
            pass

    # dim, base, max_positions, scaling
    cases = [
        # DeepSeek V4 Pro's shape: a long extension over a short original window.
        (64, 10000.0, 4096, {"rope_type": "yarn", "factor": 40.0,
                             "original_max_position_embeddings": 4096}),
        # A modest extension, and the beta defaults left implicit.
        (128, 10000.0, 32768, {"rope_type": "yarn", "factor": 4.0,
                               "original_max_position_embeddings": 8192}),
        # Non-default ramp boundaries, which move `low`/`high` and so the whole blend.
        (128, 500000.0, 131072, {"rope_type": "yarn", "factor": 8.0,
                                 "original_max_position_embeddings": 16384,
                                 "beta_fast": 16, "beta_slow": 2}),
        # An explicit attention factor overrides the derived one.
        (64, 10000.0, 8192, {"rope_type": "yarn", "factor": 16.0,
                             "original_max_position_embeddings": 2048,
                             "attention_factor": 1.25}),
        # Linear, which is position scaling rather than a frequency blend.
        (64, 10000.0, 8192, {"rope_type": "linear", "factor": 4.0}),
        (64, 500000.0, 131072, {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}),
        (128, 500000.0, 131072, {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                                 "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}),
    ]

    kinds = {"yarn": 1.0, "linear": 0.0, "llama3": 2.0}
    _extra = globals().setdefault("_extra", {})
    _extra.clear()
    outputs = []
    for index, (dim, base, max_positions, scaling) in enumerate(cases):
        config = Config(dim, base, max_positions, scaling)
        initializer = ROPE_INIT_FUNCTIONS[scaling["rope_type"]]
        inv_freq, attention_factor = initializer(config, torch.device("cpu"))
        _extra[f"case{index}_inv_freq"] = inv_freq.float().contiguous()
        _extra[f"case{index}_attention_factor"] = torch.tensor([float(attention_factor)])
        _extra[f"case{index}_params"] = torch.tensor([
            float(dim), float(base), float(max_positions),
            float(scaling["factor"]),
            float(scaling.get("original_max_position_embeddings", max_positions)),
            float(scaling.get("beta_fast", 32)),
            float(scaling.get("beta_slow", 1)),
            kinds[scaling["rope_type"]],
            # -1 marks "the config declares none", so the Swift side exercises the derivation.
            float(scaling.get("attention_factor", -1.0)),
            float(scaling.get("low_freq_factor", 1.0)),
            float(scaling.get("high_freq_factor", 4.0)),
        ])
        outputs.append(inv_freq.float())
    _extra["case_count"] = torch.tensor([len(cases)], dtype=torch.int32)
    return torch.cat(outputs)


def run_rtdetr(image):
    """RT-DETR object detection end to end at a tiny random configuration, from transformers' own
    RTDetrForObjectDetection: the ResNet-D backbone (deep 3-conv stem, avgpool-in-shortcut bottleneck),
    the hybrid encoder (an AIFI transformer on the deepest level + a CSP-RepVGG FPN/PAN), query
    selection over generated anchors, and the deformable-attention decoder with iterative box
    refinement. `anchor_image_size=None`, so anchors are generated per forward; `with_box_refine=True`,
    so class/box heads are cloned per decoder layer. Records the backbone features, the encoder's PAN
    outputs, the query-selection seams, and the per-layer decoder logits/boxes so a divergence localizes.
    License-clean (Apache-2.0). Runs under the `llm` oracle env (transformers). `image` unused.
    """
    from transformers import RTDetrConfig, RTDetrForObjectDetection
    from transformers.models.rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig

    backbone = RTDetrResNetConfig(
        embedding_size=16, hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 1, 1],
        layer_type="bottleneck", downsample_in_bottleneck=False, downsample_in_first_stage=False,
        out_features=["stage2", "stage3", "stage4"], num_channels=3)
    config = RTDetrConfig(
        backbone_config=backbone, use_timm_backbone=False, backbone=None,
        encoder_in_channels=[32, 64, 128], feat_strides=[8, 16, 32], encoder_hidden_dim=32,
        encoder_ffn_dim=48, num_attention_heads=2, encoder_layers=1, encode_proj_layers=[2],
        d_model=32, decoder_attention_heads=2, decoder_ffn_dim=48, decoder_layers=2,
        decoder_n_points=4, num_feature_levels=3, decoder_in_channels=[32, 32, 32],
        num_queries=10, num_denoising=0, learn_initial_query=False, anchor_image_size=None,
        with_box_refine=True, num_labels=4, hidden_expansion=1.0)
    # Randomize the trainable parameters only, NOT the BatchNorm buffers: a randomized `running_var`
    # can go negative, and `rsqrt(var + eps)` is then NaN in both the reference and the port. Left at
    # their init (mean 0, var 1), the frozen BatchNorms stay physical and deterministic.
    model = RTDetrForObjectDetection(config)
    torch.manual_seed(29)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    model = model.eval().float()

    generator = torch.Generator().manual_seed(7)
    pixels = torch.randn(1, 3, 64, 64, generator=generator)

    feats = {}
    model.model.backbone.register_forward_hook(
        lambda m, i, o: feats.__setitem__("bb", [f.detach() for f, _ in o]))
    with torch.no_grad():
        out = model(pixels, output_hidden_states=False, return_dict=True)
        topk_ind = torch.topk(out.enc_outputs_class.max(-1).values, config.num_queries, dim=1).indices

    extra = {
        "pixels": pixels[0].permute(1, 2, 0).contiguous(),                 # [H, W, 3] NHWC
        "pred_boxes": out.pred_boxes[0].clone().contiguous(),              # [Q, 4]
        "enc_class": out.enc_outputs_class[0].contiguous(),               # [S, num_labels]
        "enc_coord": out.enc_outputs_coord_logits[0].contiguous(),         # [S, 4]
        "init_ref": out.init_reference_points[0].contiguous(),             # [Q, 4]
        "topk_ind": topk_ind[0].to(torch.int32).contiguous(),              # [Q]
        "inter_logits": out.intermediate_logits[0].clone().contiguous(),   # [layers, Q, num_labels]
        "inter_ref": out.intermediate_reference_points[0].clone().contiguous(),  # [layers, Q, 4]
        "enc_last": out.encoder_last_hidden_state[-1][0].permute(1, 2, 0).contiguous(),  # last PAN, NHWC
    }
    for level, feature in enumerate(feats["bb"]):
        extra[f"bb.{level}"] = feature[0].permute(1, 2, 0).contiguous()     # backbone stage NHWC
    for key, value in model.state_dict().items():
        if not key.startswith("model.") or key.endswith("num_batches_tracked"):
            continue
        extra[f"w::{key}"] = value.float().contiguous() if value.is_floating_point() else value.contiguous()
    globals()["_extra"] = extra
    return out.logits[0].clone().contiguous()                              # [Q, num_labels]


def run_kokoro(image, checkpoint):
    """Kokoro-82M (StyleTTS2 / iSTFTNet) text-to-speech, from the vendored `kokoro_vendor.KModel` on the
    RELEASED weights — a real-weights end-to-end validation captured seam by seam: the PL-BERT (Albert)
    text encoder, the bert_encoder projection, the duration predictor (DurationEncoder + LSTM + duration
    proj), the F0/energy predictor, the TextEncoder, the alignment-expanded asr, and the iSTFTNet decoder
    audio. The phonemes are supplied directly (misaki is NOT installed), so the model is verified without
    the phonemizer. The sine source's phase noise and additive noise are zeroed so the decoder is
    deterministic and comparable. `checkpoint` is the local Kokoro release directory. Runs under the
    `llm` oracle env (torch, transformers, scipy; the model files are vendored, no spacy).
    """
    import sys
    sys.path.insert(0, os.path.expanduser("~/.inferkit-validation"))
    import kokoro_vendor.istftnet as istftnet
    from kokoro_vendor.model import KModel

    # Make the sine source deterministic: zero the initial phase noise and the additive noise, so the
    # excitation is a pure function of F0 and the whole decoder is reproducible cross-framework.
    original_rand, original_randn = torch.rand, torch.randn_like
    istftnet.torch.rand = lambda *a, **k: torch.zeros(*a, **{key: v for key, v in k.items() if key != "device"})
    istftnet.torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)

    model = KModel(config=os.path.join(checkpoint, "config.json"),
                   model=os.path.join(checkpoint, "kokoro-v1_0.pth")).eval()
    voice = torch.load(os.path.join(checkpoint, "voices/af_heart.pt"), weights_only=True)
    phonemes = "həlˈoʊ wˈɜːld"
    ref_s = voice[len(phonemes) - 1]

    with torch.no_grad():
        input_ids = [i for i in (model.vocab.get(p) for p in phonemes) if i is not None]
        input_ids = torch.LongTensor([[0, *input_ids, 0]])
        input_lengths = torch.LongTensor([input_ids.shape[-1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        lstm_out, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(lstm_out)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        decoder_seams = {}
        model.decoder.encode.register_forward_hook(
            lambda m, i, o: decoder_seams.__setitem__("encode", o.detach()))
        model.decoder.generator.register_forward_pre_hook(
            lambda m, args: decoder_seams.__setitem__("gen_in", args[0].detach()))
        model.decoder.generator.conv_post.register_forward_hook(
            lambda m, i, o: decoder_seams.__setitem__("conv_post", o.detach()))
        model.decoder.generator.m_source.register_forward_hook(
            lambda m, i, o: decoder_seams.__setitem__("har_source", o[0].detach()))
        audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

    istftnet.torch.rand, istftnet.torch.randn_like = original_rand, original_randn

    extra = {
        "ids": input_ids[0].to(torch.int32).contiguous(),
        "ref_s": ref_s[0].contiguous(),                                    # [256]
        "bert": bert_dur[0].contiguous(),                                  # [T, 768]
        "d_en": d_en[0].contiguous(),                                      # [512, T]
        "d": d[0].contiguous(),                                            # [T, 640]
        "duration": duration[0].contiguous(),                             # [T] pre-round
        "pred_dur": pred_dur.to(torch.int32).contiguous(),                 # [T]
        "f0": F0_pred[0].contiguous(),                                     # [frames]
        "n": N_pred[0].contiguous(),                                       # [frames]
        "t_en": t_en[0].contiguous(),                                      # [512, T]
        "asr": asr[0].contiguous(),                                        # [512, frames]
        "dec_encode": decoder_seams["encode"][0].contiguous(),            # [1024, frames]
        "dec_gen_in": decoder_seams["gen_in"][0].contiguous(),            # [512, frames·2]
        "dec_conv_post": decoder_seams["conv_post"][0].contiguous(),      # [22, T_stft]
        "dec_har_source": decoder_seams["har_source"][0].contiguous(),    # [L, 1] the sine excitation
    }
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous() if value.is_floating_point() else value.contiguous()
    globals()["_extra"] = extra
    return audio.contiguous()                                              # [samples]


def run_rtdetr_real(image, checkpoint):
    """RT-DETR on the RELEASED `PekingU/rtdetr_r50vd` weights, from transformers' own
    RTDetrForObjectDetection and its image processor — a real-weights end-to-end validation (the r50vd
    ResNet-50-vd geometry, the actual checkpoint, and the loader, including the stage-1 stride-1 shortcut
    the tiny config does not exercise). `checkpoint` is the local release directory. Records the
    preprocessed pixel values so the port runs on the identical input, plus the reference's query
    selection. Runs under the `llm` oracle env (transformers, needs Pillow).
    """
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    from PIL import Image

    model = RTDetrForObjectDetection.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype("uint8"))
    pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]   # [1, 3, 640, 640]
    with torch.no_grad():
        out = model(pixel_values, return_dict=True)
        topk = torch.topk(out.enc_outputs_class.max(-1).values, model.config.num_queries, dim=1).indices

    globals()["_extra"] = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),                # [640, 640, 3] NHWC
        "pred_boxes": out.pred_boxes[0].clone().contiguous(),                   # [300, 4]
        "topk_ind": topk[0].to(torch.int32).contiguous(),                       # [300]
    }
    return out.logits[0].clone().contiguous()                                   # [300, 80]


def run_parakeet(image, checkpoint):
    """NVIDIA Parakeet-TDT (0.6B v2) speech recognition, from NeMo's own EncDecRNNTBPEModel on the
    RELEASED `.nemo` archive (`--checkpoint`): the FastConformer encoder (dw-striding 8x subsampling,
    24 rel-pos conformer layers, no biases) and the token-and-duration transducer (an LSTM prediction
    net, the joint, greedy TDT decoding over durations [0..4]). The clip is the validation speech WAV
    (`IK_VAL_AUDIO`, 16 kHz), so the transcription is a real measurement. Recorded seam by seam: the
    normalized mel features, the pre-encode output, the first conformer layer, the encoder output, the
    joint logits for frame 0 at the blank/SOS state, and the greedy tokens, text, and frame timestamps.
    Dither is zeroed (training-time noise). Runs under the `nemo` oracle env (nemo_toolkit[asr]).
    """
    import wave as wavemodule
    import huggingface_hub
    for name in ["ModelFilter", "DatasetFilter"]:
        if not hasattr(huggingface_hub, name):
            setattr(huggingface_hub, name, type(name, (), {}))
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(checkpoint, strict=False).eval()
    model.preprocessor.featurizer.dither = 0.0

    path = os.environ.get("IK_VAL_AUDIO", os.path.expanduser("~/.inferkit-validation/inputs/speech.wav"))
    with wavemodule.open(path) as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1 and handle.getsampwidth() == 2
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    wave = (pcm.astype(np.float32) / 32768.0)
    signal = torch.from_numpy(wave)[None]
    length = torch.tensor([wave.shape[0]])

    seams = {}
    model.encoder.pre_encode.register_forward_hook(lambda m, i, o: seams.__setitem__("pre", o[0].detach()))
    model.encoder.layers[0].register_forward_hook(lambda m, i, o: seams.__setitem__("layer0", o.detach()))
    with torch.no_grad():
        features, feature_length = model.preprocessor(input_signal=signal, length=length)
        encoded, encoded_length = model.encoder(audio_signal=features, length=feature_length)  # [1, D, T]
        g, _ = model.decoder.predict(None, None, add_sos=False, batch_size=1)            # SOS state
        f = encoded.transpose(1, 2)[:, 0:1, :]
        joint0 = model.joint.joint(f, g)                                                 # [1, 1, 1, V+1+durations]
        hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_length, return_hypotheses=True)
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    hypothesis = hypotheses[0]
    tokens = list(hypothesis.y_sequence.tolist() if hasattr(hypothesis.y_sequence, "tolist") else hypothesis.y_sequence)
    text = model.tokenizer.ids_to_text(tokens)
    timestamps = list(hypothesis.timestamp.tolist() if hasattr(hypothesis.timestamp, "tolist") else hypothesis.timestamp)
    print(f"parakeet transcription: {text!r} ({len(tokens)} tokens, {int(encoded_length[0])} frames)")

    extra = {
        "waveform": torch.from_numpy(wave).contiguous(),
        "features": features[0].transpose(0, 1).contiguous(),               # [frames, mels] normalized
        "pre": seams["pre"][0].contiguous(),                                 # [T', d_model]
        "layer0": seams["layer0"][0].contiguous(),                           # [T', d_model]
        "encoded": encoded[0].transpose(0, 1).contiguous(),                  # [T', d_model]
        "joint0": joint0[0, 0, 0].contiguous(),                              # [V + 1 + durations]
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "timestamps": torch.tensor(timestamps, dtype=torch.int32),
        "text": torch.tensor(list(text.encode("utf-8")), dtype=torch.int32),
    }
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked"):
            continue
        extra[f"w::{key}"] = value.float().contiguous() if value.is_floating_point() else value.contiguous()
    globals()["_extra"] = extra
    return encoded[0].transpose(0, 1).clone().contiguous()


def _chatterbox_reference_audio():
    """The validation clip prepared the way `ChatterboxTTS.prepare_conditionals` prepares a voice
    prompt: `librosa.load(sr=24000)` (a 16 kHz file resampled up), then `librosa.resample` back down to
    16 kHz. Returns (wav24, wav16) as float32 numpy arrays."""
    import librosa
    path = os.environ.get("IK_VAL_AUDIO", os.path.expanduser("~/.inferkit-validation/inputs/speech.wav"))
    wav24, _ = librosa.load(path, sr=24000)
    wav16 = librosa.resample(wav24, orig_sr=24000, target_sr=16000)
    return wav24.astype(np.float32), wav16.astype(np.float32)


def run_chatterbox_voice(image, checkpoint):
    """Chatterbox stage 1 + 2 (ResembleAI, `chatterbox-tts`): the VoiceEncoder speaker embedding and
    the S3 speech tokenizer, on the RELEASED weights (`--checkpoint` is the release directory holding
    `ve.safetensors` and `s3gen.safetensors`) and the validation clip, prepared exactly as
    `prepare_conditionals` does. Recorded: `wav16` (the 16 kHz prompt both networks read); the voice
    encoder's librosa-trimmed input (`ve_wav`), unscaled 40-band power mel (`ve_mel`, [T, 40]), the
    per-partial embeddings (`ve_partials`, [N, 256]) and the utterance embedding (`ve_embed`); the S3
    tokenizer's log-mel on the six-second crop (`s3_mel`, [128, T]), the encoder output (`s3_hidden`,
    [T', 1280]) and the FSQ codes truncated to the 150-token conditioning prompt (`s3_codes`), and the
    codes of the whole 16 kHz prompt as `embed_ref` tokenizes it (`s3_ref_codes`, from torchaudio's
    24 kHz to 16 kHz resample, recorded as `ref_wav16`). Weights as `w::ve.*` and `w::tokenizer.*`.
    Runs under the `chatterbox` oracle env.
    """
    import librosa
    import torchaudio
    from safetensors.torch import load_file
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.voice_encoder.melspec import melspectrogram
    from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
    from chatterbox.models.s3gen.const import S3GEN_SR

    directory = checkpoint
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(directory, "ve.safetensors")))
    ve.eval()
    s3gen_state = load_file(os.path.join(directory, "s3gen.safetensors"))
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    tokenizer_state = {k[len("tokenizer."):]: v for k, v in s3gen_state.items() if k.startswith("tokenizer.")}
    missing, unexpected = tokenizer.load_state_dict(tokenizer_state, strict=False)
    assert not unexpected and set(missing) <= {"_mel_filters", "window"}, (missing, unexpected)
    tokenizer.eval()

    wav24, wav16 = _chatterbox_reference_audio()

    # Voice encoder, as embeds_from_wavs: trim at 20 dB, unscaled power mel, partials at rate 1.3.
    ve_wav = librosa.effects.trim(wav16, top_db=20)[0]
    ve_mel = melspectrogram(ve_wav, ve.hp).T                                  # [T, 40]
    from chatterbox.models.voice_encoder.voice_encoder import get_frame_step, get_num_wins
    frame_step = get_frame_step(0.5, 1.3, ve.hp)
    n_partials, target_len = get_num_wins(len(ve_mel), frame_step, 0.8, ve.hp)
    mel_t = torch.from_numpy(np.ascontiguousarray(ve_mel)).float()
    if target_len > mel_t.shape[0]:
        mel_t = torch.cat([mel_t, torch.zeros(target_len - mel_t.shape[0], ve.hp.num_mels)], dim=0)
    partials = torch.stack([mel_t[i * frame_step: i * frame_step + ve.hp.ve_partial_frames] for i in range(n_partials)])
    with torch.inference_mode():
        partial_embeds = ve(partials)                                          # [N, 256]
        ve_embed = torch.from_numpy(ve.embeds_from_wavs([wav16], sample_rate=S3_SR))  # [1, 256]
    assert ve_embed.shape == (1, 256)
    print(f"chatterbox voice encoder: trimmed {len(ve_wav)} of {len(wav16)} samples, {len(ve_mel)} mel frames, "
          f"{n_partials} partials at step {frame_step}")

    # S3 tokenizer on the six-second crop (the T3 conditioning prompt), seam by seam.
    crop = torch.from_numpy(wav16[: 6 * S3_SR])[None]
    with torch.inference_mode():
        s3_mel = tokenizer.log_mel_spectrogram(crop)                           # [1, 128, T]
        s3_mel = s3_mel[..., :150 * 4]
        mel_len = torch.tensor([s3_mel.shape[-1]])
        hidden, code_len = tokenizer.encoder(s3_mel, mel_len)                  # [1, T', 1280]
        codes = tokenizer.quantizer.encode(hidden)                             # [1, T']
        cond_codes, cond_len = tokenizer.forward([wav16[: 6 * S3_SR]], max_len=150)
    assert torch.equal(codes[0, :cond_len[0]], cond_codes[0, :cond_len[0]])

    # The whole prompt as `S3Gen.embed_ref` tokenizes it: torchaudio's resample of the 24 kHz wav.
    ref_wav16 = torchaudio.transforms.Resample(S3GEN_SR, S3_SR)(torch.from_numpy(wav24)[None])
    with torch.inference_mode():
        ref_codes, ref_len = tokenizer(ref_wav16.float())
    print(f"chatterbox s3 tokenizer: {s3_mel.shape[-1]} mel frames -> {int(code_len[0])} codes; "
          f"ref {ref_wav16.shape[-1]} samples -> {int(ref_len[0])} codes; first codes {codes[0, :8].tolist()}")

    extra = {
        "wav24": torch.from_numpy(wav24).contiguous(),
        "wav16": torch.from_numpy(wav16).contiguous(),
        "ve_wav": torch.from_numpy(np.ascontiguousarray(ve_wav)).contiguous(),
        "ve_mel": torch.from_numpy(np.ascontiguousarray(ve_mel)).float().contiguous(),
        "ve_partials": partial_embeds.contiguous(),
        "ve_embed": ve_embed[0].contiguous(),
        "s3_mel": s3_mel[0].contiguous(),
        "s3_hidden": hidden[0].contiguous(),
        "s3_codes": codes[0, :cond_len[0]].to(torch.int32).contiguous(),
        "ref_wav16": ref_wav16[0].contiguous(),
        "s3_ref_codes": ref_codes[0, :ref_len[0]].to(torch.int32).contiguous(),
    }
    for key, value in ve.state_dict().items():
        extra[f"w::ve.{key}"] = value.float().contiguous()
    for key, value in tokenizer_state.items():
        extra[f"w::tokenizer.{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return ve_embed[0].clone().contiguous()


def run_chatterbox_t3(image, checkpoint):
    """Chatterbox stage 3: T3, the text-to-speech-token model (a Llama 520M with llama3 rope scaling
    over learned position embeddings, conditioned on the VoiceEncoder speaker embedding, a 32-token
    Perceiver resample of the prompt's speech codes, and an emotion scalar), on the RELEASED
    `t3_cfg.safetensors`. The conditioning is rebuilt from the validation clip exactly as
    `prepare_conditionals` builds it. Recorded: the text tokens with SOT/EOT (`text_tokens`), the
    speaker embedding and prompt codes, the conditioning sequence (`cond_emb`, [34, 1024]), the CFG pair
    of input embeddings (`embeds`, [2, L, 1024], row 1 with the text embedding zeroed), the reference's
    own sampled speech tokens under a fixed seed (`speech_tokens`), the teacher-forced speech logits over
    that sequence for both CFG rows (`tf_logits`, [2, n, 8194]), and the first step's fully processed
    logits (`step0_processed`: CFG 0.5, repetition penalty 1.2, temperature 0.8, min-p 0.05, top-p 1.0).
    The 2 GB of weights are not copied into the record; the Swift side loads the release itself. Runs
    under the `chatterbox` oracle env.
    """
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers.generation.logits_process import (RepetitionPenaltyLogitsProcessor, MinPLogitsWarper,
                                                        TopPLogitsWarper)
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.tokenizers import EnTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
    from chatterbox.tts import punc_norm

    directory = checkpoint
    t3 = T3()
    t3.load_state_dict(load_file(os.path.join(directory, "t3_cfg.safetensors")))
    t3.eval()
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(directory, "ve.safetensors")))
    ve.eval()
    s3gen_state = load_file(os.path.join(directory, "s3gen.safetensors"))
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    tokenizer.load_state_dict({k[len("tokenizer."):]: v for k, v in s3gen_state.items() if k.startswith("tokenizer.")},
                              strict=False)
    tokenizer.eval()
    del s3gen_state
    text_tokenizer = EnTokenizer(os.path.join(directory, "tokenizer.json"))

    wav24, wav16 = _chatterbox_reference_audio()
    with torch.inference_mode():
        ve_embed = torch.from_numpy(ve.embeds_from_wavs([wav16], sample_rate=S3_SR)).mean(axis=0, keepdim=True)
        cond_tokens, _ = tokenizer.forward([wav16[: 6 * S3_SR]], max_len=t3.hp.speech_cond_prompt_len)
    cond_tokens = torch.atleast_2d(cond_tokens)
    exaggeration = 0.5
    t3_cond = T3Cond(speaker_emb=ve_embed, cond_prompt_speech_tokens=cond_tokens,
                     emotion_adv=exaggeration * torch.ones(1, 1, 1))

    raw_text = "The quick brown fox jumps over the lazy dog."
    text = punc_norm(raw_text)
    text_tokens = text_tokenizer.text_to_tokens(text)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    text_tokens = F.pad(text_tokens, (1, 0), value=t3.hp.start_text_token)
    text_tokens = F.pad(text_tokens, (0, 1), value=t3.hp.stop_text_token).long()

    cfg_weight, temperature, repetition_penalty, min_p, top_p = 0.5, 0.8, 1.2, 0.05, 1.0
    sos = t3.hp.start_speech_token
    with torch.inference_mode():
        cond_emb = t3.prepare_conditioning(t3_cond)                                         # [1, 34, 1024]
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens,
            speech_tokens=sos * torch.ones_like(text_tokens[:, :1]), cfg_weight=cfg_weight)  # [2, L0, 1024]
        torch.manual_seed(0)
        speech_tokens = t3.inference(t3_cond=t3_cond, text_tokens=text_tokens, max_new_tokens=1000,
                                     temperature=temperature, cfg_weight=cfg_weight,
                                     repetition_penalty=repetition_penalty, min_p=min_p, top_p=top_p)
        gen = speech_tokens[0].long()                                                        # [n]
        n = gen.numel()
        bos_embed = t3.speech_emb(torch.tensor([[sos]])) + t3.speech_pos_emb.get_fixed_embedding(0)
        gen_embed = t3.speech_emb(gen[None, :-1]) + t3.speech_pos_emb.get_fixed_embedding(torch.arange(1, n))
        tf_embeds = torch.cat([embeds, torch.cat([bos_embed, bos_embed]), torch.cat([gen_embed, gen_embed])], dim=1)
        out = t3.tfmr(inputs_embeds=tf_embeds, output_hidden_states=True, return_dict=True)
        logits = t3.speech_head(out.hidden_states[-1])                                       # [2, L, 8194]
        start = len_cond + text_tokens.size(1) + 1                                           # the second BOS
        tf_logits = logits[:, start: start + n]                                               # predicts gen[0..n-1]
        step0 = tf_logits[:, 0]
        cfg = step0[0:1] + cfg_weight * (step0[0:1] - step0[1:2])
        ids = torch.tensor([[sos]])
        processed = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)(ids, cfg) / temperature
        processed = MinPLogitsWarper(min_p=min_p)(ids, processed)
        processed = TopPLogitsWarper(top_p=top_p)(ids, processed)
    assert bool(torch.isfinite(processed[0, gen[0]])), "the sampled first token is admissible under the processors"
    print(f"chatterbox t3: text {text!r} -> {text_tokens.size(1)} tokens; cond {cond_emb.shape[1]}; "
          f"{n} speech tokens sampled (last {int(gen[-1])}), embeds {tuple(embeds.shape)}")

    extra = {
        "text_tokens": text_tokens[0].to(torch.int32).contiguous(),
        "speaker_emb": ve_embed[0].contiguous(),
        "cond_tokens": cond_tokens[0].to(torch.int32).contiguous(),
        "emotion_adv": torch.tensor([exaggeration]),
        "cond_emb": cond_emb[0].contiguous(),
        "embeds": embeds.contiguous(),
        "speech_tokens": gen.to(torch.int32).contiguous(),
        "tf_logits": tf_logits.contiguous(),
        "step0_processed": processed[0].contiguous(),
        "text": torch.tensor(list(raw_text.encode("utf-8")), dtype=torch.int32),
    }
    globals()["_extra"] = extra
    return tf_logits[0, 0].clone().contiguous()


def run_chatterbox_s3gen(image, checkpoint):
    """Chatterbox stages 4 and 5: S3Gen, the speech-code-to-waveform decoder, on the RELEASED
    `s3gen.safetensors` and the validation clip as the voice prompt (`embed_ref` on its first ten
    seconds at 24 kHz). Stage 4 is the CAMPPlus x-vector over a Kaldi fbank, the 24 kHz prompt mel, the
    UpsampleConformerEncoder over the prompt-plus-target codes, and the causal conditional flow matching
    (ten Euler steps on a cosine schedule, classifier-free guidance 0.7) through the ConditionalDecoder
    U-Net; stage 5 is the HiFT vocoder (ConvRNNF0Predictor, harmonic sine source, iSTFT head). The
    target codes are the T3 record's sampled speech tokens when that record exists (the real pipeline's
    input), else the prompt's own codes. Two random sources are pinned so the seams are exact: the flow's
    initial noise is drawn once and RECORDED (`flow_noise`, the Swift side takes it as an input), and the
    sine source's random harmonic phases and additive noise are zeroed (the reference's `SineGen` draws
    them fresh per call; the port draws its own in the consumer path). Recorded seam by seam: `fbank`,
    `xvector_head`, `xvector`, `prompt_mel`, `token_emb`, `encoder_h`, `mu`, `spk80`, `t_span`,
    `estimator0` (the first Euler step's velocity, both guidance rows), `mel_full`, `mel`, `f0`,
    `source`, and `wav` (with the reference's 40 ms leading fade). Runs under the `chatterbox` oracle env.
    """
    import torch.nn.functional as F
    import torchaudio
    import torchaudio.compliance.kaldi as Kaldi
    from safetensors.torch import load_file, load_file as _lf
    from chatterbox.models.s3gen import S3Gen, S3GEN_SR
    from chatterbox.models.s3tokenizer import S3_SR
    import chatterbox.models.s3gen.hifigan as hifigan_module

    directory = checkpoint
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(os.path.join(directory, "s3gen.safetensors")), strict=False)
    s3gen.eval()

    wav24, wav16 = _chatterbox_reference_audio()
    ref24 = torch.from_numpy(wav24[: 10 * S3GEN_SR])[None]
    ref16 = torchaudio.transforms.Resample(S3GEN_SR, S3_SR)(ref24)

    seams = {}
    def record(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            seams.setdefault(name, []).append(value.detach().clone())
        return hook
    hooks = [
        s3gen.speaker_encoder.head.register_forward_hook(record("xvector_head")),
        s3gen.flow.input_embedding.register_forward_hook(record("token_emb")),
        s3gen.flow.encoder.register_forward_hook(record("encoder_h")),
        s3gen.flow.encoder_proj.register_forward_hook(record("mu")),
        s3gen.flow.spk_embed_affine_layer.register_forward_hook(record("spk80")),
        s3gen.flow.decoder.register_forward_hook(record("mel_full")),
        s3gen.mel2wav.f0_predictor.register_forward_hook(record("f0")),
        s3gen.mel2wav.m_source.register_forward_hook(record("source")),
    ]

    # `solve_euler` calls `estimator.forward` directly, which bypasses forward hooks; wrap the method.
    estimator = s3gen.flow.decoder.estimator
    original_estimator_forward = estimator.forward
    def recording_estimator_forward(*args, **kwargs):
        output = original_estimator_forward(*args, **kwargs)
        seams.setdefault("estimator", []).append(output.detach().clone())
        return output
    estimator.forward = recording_estimator_forward

    t3_record = os.path.expanduser("~/.inferkit-validation/records/chatterbox_t3.safetensors")
    with torch.inference_mode():
        fbank = Kaldi.fbank(ref16, num_mel_bins=80)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        ref_dict = s3gen.embed_ref(ref24, S3GEN_SR)
        if os.path.exists(t3_record):
            tokens = _lf(t3_record)["speech_tokens"].long()
            tokens = tokens[tokens < 6561]
            source = "t3 record"
        else:
            tokens = ref_dict["prompt_token"][0].long()
            source = "prompt codes"
        n_prompt, n_tokens = ref_dict["prompt_token"].shape[1], tokens.numel()
        total_mel = 2 * (n_prompt + n_tokens)
        torch.manual_seed(0)
        noise = torch.randn(1, 80, total_mel)

    class ZeroUniform:
        def __init__(self, low, high): pass
        def sample(self, sample_shape):
            return torch.zeros(sample_shape)
    original_uniform, original_randn_like = hifigan_module.Uniform, torch.randn_like
    def pinned_randn_like(tensor, **kwargs):
        if tuple(tensor.shape) == tuple(noise.shape):
            return noise.to(tensor.dtype)
        return torch.zeros_like(tensor)
    hifigan_module.Uniform = ZeroUniform
    torch.randn_like = pinned_randn_like
    try:
        with torch.inference_mode():
            wav, _ = s3gen.inference(speech_tokens=tokens, ref_dict=ref_dict)
    finally:
        hifigan_module.Uniform = original_uniform
        torch.randn_like = original_randn_like
        for hook in hooks:
            hook.remove()
        estimator.forward = original_estimator_forward

    t_span = torch.linspace(0, 1, 11)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    mel_full = seams["mel_full"][0]                         # [1, 80, total]
    mel = mel_full[:, :, ref_dict["prompt_feat"].shape[1]:]
    print(f"chatterbox s3gen: prompt {n_prompt} codes / {ref_dict['prompt_feat'].shape[1]} mel frames, "
          f"{n_tokens} target codes ({source}) -> mel {tuple(mel.shape)}, wav {tuple(wav.shape)}; "
          f"{len(seams['estimator'])} estimator calls")
    assert len(seams["estimator"]) == 10

    extra = {
        "ref_wav24": ref24[0].contiguous(),
        "ref_wav16": ref16[0].contiguous(),
        "fbank": fbank.contiguous(),                                          # [T, 80]
        "xvector_head": seams["xvector_head"][0][0].contiguous(),             # [C·F/8, T']
        "xvector": ref_dict["embedding"][0].contiguous(),                     # [192]
        "prompt_mel": ref_dict["prompt_feat"][0].contiguous(),                # [T2, 80]
        "prompt_tokens": ref_dict["prompt_token"][0].to(torch.int32).contiguous(),
        "speech_tokens": tokens.to(torch.int32).contiguous(),
        "token_emb": seams["token_emb"][0][0].contiguous(),                   # [T_all, 512]
        "encoder_h": seams["encoder_h"][0][0].contiguous(),                   # [2·T_all, 512]
        "mu": seams["mu"][0][0].contiguous(),                                 # [2·T_all, 80]
        "spk80": seams["spk80"][0][0].contiguous(),                           # [80]
        "flow_noise": noise[0].contiguous(),                                  # [80, total]
        "t_span": t_span.contiguous(),
        "estimator0": seams["estimator"][0].contiguous(),                     # [2, 80, total]
        "mel_full": mel_full[0].contiguous(),                                 # [80, total]
        "mel": mel[0].contiguous(),                                           # [80, 2·n_tokens]
        "f0": seams["f0"][0][0].contiguous(),                                 # [2·n_tokens]
        "source": seams["source"][0][0, :, 0].contiguous(),                   # [samples]
        "wav": wav[0].contiguous(),                                           # [samples]
    }
    globals()["_extra"] = extra
    return wav[0].clone().contiguous()


MODELS = {"sd_scheduler": run_sd_scheduler, "clip": run_clip, "segformer": run_segformer, "zero_dce_losses": run_zero_dce_losses,
          "segformer_loss": run_segformer_loss,
          "clip_text": run_clip_text, "sd_tokenizer": run_sd_tokenizer,
          "rope_scaling": run_rope_scaling, "silero_vad": run_silero_vad, "dac": run_dac,
          "snac": run_snac, "siglip2": run_siglip2, "taesd": run_taesd, "ltx_vae": run_ltx_vae, "ltx_transformer": run_ltx_transformer, "ltx_t5": run_ltx_t5, "z_image": run_z_image, "sana": run_sana, "wan": run_wan, "flux_vae": run_flux_vae, "dc_ae": run_dc_ae, "wan_vae": run_wan_vae, "dpm_solver": run_dpm_solver, "unipc": run_unipc, "gemma2": run_gemma2, "umt5": run_umt5, "wan_vae_21": run_wan_vae_21, "dc_ae_real": run_dc_ae_real, "ip_adapter": run_ip_adapter, "rtdetr": run_rtdetr}
CHECKPOINT_MODELS = {"sam_encoder": run_sam_encoder, "sam2_encoder": run_sam2_encoder, "sam2_decoder": run_sam2_decoder, "sam2_memory": run_sam2_memory, "sam": run_sam, "sam_decoder": run_sam_decoder,
                     "swinir": run_swinir,
                     "sd_unet": run_sd_unet, "sd_vae": run_sd_vae, "sd_text_encoder": run_sd_text_encoder, "sd_text_to_image": run_sd_text_to_image, "convtasnet": run_convtasnet, "demucs": run_demucs, "htdemucs": run_htdemucs, "denoiser": run_denoiser,
                     "vad": run_vad, "deeplab": run_deeplab, "u2net": run_u2net, "pose": run_pose,
                     "audio_tagger": run_audio_tagger, "raft": run_raft, "rvm": run_rvm,
                     "depth": run_depth, "depth_encoder": run_depth_encoder,
                     "videosr": run_videosr, "yolo": run_yolo, "nafnet": run_nafnet, "rife": run_rife, "rife_v4": run_rife_v4, "modnet": run_modnet, "bisenet": run_bisenet, "bisenetv2": run_bisenetv2, "siggraph17": run_siggraph17, "whisper": run_whisper, "lama": run_lama, "yolo_detections": run_yolo_detections, "codeformer": run_codeformer, "retinaface": run_retinaface, "qwen3": run_qwen3, "qwen3_embedding": run_qwen3_embedding, "embeddinggemma": run_embeddinggemma, "modernbert_reranker": run_modernbert_reranker, "smolvlm": run_smolvlm, "qwen3vl": run_qwen3vl, "gguf": run_gguf, "gguf_lm": run_gguf_lm, "qwen3_moe": run_qwen3_moe, "mixtral": run_mixtral, "gemma4": run_gemma4, "gemma4_moe": run_gemma4_moe, "gemma4_unified": run_gemma4_unified, "gemma4_vision": run_gemma4_vision, "gemma4_audio": run_gemma4_audio, "gemma4_mel": run_gemma4_mel, "gemma4_embedder": run_gemma4_embedder, "gemma4_audio_real": run_gemma4_audio_real, "gemma4_conditional_real": run_gemma4_conditional_real, "gemma4_vision_real": run_gemma4_vision_real, "qwen3_5": run_qwen3_5, "deepseek_quant": run_deepseek_quant, "deepseek_v4": run_deepseek_v4, "hifigan": run_hifigan, "fastspeech2": run_fastspeech2, "music_vocoder": run_music_vocoder, "music_depth": run_music_depth, "music_condition": run_music_condition, "music_dit": run_music_dit, "music_ar": run_music_ar, "music_tokenizer": run_music_tokenizer,
                     "zero_dce": run_zero_dce, "style_transfer": run_style_transfer,
                     "realesrgan": run_realesrgan, "colorizer": run_colorizer, "rtdetr_real": run_rtdetr_real, "kokoro": run_kokoro, "parakeet": run_parakeet,
                     "chatterbox_voice": run_chatterbox_voice,
                     "chatterbox_t3": run_chatterbox_t3,
                     "chatterbox_s3gen": run_chatterbox_s3gen}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS) + sorted(CHECKPOINT_MODELS), help="which reference to run")
    parser.add_argument("output", help="path to write the .safetensors record")
    parser.add_argument("--size", type=int, default=224, help="input plate size (square)")
    parser.add_argument("--plate", choices=["blocks", "subject"], default="blocks",
                        help="blocks: unstructured texture. subject: a foreground shape, for the "
                             "saliency and matting models, whose reference output is near-constant "
                             "on unstructured input")
    parser.add_argument("--checkpoint", help="local checkpoint path, for references that need one (sam)")
    parser.add_argument("--image", help="a real photo to use as the plate, resized to --size. For "
                                        "models whose reference output is degenerate on synthetic "
                                        "plates (human matting finds no person in an ellipse)")
    args = parser.parse_args()

    if args.image:
        from PIL import Image
        photo = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        image = np.ascontiguousarray(np.asarray(photo).astype(np.float32) / 255.0)
    else:
        image = (subject_image(args.size, args.size) if args.plate == "subject"
                 else deterministic_image(args.size, args.size))
    if args.model in CHECKPOINT_MODELS:
        if not args.checkpoint:
            raise SystemExit(f"{args.model} needs --checkpoint")
        result = CHECKPOINT_MODELS[args.model](image, args.checkpoint)
    else:
        result = MODELS[args.model](image)
    record = {"input_image": torch.from_numpy(image).contiguous(), "output": result.float()}
    for name, value in globals().get("_extra", {}).items():
        # Integer extras stay integer: class labels are indices, and casting them to float would make
        # the Swift side guess at the conversion back.
        record[name] = value.float() if value.is_floating_point() else value
    save_file(record, args.output)
    print(f"wrote {args.output}: input {tuple(image.shape)}, reference output {tuple(result.shape)}")


if __name__ == "__main__":
    sys.exit(main())
