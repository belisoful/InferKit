"""The Wan-Dancer oracles `run_reference.py` runs (modes `wan_dancer_music`, `wan_dancer_planning`,
`wan_dancer_pipeline`), each returning its record as a dict of tensors. The LoRA objective,
`wan_dancer_lora_diffsynth`, runs DiffSynth-Studio's own Wan-Dancer code in `run_reference.py`.

- `music()`: Wan-Dancer's music features (`gen_video_global.get_music_base_feature`) under librosa 0.10.2 for a
  synthesized 15360 Hz clip, every seam with the constant-Q decimation through `polyphase` (`poly.*`), and the
  features through librosa's own soxr path (`ref.*`).
- `planning()`: the local stage's planning as `gen_video_local.py` computes it, the keyframe masks of
  `process_global_video_firstlastframe` for several track lengths over a 38-frame global video (its frames stubbed,
  as the masks read only their count), and the music clips of `get_music_clip_149f` for several synthesized tracks
  (their count, sample spans, the 16-bit PCM samples of the last clip, and the container duration moviepy reads).
- `pipeline(checkpoint)`: one global-stage and one local-stage call of the fork's `WanVideoPipeline.__call__` as the
  released scripts make them, at 3 steps, 17 frames, and 256 × 448 on a test-dev release, with the music encoder read
  in eight spans as the released eight-way sequence parallelism reads it.

Every oracle runs under the `wandancer` env (librosa 0.10.2, transformers 4.46.2, torch 2.6, peft 0.21.2,
opencv-python-headless) with `IK_WAN_DANCER_SRC` set.
"""

import importlib.util
import json
import os
import sys
import tempfile
import uuid

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def fork():
    """The fork on `sys.path` behind the CPU shims."""
    source = os.environ["IK_WAN_DANCER_SRC"]
    if source not in sys.path:
        sys.path.insert(0, source)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import dancer_shims
    return source, dancer_shims


def script(name):
    source, _ = fork()
    spec = importlib.util.spec_from_file_location(name, os.path.join(source, "gen_video", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(arrays):
    out = {}
    for key, value in arrays.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(value))
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        elif tensor.dtype == torch.int64:
            tensor = tensor.int()
        out[key] = tensor.contiguous()
    return out


# MARK: - Music features

def synthesized_clip(seconds=20.0, rate=15360):
    """A deterministic beat-and-chord clip at `rate`."""
    rng = np.random.default_rng(20261126)
    t = np.arange(int(seconds * rate)) / rate
    y = np.zeros_like(t)
    beat = 60 / 124.0
    for k in range(int(seconds / beat)):
        start = k * beat + rng.normal(0, 0.004)
        envelope = np.exp(-np.clip(t - start, 0, None) * 18) * (t >= start)
        y += 0.6 * envelope * np.sin(2 * np.pi * (55 + 40 * np.exp(-np.clip(t - start, 0, None) * 30)) * (t - start))
        if k % 2 == 1:
            y += 0.25 * envelope * rng.normal(0, 1, len(t)) * (t < start + 0.12)
    chords = [(261.63, 329.63, 392.0), (220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66)]
    for k in range(int(seconds / (4 * beat)) + 1):
        gate = ((t >= k * 4 * beat) & (t < (k + 1) * 4 * beat)).astype(float)
        for f in chords[k % 4]:
            y += 0.08 * gate * np.sin(2 * np.pi * f * 1.003 * t)
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32)


def music():
    import librosa
    import soundfile as sf
    assert librosa.__version__ == "0.10.2.post1", librosa.__version__
    y = synthesized_clip()
    path = os.path.join(tempfile.gettempdir(), f"dancer-music-{uuid.uuid4().hex}.wav")
    sf.write(path, y, 15360, subtype="FLOAT")
    hop, sr = 512, 22050

    def seams(cqt_res, load_res):
        data, _ = librosa.load(path, sr=30 * hop, res_type=load_res)
        env = librosa.onset.onset_strength(y=data, sr=sr)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20).T
        tuning = librosa.estimate_tuning(y=data, sr=sr, bins_per_octave=36)
        C = np.abs(librosa.cqt(data, sr=sr, hop_length=hop, n_bins=252, bins_per_octave=36, tuning=tuning, res_type=cqt_res))
        chroma = librosa.feature.chroma_cens(C=C, sr=sr, hop_length=hop, n_chroma=12).T
        full = librosa.feature.chroma_cens(y=data, sr=sr, hop_length=hop, n_chroma=12).T if cqt_res == "soxr_hq" else chroma
        peaks = librosa.onset.onset_detect(onset_envelope=env.flatten(), sr=sr, hop_length=hop)
        tempo_data = librosa.load(path, res_type=load_res)[0]
        start_bpm = librosa.beat.tempo(y=tempo_data)[0]
        bpm, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, start_bpm=start_bpm, tightness=100)
        onsets = np.zeros_like(env, dtype=np.float32)
        onsets[peaks] = 1
        beat_marks = np.zeros_like(env, dtype=np.float32)
        beat_marks[beats] = 1
        features = np.concatenate([env[:, None], mfcc, full, onsets[:, None], beat_marks[:, None]], axis=-1)
        return dict(data=data, mel_db=librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr)).T, envelope=env,
                    mfcc=mfcc, tuning=np.array([tuning]), cqt=C, chroma=chroma, peaks=peaks, start_bpm=np.array([start_bpm]),
                    bpm=np.atleast_1d(np.asarray(bpm, dtype=np.float64)), beats=beats, tempo_data=tempo_data,
                    features=features.astype(np.float32))

    arrays = {"samples": y}
    for name, cqt_res, load_res in (("poly", "polyphase", "polyphase"), ("ref", "soxr_hq", "soxr_hq")):
        arrays.update({f"{name}.{key}": value for key, value in seams(cqt_res, load_res).items()})
    os.remove(path)
    return _record(arrays)


# MARK: - Planning

def planning():
    import cv2
    import soundfile as sf
    from moviepy import AudioFileClip
    local = script("gen_video_local")

    class Capture:
        def __init__(self, count):
            self.left = count

        def read(self):
            if self.left == 0:
                return False, None
            self.left -= 1
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            pass

    arrays = {}
    for total in (149, 300, 447, 600, 1000, 5400):
        cv2.VideoCapture = lambda _path, count=38: Capture(count)
        _, masks = local.process_global_video_firstlastframe("unused", 16, 16, total)
        arrays[f"masks.{total}"] = np.stack(masks).astype(np.int32)
    directory = tempfile.mkdtemp(prefix=f"dancer-planning-{uuid.uuid4().hex}-")
    rng = np.random.default_rng(3)
    for seconds, rate in ((20.0, 44100), (12.34, 48000), (5.1, 22050)):
        samples = (0.5 * rng.standard_normal(int(seconds * rate))).clip(-1, 1).astype(np.float32)
        track = os.path.join(directory, f"track-{rate}.wav")
        sf.write(track, samples, rate, subtype="FLOAT")
        clips = os.path.join(directory, f"clips-{rate}")
        os.makedirs(clips)
        local.get_music_clip_149f(track, clips)
        names = sorted(name for name in os.listdir(clips) if name.endswith(".wav"))
        key = f"music.{rate}"
        arrays[f"{key}.samples"] = samples
        arrays[f"{key}.duration"] = np.array([AudioFileClip(track).duration], dtype=np.float32)
        arrays[f"{key}.lengths"] = np.array([sf.info(os.path.join(clips, n)).frames for n in names], dtype=np.int32)
        arrays[f"{key}.last"] = sf.read(os.path.join(clips, names[-1]), dtype="float32")[0]
    return _record(arrays)


# MARK: - Pipeline

HEIGHT, WIDTH, FRAMES, STEPS = 256, 448, 17, 3
PROMPT = "A dancer in a red dress turns on a bright stage, full body, steady camera."
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def synthesized_music(seconds, rate=22050, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    y = np.zeros_like(t)
    beat = 60 / 118.0
    for k in range(int(seconds / beat)):
        start = k * beat + rng.normal(0, 0.003)
        envelope = np.exp(-np.clip(t - start, 0, None) * 16) * (t >= start)
        y += 0.6 * envelope * np.sin(2 * np.pi * 60 * (t - start))
    for k, f in enumerate((261.63, 293.66, 329.63, 392.0)):
        y += 0.06 * np.sin(2 * np.pi * f * t) * (((t / (4 * beat)).astype(int) % 4) == k)
    return (0.8 * y / np.abs(y).max()).astype(np.float32)


def music_features(script, seconds):
    import soundfile as sf
    path = os.path.join(tempfile.gettempdir(), f"dancer-oracle-{uuid.uuid4().hex}.wav")
    feature_path = path[:-4] + ".npy"
    sf.write(path, synthesized_music(seconds), 22050, subtype="FLOAT")
    script.get_music_base_feature(path, feature_path, fps=30)
    features = np.load(feature_path)
    os.remove(path)
    os.remove(feature_path)
    return features.astype(np.float32)


def reference_image():
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    image = np.stack([(x * 255 // WIDTH), (y * 255 // HEIGHT), ((x + y) * 3) % 256], axis=-1).astype(np.uint8)
    image[HEIGHT // 4: 3 * HEIGHT // 4, WIDTH // 3: 2 * WIDTH // 3] = (220, 40, 60)
    return image


def build_pipeline(checkpoint, stage):
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.models.wan_video_image_encoder import WanImageEncoder, clip_xlm_roberta_vit_h_14
    from diffsynth.models.wan_video_text_encoder import WanTextEncoder
    from diffsynth.models.wan_video_vae import WanVideoVAE
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline
    from safetensors.torch import load_file
    with open(os.path.join(checkpoint, "testdev-manifest.json")) as handle:
        geometry = json.load(handle)["geometry"]
    with open(os.path.join(checkpoint, "config.json")) as handle:
        config = json.load(handle)
    import make_testdev
    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32,
                            tokenizer_path=os.path.join(checkpoint, "google/umt5-xxl"))
    # The released scripts run sequence parallel, and `model_fn_wan_video` reads the world size only on that path.
    pipe.use_unified_sequence_parallel = True
    dit = WanModel(**make_testdev.transformer_arguments(config)).eval()
    dit.load_state_dict(load_file(os.path.join(checkpoint, f"{stage}_model.safetensors")), assign=True, strict=False)
    pipe.dit = dit.to(torch.float32)
    text_encoder = WanTextEncoder(**geometry["text_encoder"]).eval()
    text_encoder.load_state_dict(torch.load(os.path.join(checkpoint, "models_t5_umt5-xxl-enc-bf16.pth")))
    pipe.text_encoder = text_encoder.to(torch.float32)
    pipe.prompter.fetch_models(pipe.text_encoder)
    vae = WanVideoVAE(z_dim=16).eval()
    vae.model.load_state_dict(torch.load(os.path.join(checkpoint, "Wan2.1_VAE.pth")))
    pipe.vae = vae.to(torch.float32)
    image_encoder = WanImageEncoder.__new__(WanImageEncoder)
    torch.nn.Module.__init__(image_encoder)
    image_encoder.model, image_encoder.transforms = clip_xlm_roberta_vit_h_14(
        pretrained=False, return_transforms=True, return_tokenizer=False, dtype=torch.float32, device="cpu",
        **geometry["clip"])
    state = torch.load(os.path.join(checkpoint, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
    image_encoder.model.load_state_dict(state)
    pipe.image_encoder = image_encoder.eval().to(torch.float32)
    return pipe


def run(pipe, shims, record, prefix, call):
    shims.music_spans(pipe.dit, 8)
    captured = {"steps": [], "calls": []}
    generate_noise = pipe.generate_noise

    def noise(*arguments, **keywords):
        value = generate_noise(*arguments, **keywords)
        captured["noise"] = value.clone()
        return value
    pipe.generate_noise = noise

    step = pipe.scheduler.step

    def stepped(model_output, timestep, sample, **keywords):
        out = step(model_output, timestep, sample, **keywords)
        captured["steps"].append(out.clone())
        return out
    pipe.scheduler.step = stepped

    model_fn = pipe.model_fn

    def traced(**keywords):
        out = model_fn(**keywords)
        if len(captured["calls"]) < 2:
            captured["calls"].append(out.clone())
            if len(captured["calls"]) == 1:
                captured["context"] = keywords["context"].clone()
                captured["clip"] = keywords["clip_feature"].clone()
                captured["refimage"] = keywords["refimage_feature"].clone()
                captured["y"] = keywords["y"].clone()
                captured["music_tokens"] = pipe.dit.merged_audio_emb.clone()
        return out
    pipe.model_fn = traced

    decode = pipe.vae.decode
    decoded = []

    def decoding(hidden_states, **keywords):
        out = decode(hidden_states, **keywords)
        decoded.append(out.clone())
        return out
    pipe.vae.decode = decoding

    encoder = pipe.dit.music_encoder

    def encoder_hook(_module, _input, output):
        captured["music_encoded"] = output.clone()
    handle = encoder.register_forward_hook(encoder_hook)
    with torch.no_grad():
        frames = pipe(**call)
    handle.remove()
    record[f"{prefix}.noise"] = captured["noise"][0]
    record[f"{prefix}.steps"] = torch.stack([s[0] for s in captured["steps"]])
    record[f"{prefix}.call0"] = captured["calls"][0][0]
    record[f"{prefix}.call1"] = captured["calls"][1][0]
    record[f"{prefix}.context"] = captured["context"][0]
    record[f"{prefix}.clip_feature"] = captured["clip"][0]
    record[f"{prefix}.refimage_feature"] = captured["refimage"][0]
    record[f"{prefix}.y"] = captured["y"][0]
    record[f"{prefix}.music_tokens"] = captured["music_tokens"][0]
    record[f"{prefix}.music_encoded"] = captured["music_encoded"].reshape(-1, captured["music_encoded"].shape[-1])
    record[f"{prefix}.decoded"] = torch.cat(decoded, dim=2)[0]
    record[f"{prefix}.video"] = torch.from_numpy(np.stack([np.array(f) for f in frames]))
    record[f"{prefix}.timesteps"] = pipe.scheduler.timesteps.clone()


def pipeline(checkpoint):
    _, shims = fork()
    global_script = script("gen_video_global")
    from PIL import Image
    image = Image.fromarray(reference_image())
    record = {"reference_image": torch.from_numpy(reference_image())}

    global_music = music_features(global_script, 20.0)
    fps = 30.0 / int(global_music.shape[0] / 149.0 + 0.5)
    fps = float("{:.4f}".format(fps))
    global_prompt = PROMPT + "帧率是{:.4f}".format(fps)
    mask = np.zeros(FRAMES, dtype=np.int32)
    mask[0] = 1
    keyframes = np.zeros((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    keyframes[0] = reference_image()
    record["global.music"] = torch.from_numpy(global_music)
    record["global.keyframes"] = torch.from_numpy(keyframes)
    record["global.mask"] = torch.from_numpy(mask)
    pipe = build_pipeline(checkpoint, "global")
    run(pipe, shims, record, "global", dict(
        prompt=global_prompt, negative_prompt=NEGATIVE, input_image=None, num_inference_steps=STEPS, seed=0, tiled=True,
        height=HEIGHT, width=WIDTH, enable_music_inject=True, music_feature=torch.from_numpy(global_music),
        num_frames=FRAMES, interp_mode="bilinear", enable_refimage=True, refimage=image, enable_global=True,
        keyframes=[Image.fromarray(k) for k in keyframes], keyframes_mask=torch.tensor(mask).to(torch.int32),
        enable_dynamicfps=True, input_fps=fps, enable_vae_decode_framewise=True, enable_skip_layer=True,
        enable_unimodel=True, sigma_shift=5, cfg_scale=5))

    local_music = music_features(global_script, 149 / 30.0)
    local_prompt = PROMPT + ", 帧率是30fps。"
    mask = np.zeros(FRAMES, dtype=np.int32)
    mask[[0, 4, 9, 13, FRAMES - 1]] = 1
    rng = np.random.default_rng(11)
    keyframes = np.zeros((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    for index in np.flatnonzero(mask):
        keyframes[index] = np.clip(reference_image().astype(int) + rng.integers(-40, 40, size=3), 0, 255)
    record["local.music"] = torch.from_numpy(local_music)
    record["local.keyframes"] = torch.from_numpy(keyframes)
    record["local.mask"] = torch.from_numpy(mask)
    pipe = build_pipeline(checkpoint, "local")
    run(pipe, shims, record, "local", dict(
        prompt=local_prompt, negative_prompt=NEGATIVE, num_inference_steps=STEPS, seed=10, tiled=True,
        height=HEIGHT, width=WIDTH, enable_music_inject=True, music_feature=torch.from_numpy(local_music),
        num_frames=FRAMES, interp_mode="bilinear", enable_refimage=True, refimage=image,
        keyframes=[Image.fromarray(k) for k in keyframes], keyframes_mask=torch.tensor(mask).to(torch.int32),
        enable_dynamicfps=True, input_fps=30, enable_skip_layer=True, sigma_shift=5, cfg_scale=5))

    record["global.fps"] = torch.tensor([fps])
    return _record(record)
