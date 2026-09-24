<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: music transcription and structure

Two capabilities the toolkit had no path for, scoped together by a 2026-09-21 survey because they
share a value-type foundation and both read a whole song rather than a clip:

- **Music transcription** (audio → MIDI): a recording in, timed notes out.
- **Music structure** (audio → sections): beats, downbeats, tempo, and the functional sections
  (intro, verse, chorus, bridge, outro) a track divides into.

## The survey

| Model | License | Gated | Oracle | Role here |
| --- | --- | --- | --- | --- |
| Basic Pitch (`spotify/basic-pitch`) | Apache-2.0, code and weights | no | `onnxruntime` over the repo's own 230 KB `nmp.onnx` | The permissive default. Instrument-agnostic polyphonic notes with pitch bends, under 17k parameters. |
| hFT-Transformer (`sony/hFT-Transformer`) | MIT, code and weights | no | the reference package, whose model classes the released pickle needs | The accuracy counterpart to the agnostic default: piano only, two-level frequency-then-time attention, 5.5M parameters. |
| MuScriptor (`MuScriptor/muscriptor-{small,medium,large}`) | code MIT, **weights CC BY-NC 4.0** | **yes**, auto-approve | the `muscriptor` package | The 2026 multi-instrument flagship: mel in, MT3-style tokens out, one MIDI track per instrument. Unblocked 2026-09-21 when the license was accepted and a token provided. |
| All-In-One (`taejunkim/allinone`) | MIT, code and weights | no | the `allin1` package (needs `natten` and `madmom`) | The structure model. Beats, downbeats, tempo, section boundaries, and section labels from demixed audio, which the shipped htdemucs already produces. |

Skipped: MT3 and YourMT3 (t5x checkpoints, no maintained PyTorch release), the `llama-midi` family
(symbolic continuation, not transcription), Pop2Piano (piano arrangement of a mix rather than
transcription of what is played).

One correction to this table, recorded because the error is instructive. The hFT-Transformer row
first named `xavriley/midi-transcription-models`, on the strength of that repository's tags and a
checkpoint filename reading `note_F1=0.9677`. Opening its code shows it is not hFT-Transformer at
all: it imports `piano_transcription_inference` and builds
`Regress_onset_offset_frame_velocity_CRNN`, which is ByteDance's high-resolution piano transcriber,
fine-tuned by its author to guitar, bass, and saxophone. The F1 in the filename is ByteDance's. An
architecture is what the code builds, not what the tags say, and the same check that caught
MuScriptor's attribution catches this: read the source before naming the model. The ByteDance family
remains a reasonable future port, under its own name.

What is verified about MuScriptor and what is not: the three repositories sit under a standalone
`MuScriptor` organization created 2026-06-30, carry `cc-by-nc-4.0`, and hold a `config.json` and a
`model.safetensors` each; the paper is arXiv 2607.08168; the medium release loads and runs here. The
"Kyutai release" framing comes from secondary press rather than from the vendor: the `kyutai`
organization's models do not include it, and `mirelo-ai` has none. Nothing in the released files
names a vendor either, so the attribution stays unconfirmed while the model itself is not in doubt.

MuScriptor's gating is `auto`, so accepting the license on the model page and authenticating unblocks
it with no waiting; `NFKHFHub.accessToken` / `HF_TOKEN` read the same credential the gated Stable
Diffusion 2.1 base uses, and `hf auth login` writes one to `~/.cache/huggingface/token`. Accepting the
license is per ACCOUNT: the machine still needs a token afterwards, which is the step that looks like
the gate not having opened.

## Core value types

Both capabilities return timed structure rather than a tensor, so the types are core, not MLX:

- `NFKMIDINote` — one note: pitch, start and end in seconds, velocity, an optional pitch-bend curve.
- `NFKMIDISequence` — the notes of one performance plus the tempo, with `standardMIDIFileData`
  writing a type-1 Standard MIDI File a DAW opens. Under the new `NFKOutputMIDI` key.
- `NFKMusicBeat` — one beat: its time, its position in the bar (1 is a downbeat). Under
  `NFKOutputBeats`, with the estimated tempo under `NFKOutputTempo`.

Sections need no new type: a functional section is a labeled span, which is what `NFKAudioSegment`
already carries under `NFKOutputSegments`.

## The models

- `NFKMLXBasicPitch` (`@objc`) — Basic Pitch (`spotify/basic-pitch`, ICASSP 2022), the
  instrument-agnostic note transcriber and the toolkit's first audio → MIDI path. A nine-octave
  constant-Q front end, a normalized log, harmonic stacking, and three small convolutional heads that
  score, per frame, a pitch contour (264 bins, three per semitone), a note activation (88 bins), and an
  onset (88 bins). 35,736 parameters, the smallest network in the package. `transcribe(_:sampleRate:)`
  returns an `NFKMIDISequence`; the backend puts it under `NFKOutputMIDI`; `+register` under
  `basic-pitch`.
  - **The released artifact is a graph, not a state dict.** `nmp.onnx` (230 KB, Apache-2.0) holds the
    whole pipeline, so the CQT's two quadrature kernels, the anti-aliasing filter that carries one
    octave to the next, and the per-bin scale arrive as checkpoint tensors rather than as constants the
    port derives from nnAudio's kernel construction. `Tools/basic-pitch-to-safetensors` lifts twenty
    tensors out of the graph's initializers, matching each convolution by the shape that is unique to
    it and the folded normalization by which node reads it (a `Mul` feeding an `Add`, both of one
    element, indistinguishable by name because the export fuses whole name chains into one
    identifier).
  - **The CQT is nnAudio's `CQT2010v2`.** Rather than one transform at a huge window, it computes one
    octave at a time with the same 36 kernels, halving the signal rate between octaves and halving the
    hop with it, so all nine octaves land on the same 172-frame grid. The stack is 324 bins and the
    lowest 15, which sit below the requested 27.5 Hz, fall off the bottom. Each octave reflect-pads by
    128 before its transform; the anti-aliasing filter pads with zeros instead.
  - The Keras batch normalizations are already folded into the convolutions they precede, so the
    module has none. The one that survives is the single-channel normalization after the log, kept as
    a scale and a bias.
  - Which kernel is the real part does not reach any output: the graph negates one of the two and the
    magnitude squares both, so they load as `a` and `b`.
  - **Note creation** (`NFKBasicPitchNoteCreation`) is the reference's `note_creation.py`: onset
    peak-picking, onsets inferred from a sharp rise in the note activation, a forward walk that ends a
    note after eleven frames below the threshold, the melodia pass that picks up what the onset head
    missed, and a pitch-bend curve read from the contour under a Gaussian window. It is sequential
    array work, so it ports as plain arrays rather than as tensors. The onsets are walked backwards in
    time, which is how an earlier note claims the energy a later one would otherwise take.
  - At reference parity against the released graph through onnxruntime, on the first numeric run:
    CQT 1.0, normalized log 1.0, harmonic stack 1.0, contour 0.9999999, note 0.9999998, onset 1.0,
    the stitched posteriorgram 0.9999999, all ten notes of the reference clip at the same pitch,
    start, and end (to 1e-6 s), and every pitch-bend curve identical bin for bin. Oracle
    `run_reference.py basic_pitch` under `bpvenv`.
  - Two frame-rate details the reference carries and this port reproduces: the trim after stitching
    counts frames at the integer rate `sampleRate / hop` (86, not 86.13), and `frameTimes` subtracts
    one window offset per 172 frames plus the reference's own 0.0018 second alignment constant.
  - Customization: trainable at `full`, with no recipe written yet. The reference publishes its
    training code: the installed `basic_pitch` distribution holds `train.py`, `models.py` with `loss()`,
    `transcription_loss`, and `weighted_transcription_loss`, and a data module. At 35,736 parameters it
    is the cheapest full fine-tune in the package. The released graph carries no optimizer state or
    training configuration, so a recipe takes both from `train.py`.

- `NFKMLXAllInOne` (`@objc`) — All-In-One music structure analysis (`mir-aidj/all-in-one`, Kim and
  Nam, ISMIR 2023, MIT), the model that divides a track into what a listener hears as its parts. It
  reads the four HT Demucs stems, so the drums carry the meter and the vocals carry the form rather
  than both competing in one mixture. A convolutional embedding collapses 81 filterbank bands to one
  embedding per frame, then eleven blocks each run a dilated neighborhood attention across time
  followed by a neighborhood attention across the four instruments, and four linear heads score a
  beat, a downbeat, a section boundary, and ten functional labels per frame. 383,577 parameters.
  `analyze(stems:barTracker:)` returns the sections, beats, and tempo; the backend puts them under
  `NFKOutputSegments`, `NFKOutputBeats`, and `NFKOutputTempo`; `+register` under `allin1`.
  - **Neighborhood attention** is NATTEN's (`natten1dqkrpb` / `natten2dqkrpb`). The neighbor and
    relative-position-bias indices depend only on the length, the kernel, and the dilation, so they
    are index tables built once per shape and applied with `take`, one gather per kernel offset. A
    three-minute track is 18,000 frames, whose full attention matrix does not fit; the tables cost
    the size of the input instead. Both index rules are NATTEN 0.14.6's `get_window_start` and
    `get_pb_start`: the window slides inward at the edges rather than being masked, and the bias is
    re-indexed by the same shift.
  - **The instrument attention is two-dimensional and always padded.** It runs at kernel 5 over four
    instruments, so every call pads the instrument axis to five and crops afterward. The padded path
    is the only path the released model ever takes.
  - **The doubled time attention is not a residual stack.** Each time layer runs two attentions, at
    its dilation and at twice it; their outputs are concatenated for the feed-forward, which
    therefore reads twice the width, while the residual takes their mean. That asymmetry is why
    `layernorm_after` is `2·dim` wide, and it is load-bearing.
  - **The stem order is not HT Demucs's.** The separator emits drums, bass, other, vocals; the model
    was trained on the stems in the order the reference reads its files, which is alphabetical: bass,
    drums, other, vocals. Swapping the first two is silent: the model still runs and still produces a
    plausible structure. `NFKMLXAllInOneBackend.separate` fixes the order explicitly.
  - **The front end is madmom's filtered logarithmic spectrogram**, not a mel: frames of 2048 at 100
    per second (hop 441), `np.hanning` (the symmetric window), the bins below Nyquist only (madmom
    drops the Nyquist bin), a logarithmic filterbank of 81 bands at 12 per octave from 30 Hz to
    17 kHz with each filter normalized, then `log10(1 + x)`. The filterbank is a constant of the
    configuration, so `Tools/allin1-to-safetensors` takes it from madmom itself and ships it in the
    checkpoint as `frontend.filterbank`; the consumer needs no madmom. The frame count is
    `ceil(samples / hop)`, and each frame is centered on its own hop.
  - **The checkpoint tunes the post-processing threshold, and it is load-bearing.** The beat decoder
    trims the track to the span that reaches `best_threshold_downbeat` before decoding, so the value
    decides where the beats stop; the released folds disagree (0.21 against 0.22), which is why it
    travels in the converted file as `postprocess.thresholds` rather than sitting as a default in
    code. Ignoring it made this port find one beat more than the reference at the end of the validation
    clip, with every other beat already matching. A tail bug hides that way.
  - **`NFKMLXBarTracker`** is madmom's `DBNDownBeatTrackingProcessor`: a bar-pointer hidden Markov
    model whose state is a position in the bar together with a tempo, decoded by Viterbi. The tempo
    grid is whole frame intervals spaced geometrically between 55 and 215 BPM, widened until sixty
    distinct ones fall out (60 intervals, 3719 states per beat, matching madmom's own state space
    exactly). Three- and four-beat bars are both decoded and the higher path probability wins, which
    is how the meter is inferred. Inside a beat the pointer only advances, so the decode takes a fast
    path for every state that is not the first of its beat, which is most of the space.
  - At reference parity against `mir-aidj/all-in-one` on the released harmonix-fold0 weights, on the
    first numeric run: spectrogram 1.0, embedding 1.0, blocks 0 / 5 / 10 at 1.0 / 1.0 / 0.9999998,
    and the four heads at 1.0 (beat), 1.0 (downbeat), 0.99999994 (section), 1.0 (function). Through
    the post-processing: the same sections with the same labels and boundaries, and the same 24 beats
    at the same frames and the same positions in the bar.
  - **The oracle substitutes the attention, and says so.** NATTEN dropped the biased 1D and 2D
    kernels after 0.14, so the released model's own dependency no longer installs; the oracle
    transcribes the index rules from NATTEN 0.14.6's `natten_cpu_commons.h` into torch and checks
    that stand-in against the installed NATTEN's kernel (through its flex backend, at a power-of-two
    head dimension, at dilations 1, 2, and 4 and in two dimensions) before recording anything. The
    neighbor rule is therefore verified by the library itself; the bias index rule is transcribed on
    both sides from the same header, which is the one link in the chain that no third party checks.
  - **Customization is a FULL fine-tune and it ships** (`NFKMLXAllInOneTraining.swift`), ported from
    the authors' `training/` package (mir-aidj/all-in-one 18e7890). `NFKMLXAllInOne.network(weightsURL:)`
    builds the net, `NFKMLXAllInOneTargets` turns an annotation (beat and downbeat times, section
    boundaries, and one more label than boundaries) into frame targets the way `DatasetBase` does
    (`librosa.time_to_frames`, then `widen_temporal_events`: neighbors at 0.5, the section's second ring
    at 0.25; each frame labeled by the last boundary at or before it), and `fineTune(_:examples:…)`
    trains every weight but the spectrogram front end and the post-processing thresholds.
    `NFKMLXAllInOneObjective` is `compute_losses`: masked BCE with logits on beat, downbeat, and
    section and masked cross-entropy on function, weighted 1, 3, 15, and 0.1. The optimizer is timm
    0.9's `RAdam` (`NFKMLXRAdam`, rate 0.005, weight decay 0.00025 outside biases and one-dimensional
    parameters); the reference's plateau schedule needs a validation set, so the recipe holds the rate.
    Measured against the authors' sources (`run_reference.py allin1_training`, the oracle running
    `compute_losses` from `trainer.py`'s own text; `testAllInOneTrainingMatchesTheReference`): every
    target frame equal, the loss 20.67702 vs 20.67702 with each term equal, RAdam within 3e-8 over
    twelve steps across its rectification threshold. The network gained the reference's training
    dropouts (0.2 on the convolution stages, the attention probabilities and output, and the
    feed-forward; stochastic depth rising to 0.1) and defaults to evaluation mode, and the loader
    transposes a 4-D tensor only for a PyTorch-layout checkpoint, so a saved run reloads. The Harmonix
    audio stays unavailable; a consumer's own annotated tracks replace it. Annotation times in a
    float32 record move frames (1.23 s is frame 122 in float64 and 123 in float32), so the oracle uses
    float32-exact times.

- `NFKMLXMuScriptor` (`@objc`) — MuScriptor (`muscriptor/muscriptor`, 2026), multi-instrument
  transcription as language modeling: five seconds of mel spectrogram are projected into the
  transformer's width and prepended to the sequence, and a causal decoder writes an MT3 event stream
  that becomes notes. One MIDI track per instrument falls out of the program events rather than out of
  a separate model. Three released sizes (`small` 103M, `medium` 307M, `large` 1.4B); the geometry
  ships in the MIT inference package, so the module is built and measured without the gated weights.
  `+register` under `muscriptor` / `muscriptor-small` / `muscriptor-large`.
  - The architecture is audiocraft's language model at one codebook: a pre-norm causal transformer,
    bias-free projections, a fused `in_proj_weight`, a GELU feed-forward, sinusoidal absolute
    positions added at the transformer's entry (cosine half first, and the exponent divides by
    `halfDimension − 1`), and a LayerNorm before the head.
  - **The conditioning order is the reverse of the declaration order.** Three conditioners run — the
    mel, an instrument class, a dataset class — and the reference PREPENDS each in turn, so the last
    one processed lands first. The sequence is mel, dataset, instrument, then the generated tokens.
    Declaring them in the obvious order puts every position in the wrong place.
  - **The last mel frame of every chunk is masked away.** Centered framing yields one frame more than
    the clip fills, and the conditioner's length mask counts `samples / hop`, so that final frame is
    zeroed before the transformer sees it. Keeping it measured 0.9955 at the conditioner while the
    logits still read 0.99998 — the band where a real bug looks like float noise.
  - The mel window and filterbank are checkpoint buffers, so they load rather than being re-derived,
    the way Basic Pitch's constant-Q kernels do. The released files also store the embedding and the
    head as the first entry of a module list (`emb.0.*`, `linears.0.*`), which
    `remapReferenceKey` flattens exactly as the reference's own loader does.
  - The head is wider than the vocabulary. The tokenizer defines 1393 tokens (3 special, 1001 shifts,
    128 pitches, 2 velocities, tie, 130 programs, 128 drums) and the medium and large heads score
    1395; the reference masks from 1393 whatever the head's width, so the extra tokens are never
    sampled.
  - **The tie section is how a note survives a chunk boundary.** Each chunk opens by declaring the
    notes still sounding as program and pitch pairs, terminated by `tie`; anything the previous chunk
    left open that the section does not declare ends at the boundary, and a chunk that reaches a time
    shift without ever emitting `tie` is malformed, so everything open closes and the rest of that
    chunk is dropped. `NFKMuScriptorNoteTracker` is that state machine, and it serves both jobs the
    reference gives it: decoding notes, and reading back what is open so the next chunk's tie section
    can be teacher-forced.
  - Measured twice. At a tiny random configuration against the reference's own `_build_model`, with
    the parameters carried in the record under `w::`: mel 1.0000001, mel conditioner 1.0000005,
    prefill logits 1.0, and the greedy continuation token for token (which also says the KV cache
    carries what the prefill put in it, since the reference recomputes from scratch). On the released
    medium weights: mel 0.9999999, conditioning prefix 1.0000005, prefill logits 1.0000002, all 17
    decoded tokens identical, every decode action identical, and four notes transcribed end to end.
  - The weights are CC BY-NC 4.0, so they ship gated the way Music 3's are, and the model card also
    states that velocity is not produced: every note is written at the reference's own constant 100.
  - Customization: not shipped. The training pipeline (synthetic data, then fine-tuning on real audio,
    then reinforcement learning) is described in the paper but not released, and the weights are
    non-commercial, so a consumer adapting them is outside what the license contemplates.

- `NFKMLXHFTTransformer` (`@objc`) — hFT-Transformer (`sony/hFT-Transformer`, Toyama et al., ISMIR
  2023, MIT), piano transcription by attending in two directions in turn, and the accuracy
  counterpart to the instrument-agnostic Basic Pitch. A convolutional stem reads a 65-frame window
  around each output frame; a transformer attends ACROSS FREQUENCY, turning 256 mel bins into 88 note
  queries through cross-attention; a second transformer attends ACROSS TIME, refining each note's own
  track of 128 frames. Four heads read each level: onset, offset, multi-pitch, and velocity. 5.5M
  parameters. `+register` under `hft-transformer`.
  - **Both levels are outputs.** The frequency level is the model's first answer and the time level
    is its refinement; the reference evaluates the second, which is what `transcribe` reads. The port
    returns both, with the first decoder block's note-to-frequency attention beside them.
  - **A layer holds ONE LayerNorm and applies it after every residual**, rather than one norm per
    residual, so a layer has a single `layer_norm.weight`. The model is post-norm throughout. Both
    are the reference's own shape rather than an omission in it.
  - The note queries are a position embedding with no scaling, while the encoder's tokens and the
    time branch both scale by the square root of the width. Three sites, two conventions.
  - **The released checkpoint is not a `torch.save` archive.** `model_016_003.pkl` is a plain pickle
    of the live module, saved on CUDA with a torch old enough to pickle storages by value. Reading it
    needs the model's classes importable, a patch of `torch.storage._load_from_bytes` onto the CPU
    (the nested load takes no `map_location`), and a correction of the unpickled module's remembered
    device, which it otherwise uses to build its position indices. `Tools/hft-transformer-to-safetensors`
    encapsulates all three and adds the window and mel filterbank the reference builds at runtime
    from torchaudio.
  - Segments are 128 frames with 32 frames of context either side, and the padding is filled with
    `log(1e-8)`, the log-mel value of silence, rather than with zeros.
  - **Note detection** (`NFKHFTNoteDetection`) is the reference's `mpe2note`: onset and offset peaks,
    each peak's time refined by fitting its two neighbors, the note running until the offset head or
    the multi-pitch head ends it (the earlier of the two by default), velocity read at the onset
    frame, and a repeated note truncating the one before it. The peak scan walks outward until a
    different value appears, so a plateau survives where a strict neighbor comparison would drop it.
  - At reference parity on the released MAESTRO weights, on the first numeric run: log-mel 0.9999999,
    encoder 0.9999998, and every head at 1.0 (onset, offset, multi-pitch, velocity, at both levels),
    with the note-to-frequency attention at 1.0000001. Over the whole clip: onset 1.0000001, multi-pitch
    0.99999946, the velocity argmax agreeing on 22528 of 22528 frame-note pairs, and all nine notes
    matching the reference in pitch, onset, offset, and velocity.
  - One trap the parity run found, in the ORACLE rather than the port: the reference reads its audio
    from a 16-bit file, so the samples it sees are quantized. Recording the pre-quantization floats
    made both sides run different audio and read as a front-end divergence of 1e-5, which propagated
    to a one-frame difference in a single note's offset. The record now carries what the reference
    actually read.
  - Customization: not shipped. The training code is released, but it expects the MAESTRO corpus
    prepared through the repository's own pipeline, which is outside what this toolkit builds.

## Build order

1. The core value types and the Standard MIDI File writer. **Done.**
2. Basic Pitch. **Done**, at reference parity.
3. All-In-One, with madmom's front end, neighborhood attention, the structure post-processing, and
   the bar-pointer beat decoder. **Done**, at reference parity.
4. hFT-Transformer beside Basic Pitch, for piano accuracy. **Done**, at reference parity.
5. MuScriptor. **Done**, at reference parity, once the license was accepted and a token supplied.

The bucket is closed. A future addition worth its own entry is ByteDance's high-resolution piano
transcriber, which `xavriley/midi-transcription-models` carries fine-tuned to guitar, bass, and
saxophone under MIT.
