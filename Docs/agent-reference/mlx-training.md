<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX on-device training and fine-tuning

- `NFKMLXTrainer` — the supervised and zero-reference training loop, for customizing a shipped model
  on a consumer's own data, in the app. Two entry points (`batch:loss:` with a target,
  `sample:loss:` without one) share a private loop: gradient clipping, per-step progress and early
  stop, and periodic checkpoints. The toolkit owns the loop, the caller owns the loss, mirroring
  `NFKMLXModuleBackend`. Which parameters train is set by **freezing** the rest before calling in —
  `valueAndGrad` differentiates only `trainableParameters()`, so a frozen backbone costs neither
  gradients nor optimizer state, which is what makes on-device fine-tuning viable at all. Adds the
  `MLXOptimizers` product.
  - **Output is input**: `NFKMLXWeights.save` writes a plain safetensors that the model's existing
    `backendWith…weightsURL:` factory loads, so a customized model needs no separate route. The file
    records its layout in metadata (`inferkit.layout`), and `NFKMLXWeights.loadCheckpoint` reports
    `needsConvTranspose` so a `loadWeights` **skips** its PyTorch transpose for a fine-tuned file.
    Skipping rather than inverting is what keeps the models whose transpose is not the common one
    correct (SAM's `up1`/`up2` use `transposed(1,2,3,0)`, Whisper handles 3-D Conv1d). Every model
    reads through `loadCheckpoint`, and every transpose in a loader is gated on the flag — including
    the branches keyed on a name rather than a rank (LaMa's `up.`, Demucs's and Conv-TasNet's 3-D
    transposed convolutions). An ungated loader double-transposes a fine-tuned file and loads silently
    wrong weights: `NFKMLXCheckpointRoundTripTests` saves and reloads through each model's own loader,
    and removing one gate makes it fail with a transposed shape rather than a bad number.
  - Writes are atomic (scratch file then replace), because a periodic checkpoint overwrites the only
    copy of a run's progress. A non-finite loss throws `NFKMLXError.trainingDiverged` **before** that
    step can reach a checkpoint, so divergence cannot replace good weights with ruined ones.
  - `clipGradientNorm` runs through `bounded(_:maxNorm:)`, which sanitizes before it clips. A
    gradient set can be entirely finite while the sum of its squares is not — 3e20 is an ordinary
    `Float` and its square is not — so a directly computed norm comes back infinite, `maxNorm/∞` is
    zero, and the whole update is scaled to nothing; the optimizer then builds its moments from zeros
    and later steps produce non-finite parameters while the loss stays finite throughout, so the
    divergence guard never fires. Non-finite entries are zeroed first, then the norm is taken relative
    to the largest magnitude present. Reported by RVC-MLX from a real run, pinned here by
    `testAGradientWhoseSquaresOverflowIsStillScaledToTheNorm`.
  - Optimizer state is **not** checkpointed: mlx-swift keeps `stateStorage` internal and `innerState()`
    unkeyed. `SGD` resumes exactly; `Adam` rebuilds its moment estimates.
- `NFKMLXTrainingData` / `NFKMLXBatchSampler` — the app-data side of training. `tensor` / `batch` /
  `matte` / `labels` convert a consumer's `CGImage`s into what the trainer takes, reusing
  `NFKMLXImageBridge` so a training batch and an inference input are built identically. `labels`
  inverts the label-map convention the segmentation backends emit (`index / (classCount − 1)`), so
  a mask painted in an app and a mask a model outputs are the same encoding. A mixed-size batch throws
  rather than resizing: crop versus scale changes what the model learns, so the choice stays with the
  caller. `NFKMLXBatchSampler` draws reshuffled passes from a seed (SplitMix64 Fisher-Yates, matching
  the schedulers' repeatable-randomness idiom) — cycling a handful of examples in a fixed order lets
  the optimizer chase the sequence rather than the data.
- `NFKMLXCLIPProbe` / `NFKMLXCLIPProbeBackend` — a consumer's own image classifier over a frozen CLIP
  embedding, and the cheapest useful customization in the package. Because both towers stay frozen,
  `NFKMLXCLIP.embeddings(for:using:)` computes each embedding **once** and `trainProbe` then runs over
  cached vectors: a step is one 512-wide matrix multiply rather than a transformer forward. A contrastive
  fine-tune of CLIP itself is not a device workload (it needs large batches for negatives); a probe is.
  The backend emits `NSArray<NFKClassification *>` under the core key `NFKOutputClassifications`,
  softmaxed and ranked. A probe is a **separate small model**, so what it saves is a companion file
  rather than modified CLIP weights.
- `NFKMLXWhisperTraining` — domain adaptation for speech (jargon, accents, recording conditions), and
  the recipe LoRA exists for. `NFKMLXWhisperObjective` is teacher forcing: the decoder sees the whole
  target sequence at once and each position is scored on the next token, so a step is one forward pass
  rather than one per token. `loss(logits:tokens:)` is separable from the forward for the oracle, as in
  the other two objectives. Only `decoder.blocks.*.{query,value}` are adapted — the reference LoRA
  choice, and the encoder's audio features transfer across domains. `NFKMLXWhisper.spectrogram` pads or
  trims to the 30-second window, the single biggest accuracy factor in reaching reference parity. **`NFKWhisperAttention.query`/`value`/`out` gained `@ModuleInfo`** so they
  can receive adapters; the wrapper keys equal the property names, so checkpoints are unchanged.
- `NFKMLXLoRA` / `NFKMLXLoRALinear` — low-rank adaptation, for the models with no small head to train
  (CLIP, Whisper: adapting them means reaching into the attention blocks, and doing that fully needs
  optimizer state proportional to the whole model). `NFKMLXLoRALinear` **subclasses `Linear`**, which is
  what `update(modules:)` requires — a replacement must be assignable to the `@ModuleInfo` ivar's type,
  the same idiom `QuantizedLinear` uses. `apply(to:rank:alpha:where:)` does the tree surgery via
  `leafModules().flattened()`, then freezes everything and reopens only `lora_a`/`lora_b`; it returns
  the count, so a predicate that matched nothing is visible rather than silent, and it skips
  already-adapted layers. `apply` and `merge` throw: MLX's non-throwing `update(modules:)` wraps a
  `try!`, so selecting a layer stored in a plain property aborts the process. Both call the throwing
  variant and report `NFKMLXError.loRANotApplicable` naming the `@ModuleInfo` requirement. The `B` factor starts at zero, so an adapted model produces exactly what it
  produced before training. `merge(into:)` folds each detour into its base weights (`Linear` computes
  `x·Wᵀ`, so the delta is `(A·B)ᵀ·scale`) and leaves plain layers: the saved file carries no adapter
  keys, so there is no adapter format and no second loading path.
  Neither `apply` nor `merge` will touch a quantized model. `QuantizedLinear` subclasses `Linear`,
  so it satisfies a type-based predicate silently while its `weight` holds packed integers — adapting
  one builds a detour around something that is not a weight. And merging a low-rank delta into a
  quantized base then requantizing rounds the delta away, so the training is discarded while the file
  loads without complaint. Merge at float precision, then quantize, in that order.
- `NFKMLXSegFormerTraining` — the head-only recipe. A consumer rarely wants ADE20K's 150 classes and
  usually wants their own few, which is a decode-head problem: `NFKMLXSegFormerTrainable.decodeHead`
  (the default) freezes the four encoder stages, so the run's memory falls to the head's share.
  `NFKMLXSegFormer.network(weightsURL:classCount:)` drops the checkpoint's `classifier.*` when the
  class count differs and loads the rest with `strict: false` — not optional, because MLX's
  `update(parameters:)` adopts a checkpoint's shapes wholesale rather than validating them, so keeping
  it would silently restore the old class set. `NFKMLXSegFormerObjective` upsamples the stage-1 logits
  to the label resolution before cross-entropy, as the reference does, rather than downsampling the
  labels. The ImageNet input normalization moved into `NFKMLXSegFormerNet.normalized`, shared by
  `segment` and the objective: a fine-tune that normalized differently would optimize for a
  distribution the model never sees at inference, which is the bug four models here have already
  shipped. Reference parity: `NFKMLXSegFormerObjective.loss(logits:labels:)` is separable from the
  forward pass so the oracle can score identical logits, and matches transformers'
  `SegformerForSemanticSegmentation` loss (`run_reference.py segformer_loss`,
  `IK_PARITY_SEGFORMER_LOSS`).
- `NFKMLXZeroDCETraining` — the first customization recipe, and the template for the rest.
  Zero-DCE is zero-reference: the reference trains it with no ground truth, so a consumer
  customizes it from their own dark photos with nothing to annotate, which is the only kind of data an
  end user has. `NFKMLXZeroDCELoss` ships the four losses (exposure, color constancy, illumination
  smoothness, spatial consistency); `NFKMLXZeroDCEObjective` weights them, and its `wellExposedLevel`
  is the consumer-facing knob (preferred brightness). `NFKMLXZeroDCE.network(weightsURL:)` builds the
  trainable net; `fineTune` runs it. Reference parity: all four losses agree with the reference to
  float precision (`run_reference.py zero_dce_losses`, `IK_PARITY_ZERO_DCE_LOSSES`). A wrong loss is
  invisible in a fine-tune's output, so the oracle is the only check that catches it — the first
  implementation, written from the paper rather than the code, was wrong in all four. Three reference
  expressions that read like slips are reproduced deliberately and marked where they occur.
