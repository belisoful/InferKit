<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX on-device training and fine-tuning

## Customization is part of parity

A model is at parity when its inference matches the reference and its customization path is shipped
end to end, or ruled out in writing. Inference parity alone is half of done. The rule exists because
the alternative was measured: four of roughly a hundred shipped models expose a trainable network,
and the rest capture theirs inside a backend a consumer cannot open.

Every model records one of three outcomes in its `mlx-models-<class>.md` entry:

- On-device trainable → the minimum shipped set below ships with the model.
- Offline-only → the reference recipe needs what a device cannot hold: a discriminator stack (vocoders,
  codecs), a large batch of negatives (contrastive CLIP), or weights past the working set (SDXL, FLUX,
  a language model above 4B at float precision). The entry names the constraint and the offline route
  (train in Python, merge, then convert or load through the ordinary factory).
- Untrainable here → no differentiable objective exists or the reference publishes no training code.
  The entry names it.

Shipping inference and saying nothing about training is the failure this rule stops.

**Choosing the level.** Take the reference's own recipe first, then the cheapest level a device holds
that still answers the consumer's question:

- probe → a small head over frozen features (CLIP, any embedder);
- head retarget → the backbone frozen, the head trained to the consumer's own classes (segmenters,
  detectors);
- zero-reference → a loss over the output's own properties, no labels (Zero-DCE);
- LoRA → detours on the attention projections, everything else frozen (Whisper, language and
  vision-language models at 4B and under, SD 1.5 attention);
- full → every weight, for the small networks (restoration and enhancement CNNs, the speech
  enhancers under a few million parameters).

The entry records the level and the reason.

**The minimum shipped set** when a model is trainable:

1. A public `network(weightsURL:configuration:)` builder returning the `Module`. Nil weights is the
   random initialization, which is training from scratch; a released or fine-tuned file loads through
   `NFKMLXWeights.loadCheckpoint` with the transpose gate. A retargetable head takes its new size here
   and drops the checkpoint's head (the SegFormer pattern), because MLX adopts shapes rather than
   validating them.
2. A freezing policy: a `Trainable` enum or named parameter groups (backbone, head, attention). A
   frozen group's normalization statistics do not move either; a BatchNorm backbone under a head-only
   run stays in evaluation mode, and a test asserts it.
3. An objective ported from the reference's training code, not its paper, with a separable
   `loss(outputs:targets:)` so `run_reference.py <model>_loss` scores identical tensors. A wrong loss
   is invisible in a fine-tune's output; the oracle is the only check that catches it, and the first
   Zero-DCE port was wrong in all four losses.
4. A data adapter for the modality: images through `NFKMLXTrainingData`, audio through the model's own
   front end, text through the release tokenizer and chat template with a loss mask over the prompt.
5. A `fineTune` recipe over `NFKMLXTrainer`, and `NFKMLXLoRA` where LoRA is the level, whose defaults
   (optimizer, learning rate, clipping) are the reference's.
6. The round trip: `NFKMLXWeights.save`, then a reload through the model's own factory that reproduces
   the forward (`NFKMLXCheckpointRoundTripTests`). The fine-tuned file must also load through the
   `@objc` factory, which is the Objective-C reach a closure-taking recipe can have.
7. Tests on the tiny configuration: the loss falls over a few steps; the targeted parameters move and
   the frozen ones do not (`testAFineTuneMovesTheSqueezeExciteAndHardswishBlocks` is the pattern); the
   objective parity test; the round-trip test.
8. Every listing in `mlx-parity-checklist.md` ("Customization is part of parity").

Everything a consumer calls in that set is public. A recipe that compiles only under `@testable import`
is not shipped.

**The public surface.** `NFKMLXWeights`, `NFKMLXQuantization`, and `NFKMLXError` are public, so the
whole path — build, train, `save`, reload through the model's own factory, catch what it throws — is
callable by an app that links the package as an ordinary dependency. `Examples/MLXCustomizationExamples.swift`
holds the customization snippets and imports `InferKitMLX` WITHOUT `@testable`, which is what keeps
that true: a recipe that slips back behind `internal` breaks the examples build rather than being
found by a consumer. Every other file in that target still imports `@testable`, because a gallery
example reaches for internals a consumer does not need.

**The default optimizer is PyTorch's.** mlx-swift's `Adam` and `AdamW` leave out the bias correction
of the moment estimates unless `biasCorrection: true` is passed, and every PyTorch Adam applies it.
Without it each update is `(1 − β1ᵗ) / √(1 − β2ᵗ)` times the reference's: 3.2 at step one with betas
0.9 and 0.999, about 6.5 near step ten, 1.3 at step one thousand, and 16 at step one with β1 0.5. Ten
recipes shipped with the uncorrected default until 2026-09-23. `NFKMLXReferenceOptimizers` builds them
now: `adamW(learningRate:betas:eps:weightDecay:)` is `torch.optim.AdamW`,
`adamW(learningRate:weightDecay:exempting:)` splits the parameters into a decayed and an undecayed
group through `MultiOptimizer`, `adamW(learningRate:betas:over:group:)` gives each parameter its own
rate multiple and weight decay (one optimizer per distinct pair), and `NFKMLXL2Adam` is
`torch.optim.Adam` with its `weight_decay` added to the gradient. `NFKMLXReferenceOptimizersTests`
holds each to PyTorch's first step.

**The schedule is the reference's too.** `NFKMLXTrainer.train(…learningRateSchedule:)` multiplies every
group's base rate by an `NFKMLXLearningRateSchedule` before each step and restores the rates when the
run ends. A `MultiOptimizer`'s groups keep their ratios. The schedules are the references' own formulas,
checked against their code at sample steps:

- `cosine(steps:endScale:)` → fvcore's `CosineParamScheduler`, with step `k` at `k / steps`, as SAM 2's
  trainer passes `where`.
- `poly(steps:power:warmupSteps:warmupRatio:)` → mmcv's `poly` policy and its linear warm-up.
- `inverseSquareRoot(steps:timescale:warmupSteps:cooldownSteps:)` → SAM 3's
  `InverseSquareRootParamScheduler`. It returns 0 at step zero, as the reference does.
- `linearWarmup(steps:)` → cosmos-predict1's `WarmupLambdaLR`, `(k + 1) / warmup`.
- `fairseqInverseSquareRoot(warmupSteps:initialScale:)` → fairseq's `inverse_sqrt`, a linear warm-up then
  `√(warmup / k)`, update `k` at count `k`; equal to the reference at every count (`run_reference.py trocr_loss`).
- `mmengineWarmupCosine(steps:warmupRatio:startFactor:)` → mmengine's `LinearLR` then `CosineAnnealingLR`,
  the warm-up `end − begin − 1` steps long as `LinearParamScheduler` counts it; equal to mmengine's own
  schedulers at every step (`run_reference.py sa2va_loss`).
- `warmupCosine(steps:warmupSteps:startScale:endScale:)` → V-JEPA 2's `WarmupCosineLRSchedule`, which
  steps before each update, so update `k` runs at step `k + 1`; equal to the reference at every step
  (`run_reference.py vjepa2_probe`).

A recipe's nil `learningRateSchedule` is its reference's schedule when the recipe builds the reference
optimizer, and a constant rate when the caller passes an optimizer, since the caller then chose the rate.
An explicit schedule applies to either. `.constant` holds the rate. A warm-up counted in steps runs at its full
length: SegFormer's 1,500 steps and the Cosmos Tokenizer's 5,000 keep a shorter run below the base rate
throughout.

Each recipe's reference, and what it still sets that the recipe's defaults do not:

| Recipe | Reference optimizer | Not reproduced |
| --- | --- | --- |
| Zero-DCE | `lowlight_train.py`: `torch.optim.Adam`, 1e-4, weight decay 1e-4 (added to the gradient), clip 0.1 | — |
| SegFormer | NVlabs `segformer.*.py` (mmseg): AdamW 6e-5 and 6e-4 for the decode head (`head` ×10), betas 0.9 / 0.999, decay 0.01, none on `norm` parameters (the encoder's layer norms), no clip; `poly` (power 1) after a 1,500-step linear warm-up from 1e-6 | — |
| SAM 2 | `sam2.1_hiera_b+_MOSE_finetune.yaml`: AdamW 5e-6, and 3e-6 for the image encoder with the trunk decayed 0.9 per layer (`pos_embed` exempt), decay 0.1, none on `*bias*` or `nn.LayerNorm` (`LayerNorm2d` decays), clip 0.1; cosine to a tenth | — |
| SAM 3 | `odinw_text_only_train.yaml`: AdamW 8e-5 for the transformer, decay 0.1, the same exemptions, clip 0.1; inverse square root (timescale 20) with a 20-step warm-up and a 20-step cool-down | the backbones' rates (2.5e-5 vision, 5e-6 language, layer decay 0.9): the recipe trains from precomputed features, so neither backbone is in it |
| Cosmos Tokenizer | cosmos-predict1 post-training: AdamW 1e-4, betas 0.5 / 0.999, decay 0.01, no clip; a 5,000-step linear warm-up | — |
| Sa2VA | `sa2va_finetune.py` (xtuner/mmengine): AdamW 4e-5, betas 0.9 / 0.999, decay 0.05 on every trained parameter, clip 1; `LinearLR` from 1e-5 over 5% of the run, then `CosineAnnealingLR` to zero; LoRA rank 128, alpha 256, with the embeddings and head whole | LoRA dropout 0.05, the bfloat16 autocast, and the batch of two with 16-step accumulation |
| Florence-2 | none published; the release's `labels=` loss with the translators' defaults: AdamW 1e-4, no decay, clip 1, LoRA rank 8 on the decoder's query and value projections | the rate and the level are this package's |
| Table Transformer | microsoft/table-transformer `structure_config.json` / `detection_config.json`: AdamW 5e-5, 1e-5 for the backbone (its last three stages; DETR freezes the stem and the first stage), decay 1e-4 on every parameter, clip 0.1; `StepLR` 0.9 per epoch | the batch of two (the recipe steps on one image) |
| TrOCR | microsoft/unilm `trocr` (fairseq): `adam` with decoupled decay 1e-4 at 2e-5 (IAM, receipts) or 5e-5 (SROIE), betas 0.9 / 0.999, no clip; `inverse_sqrt` with a 500- (800-) update warm-up from 1e-8; every weight trained | fairseq's Adam places epsilon before the second-moment bias correction; the fp16 flag |
| V-JEPA 2 probe | `evals/video_classification_frozen`: AdamW over the whole `AttentiveClassifier`, decay on every parameter, no clip; `WarmupCosineLRSchedule` stepped before each update, no warm-up, a cosine to zero (`NFKMLXLearningRateSchedule.warmupCosine`). The configurations sweep twenty heads (rates 5e-3, 3e-3, 1e-3, 3e-4, 1e-4 by decays 0.01, 0.1, 0.4, 0.8); the default is the first | the sweep itself (the recipe trains one head; `learningRate:` and `weightDecay:` pick another), and the bfloat16 autocast with its gradient scaler |
| open-jev-deberta | `train_encoder.py`: AdamW 3e-5 for the encoder and 1e-3 for the head, decay 0.01, clip 1; a linear warm-up over the first 6% of the run, then a linear decay to zero (`NFKMLXLearningRateSchedule.openJevDeBERTa(steps:)`) | the encoder's dropout (0.1): a step here is deterministic |
| Open-Jev | `jev/train.py`: AdamW for the adapter and the head (5e-5 and 1e-4 for 2B and 9B, 2e-5 and 5e-5 for 27B, from each release's `provenance.json`), decay 0.01, clip 1, a constant rate, gradient accumulation 4 (`batchSize`) | — |
| Whisper, the translators, TranslateGemma, Granite 4.0-H, Nemotron-H | no script beyond the model's `labels=` loss: transformers' `Trainer` default, AdamW with no decay, clip 1.0 | the rate (1e-4 here, 5e-5 there) is this package's, and so is the constant schedule (the `Trainer` default decays linearly to zero) |
| Qwen3-VL retrieval | sentence-transformers' trainer default: AdamW with no decay, a bias-corrected Adam | the rate (1e-3 here, 5e-5 there) |
| CLIP probe | CLIP's own probe is an L-BFGS logistic regression; AdamW 1e-3 with decay 0.01 is this package's | — |
| Laya | the release publishes no optimizer; a bias-corrected Adam | — |

**Per-model status lives in the ledger.** [mlx-customization-ledger.md](mlx-customization-ledger.md)
carries one row per model entry with its outcome, its level, whether the path is reachable, and the
reference file that decides it. A session picking up a model reads its row first. A session shipping
a recipe updates that row along with the model's entry and the parity checklist's listings. The
counts below are the ledger's, triaged 2026-09-24 over all 164 entries.

| Outcome | Rows |
| --- | --- |
| `ships` | 22 |
| `trainable`, no recipe yet | 65 |
| `offline` | 38 |
| `uncertain`, a reference is unread | 15 |
| `untrainable` | 7 |

**Gaps against this rule.** These are package-level rather than per-model:

- No text data adapter (tokenize, template, mask) and no audio example adapter exist; no
  response-masked SFT objective exists.
- The dense Qwen, hybrid, and Gemma 3 decoders are LoRA-feasible at 4B and under and have no public
  builder, so no fine-tune of them is reachable. Feasibility and reachability are separate questions,
  and a public builder is not evidence of a training path: Qwen4-Exp and Mamba-2 have fully public
  builders and are offline on size.
- `NFKMLXTrainer` has no gradient accumulation, validation hook, or bf16 training, and does not
  checkpoint optimizer state. (2026-09-23) It schedules the learning rate.
- `NFKMLXLoRA` adapts `Linear` only, never `Conv2d` or the expert switch layers, and only through
  `@ModuleInfo` properties.

- `NFKMLXFineTune` — the sequence every recipe runs, hoisted out of the recipes that were writing it
  by hand. `run(_:freezing:optimizer:reference:referenceSchedule:steps:…)` freezes, takes the caller's
  optimizer or builds the reference's, resolves the schedule, and calls `NFKMLXTrainer.train`. Three
  of those four steps are places a recipe has already been wrong, which is the argument for writing
  them once.
  - **Freezing runs before the optimizer is built**, so a frozen parameter carries no optimizer state.
    Building the optimizer first gives it state for the whole model, which costs the memory the run
    was frozen to save and reports nothing. The freezing closure may throw, because a LoRA policy
    installs its adapters there and throws when its predicate matches no layer; the error ends the
    run before the optimizer is built or a step is taken (Sa2VA's `prepare`, Florence-2's adapter).
  - **The schedule resolves against the CALLER's optimizer, not the recipe's.** A caller who passed
    an optimizer chose its rate, so the trainer runs it with no schedule; a caller who passed none
    gets the reference's schedule. Passing the recipe's own reference optimizer into that decision
    makes every run constant-rate, and nothing reports it. The two parameters are separate for exactly
    that reason, and `testACallersOptimizerHoldsItsRateAgainstTheReferenceSchedule` pins it.
  - **Holding a caller's rate means applying no schedule, not a constant one.** `NFKMLXLearningRateSchedule.resolved`
    answers `.constant` there, and the trainer asks any scheduled optimizer for a single rate per
    group, which Adafactor does not have: a caller's Adafactor threw `unsupportedConfiguration` from a
    recipe that never scheduled it. `run` passes nil instead, which holds the rate the same way and
    works with every optimizer (`testACallersOptimizerWithNoSingleRateRunsWithoutASchedule`).
  - **The reference optimizer is built lazily**, so a recipe whose reference walks the parameter tree
    (SAM 2's layer decay, SegFormer's norm exemptions) does not pay for it when the caller supplied
    one.
  - What stays in the recipe is what a caller reads: the example tuple, the objective's call shape,
    the knobs the reference exposes, and the preconditions. The recipe keeps its own public signature.
  - **No trainable-model protocol (decided 2026-09-25).** The shared contract is the minimum shipped set
    above, and `run` is the one generic fine-tune. A Swift protocol would have to state a builder, an
    example type, and an objective, and the recipes agree on none of the three. The builders take
    `variant:`, `classCount:`, `release:`, `directoryURL:`, or a `configuration`. Twenty-five recipes
    are static over a net and seven live on a built model that holds its tokenizer or embedder. Each
    example is the model's own shape. A protocol over them needs associated types, which erase to
    nothing a caller could use and which Objective-C cannot see, so it would add conformances without
    removing code. What the protocol was meant to guarantee is enforced where it can be checked: `run`
    owns the sequence, and each recipe's tests cover its builder, freezing, objective oracle, and round
    trip.
  - It serves both recipe shapes. A static recipe takes a net (`NFKMLXSegFormer.fineTune`); an
    instance recipe lives on a built model that holds the tokenizer and release directory and takes
    `[Example]` (`NFKMLXLaya.fineTune`). `run` takes the net and closures, so either composes with it.
  - Every recipe runs through it: `NFKMLXSegFormer`, `NFKMLXVJEPA2`, `NFKMLXTrOCR`,
    `NFKMLXTableTransformer`, `NFKMLXFlorence2`, `NFKMLXSa2VA` (one internal `run` serving the InternVL,
    Qwen-VL, and LLaVA overloads, over five or six arrays a step), `NFKMLXSAM2`, `NFKMLXSAM3`, the
    Cosmos Tokenizer, `NFKMLXZeroDCE`, `NFKMLXWhisper`, the CLIP probe, both Laya recipes, the two
    Open-Jev recipes, the translators (one internal recipe in `NFKMLXTranslationTraining` serving
    Marian, M2M-100, and MADLAD-400), `NFKMLXTranslateGemma`, the Granite and Nemotron hybrids, and
    the Qwen3-VL embedding adapter and reranker head. `run` has the trainer's three forms: `batch:` for an input and a target,
    `sample:` for an unlabeled step (SAM 3, whose targets travel beside the batch, the Cosmos
    Tokenizer's reconstruction, and the instance recipes that index their own encoded examples), and
    `arrays:` for any count. A recipe with no freezing policy passes an empty closure (Zero-DCE, the
    CLIP probe). A recipe that takes no caller optimizer passes `optimizer: nil`, so its reference
    optimizer always runs and the caller's `learningRateSchedule` still overrides the reference's
    (Laya, Open-Jev, Qwen3-VL retrieval). `NFKMLXFineTuneTests` pins the rules directly.
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
  - **A frozen subtree stays in evaluation mode.** `train(true)` sets the flag on every module in the
    tree and freezing does not touch it, so a frozen `BatchNorm` normalizes with the batch's own mean
    and variance and folds them into the running statistics it was released with. A head-only run over
    a pretrained convolutional backbone would therefore change what the frozen backbone computes and
    overwrite its statistics from batches of one or two examples, and `NFKMLXWeights.save` writes
    those statistics into the checkpoint, so the damage outlives the run. `enterTrainingMode` returns
    every subtree that holds parameters and has none trainable to evaluation mode; a subtree with no
    parameters at all follows its parent, so a dropout inside the trainable group still drops.
    `testAFrozenNormalizationKeepsItsReleasedStatistics` and
    `testAFrozenNormalizationNormalizesWithItsReleasedStatistics` fail without it, and
    `testAnUnfrozenNormalizationStillUpdatesItsStatistics` is what stops the rule from over-applying.
    No shipped recipe reached the defect (SegFormer's frozen encoder normalizes with LayerNorm), and
    every head-only recipe over a BatchNorm backbone would have.
  - A run over a model with no trainable parameter left throws `NFKMLXError.nothingToTrain` rather
    than reporting a loss curve for an update that changes nothing. A LoRA predicate that matched no
    layer leaves exactly that state, and `apply`'s return count is easy to ignore.
  - `NFKMLXTrainingCheckpoint` clamps `everySteps` to at least one: the loop writes on
    `(step + 1) % everySteps`, which traps on zero.
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
- `NFKMLXEmbeddingProbe` / `NFKMLXEmbeddingProbeBackend` — the probe itself, shared by every image
  embedder: `NFKMLXCLIPProbe` and `NFKMLXCLIPProbeBackend` are CLIP's names for it, and SigLIP 2 trains
  it over its pooled image embedding (`NFKMLXSigLIP2Probe.swift`). `init(weightsURL:)` reads the width
  and class count from a saved file, which is what lets an Objective-C app install a Swift-trained probe
  (`NFKMLXSigLIP2 probeBackendWithProbeURL:labels:error:`). A model whose unloaded tower is built with
  zero placeholders embeds every input identically, so a weight-free probe test first asserts that its
  categories embed apart.
- `NFKMLXEmbeddingAdapter` / `NFKMLXEmbeddingRankingObjective` — the retrieval probe, shared by every
  text embedder: an identity-initialized linear map over the frozen embedding under sentence-transformers'
  `MultipleNegativesRankingLoss`. The Qwen3-VL embedder carries it under its own names, and
  `NFKMLXTextEmbeddingBackend` carries it for Qwen3-Embedding and EmbeddingGemma: `embeddings(for:)`
  encodes a corpus once without the adapter, `fineTune(adapter:…)` trains, and `loadAdapter(from:)`
  (`@objc loadAdapterFromURL:error:`) installs the result for every later embedding.
- `NFKMLXTranslationTraining` — the translators (OPUS-MT, M2M-100, MADLAD-400) adapt with LoRA on
  the decoder's query and value projections through `NFKMLXMarian.fineTune`, `NFKMLXM2M100.fineTune`,
  and `NFKMLXMADLAD.fineTune`, the encoder frozen. `NFKMLXTranslationObjective` is teacher forcing
  (target shifted right behind the start token, mean cross-entropy over every target position), the
  reference's `labels=` loss; measured on released weights within 3e-5 for all three
  (`mlx-models-translation.md`). The round trip goes through `network(directoryURL:)` and
  `translator(net:directoryURL:)`.
- `NFKMLXWhisperTraining` — domain adaptation for speech (jargon, accents, recording conditions), and
  the recipe LoRA exists for. `NFKMLXWhisperObjective` is teacher forcing: the decoder sees the whole
  target sequence at once and each position is scored on the next token, so a step is one forward pass
  rather than one per token. `loss(logits:tokens:)` is separable from the forward for the oracle, as in
  the other two objectives. Only `decoder.blocks.*.{query,value}` are adapted — the reference LoRA
  choice, and the encoder's audio features transfer across domains. `NFKMLXWhisper.spectrogram` pads or
  trims to the 30-second window, the single biggest accuracy factor in reaching reference parity. **`NFKWhisperAttention.query`/`value`/`out` gained `@ModuleInfo`** so they
  can receive adapters; the wrapper keys equal the property names, so checkpoints are unchanged.
- `NFKMLXGraniteTraining` — the first on-device language-decoder fine-tune, adapting Granite 4.0-H to a
  consumer's own text with LoRA on the attention query and value projections through
  `NFKMLXGraniteHybrid.fineTune`; the Mamba layers, the embeddings, and the feed-forward stay frozen.
  `NFKMLXGraniteObjective` is causal language-model teacher forcing (each position scored on the next
  token, mean cross-entropy over the `T − 1` shifted positions), the reference's `labels=` loss;
  `loss(logits:tokens:)` is separable from the forward for the oracle, as in the other decoder
  objectives. Measured against transformers' `GraniteMoeHybridForCausalLM` loss on the tiny dense
  config (`run_reference.py granite_hybrid_loss`) within 1e-3. The round trip goes through
  `network(weightsURL:configuration:)`, which reads the merged single-file checkpoint (the release
  itself is a directory). Granite gives only a minority of layers attention, so the LoRA target set is
  the reference's attention choice restricted to those layers; a run whose configuration is all-Mamba
  would adapt nothing, which `apply` reports rather than hiding.
- `NFKMLXNemotronTraining` — the Nemotron Nano 2 fine-tune, the same shape as the Granite one:
  `NFKMLXNemotronH.fineTune` adapts the attention query and value projections with LoRA, the Mamba
  layers, feed-forwards, embeddings, and `lm_head` frozen. `NFKMLXNemotronObjective` is causal
  language-model teacher forcing, `loss(logits:tokens:)` separable from the forward, measured against
  transformers' `NemotronHForCausalLM` loss (`run_reference.py nemotron_h_loss`) within 1e-3 (4.851059
  vs 4.8510590). The round trip goes through `network(weightsURL:configuration:)`. Nemotron's attention
  sits under each block's `mixer` (not `self_attn`), so the LoRA target predicate matches the `q_proj` /
  `v_proj` suffix; only the attention blocks carry those, a sparse subset of the layer array that
  `NFKMLXLoRA` reaches through its per-owner fallback.
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
  `IK_PARITY_SEGFORMER_LOSS`). The default optimizer trains the decode head at 6e-4, ten times the
  encoder's 6e-5, as the configuration's `head` key sets, and follows its `poly` schedule after a
  1,500-step warm-up; the configuration does not clip.
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

- **A training test cannot read a short run's loss.** The seed fixes the weights, not the gradients:
  MLX's backward pass is not reproducible on either device, and six steps on the GPU move the loss
  by less than the run-to-run spread moves it. Pin a run whose loss is asserted to the CPU with
  `NFKMLXDevice.perform(on: .cpu)`, or judge it by parameters that moved or by a run long enough
  that the progress dwarfs the spread. The measurements are in `mlx-runtime-gotchas.md`, and the
  consumer-facing write-up is in `Docs/mlx-runtime-hazards.md`. No single kernel is at fault: every
  layer alone is exact on both devices, so a fine-tune's instability is a property of the whole
  backward graph rather than of one operation to work around. The cause is
  MLX's buffer cache, by a mechanism that is not established. `NFKMLXTrainer` now holds the cache
  limit at zero for the duration of a GPU run, which is the only mitigation measured to fix a
  training loop: over 60 six-step runs the loss ended above where it started 8 times with the cache
  left alone, 4 times reclaiming it per step, and 0 times under the default policy. The parameter is
  ``NFKMLXTrainingCachePolicy`` and it costs 15% to 26% of throughput on a 23.6M-parameter stack,
  against 4.58 GB of buffers the run no longer holds. mlx core 0.32.0 fixes the defect, and
  mlx-swift 0.31.6 vendors core 0.31.1, so the policy is retired on the release that brings core
  0.32.0 into the package. Do not pin a training run to the CPU: a CPU
  training-mode forward kills its process about one time in ten, in MLX's own convolution.

- **`testASeededRunTrainsDown` still fails occasionally, and the threshold is not the thing to
  change.** Observed 2026-09-21 in a full `swift test` run of the companion: twelve seeded steps went
  0.5203694 to 0.4844541, a ratio of 0.931 against the test's 0.85. The test's own comment calibrates
  that threshold over 20 runs whose worst ratio was 0.656, so 0.931 is outside the sample the
  threshold was fitted to. Five isolated repeats of the same test passed. The run that failed sits
  after roughly 1200 other tests, with the GPU in a state the isolated repeats do not reproduce, so
  the isolated passes do not settle the in-suite case.

  Do not recalibrate the threshold over more runs. The distribution being fitted is produced by the
  wrong-backward-pass defect above: the package pins mlx-swift 0.31.6, which vendors mlx core 0.31.1,
  the version measured at 0 of 25 GPU gradients matching the CPU. Twelve steps on wrong gradients
  give exactly this spread. The remedy is the dependency bump the exit condition in
  `mlx-runtime-gotchas.md` describes, and this test is a concrete instance of the one condition that
  note records as still unmeasured on 0.32.x: a training loop with the cache left on. The bump was
  put to the developer on 2026-09-21 with these numbers and deliberately not taken, because it
  touches every model in the package; the failure is recorded here instead. Whoever makes the bump
  should run this test 20 or more times on both cores and record the two distributions.
