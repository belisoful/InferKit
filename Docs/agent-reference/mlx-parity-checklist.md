<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Completing an InferKitMLX model to parity (the documentation checklist)

A model is not done when its parity test passes — it is done when every listing is updated too. A
partial update leaves the model missing from some indexes, which is the failure this checklist exists
to prevent. Update all of these, in the modality's existing section, mirroring the sibling rows:

- `Docs/agent-reference/mlx-models-<class>.md` — a full per-model entry in the model list of the
  file for the model's class (image restoration, detection, speech restoration, …), and the model's
  registered name in the `registerAll` prose list in `mlx-companion.md`. Use the actual measured
  cosines, not the `> 0.999` test threshold. `CLAUDE.md` / `AGENTS.md` carry no per-model entries.
- `README.md` — the model's name in the modality bullet of the model list.
- `Docs/companions.md` — the full model gallery (README links here as "the full model gallery").
- `Docs/model-index.md` — the index row (entry class, network, configuration, registered name, the
  Swift + Objective-C copy-and-paste, base backend).
- `Docs/model-parity.md` — the parity row with the real recorded numbers (capture them by temporarily
  printing the measured cosines in the parity test, then revert the prints).
- `Docs/examples.md` — the modality's gallery snippet.
- `Docs/inference-guide.md` — the Roadmap, if the model was a roadmap item (mark it shipped).
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/InferKitMLX.md` — the gallery Topics list.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/ModelGallery.md` — the gallery table row.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/ModelIndex.md` — the index table row and the
  per-section copy-and-paste code block.
- `InferKitMLX/Examples/MLXModelGalleryExamples.swift` — the live per-model gallery example (a build +
  representative forward). This is a compiled test, so run it (`InferKitMLXExamples` scheme).
- `Tools/validation-assets/manifest.json` — the checkpoint/record/config entry (and an
  `oracle_environments` note if the model needs a new interpreter or extra packages).
- `~/.inferkit-validation.json` — the model's `IK_VAL_*` / `IK_PARITY_*` keys, as absolute paths into
  the local validation store, so the full check exercises the model by default rather than only when
  those keys are set in the environment. The test class must read them through
  `NFKMLXValidationConfig.environment` (the process environment merged with that JSON), not
  `ProcessInfo.processInfo.environment` directly, or the JSON keys never reach it. This file is a
  machine's local config, not a tracked repository file, so its "update" is provisioning rather than a
  commit — but skipping it leaves the model's parity test silently skipped on a plain run (green with
  nothing behind it), which is exactly the gap a full models test exists to catch.

Not a listing, so not required per model: `InferKitMLX/ObjCExamples/MLXObjCExample.m` is a curated
illustrative set, not an exhaustive gallery.

The code/wiring that accompanies the docs (the model file, `NFKMLXReferenceModels.registerAll`
registration, the `run_reference.py` oracle mode, and the parity test) is covered by the shipped-model
pattern; a converter under `Tools/<model>-to-safetensors/` is optional because the native `.pth`/`.pt`
reader loads most released checkpoints directly.

## Entry house style (keep the whole consistent)

These files accreted across many sessions and drifted into several voices for the same thing. A new
entry matches its neighbors in the file it lands in, not the last entry a different session happened to
write. The "Documentation Style (enforced)" section of `CLAUDE.md` governs the prose (no em-dash dramatic asides,
no antithesis, no rule-of-three, one fact per sentence, present tense, American English); it applies to
these entries too, and where an older entry breaks it, the break is not a precedent to copy. The
per-file shape:

- `Docs/agent-reference/mlx-models-*.md` model list — one bullet, `` `NFKMLXFoo`` `` (`` `(@objc)` `` only when the
  class is), then ` — `, then a lowercase noun phrase naming what it is and its reference
  (`the X (`ReferenceClass`, Vendor)`). State the architecture, the load-bearing facts, and the measured
  parity with the ACTUAL cosines. Say "at reference parity" in running prose — one casing, lowercase, no
  bold on the phrase itself; reserve bold for a specific load-bearing noun, not for emphasis. "at parity"
  is only for a short back-reference to a result already stated (a second size, another variant). Depth
  is proportional to the model's novelty, not to how recently it was added; a configuration-only variant
  is a sentence, not a section.
- **README.md** — the model's name in the modality bullet, nothing more.
- **Docs/companions.md** — a prose gallery bullet. Describe the model and its variants/presets and say
  "at reference parity against <reference>"; do NOT quote a tiny-config cosine here (those live in
  model-parity.md). A released-weight cosine may appear when it is the headline result, matching the
  neighbors that do.
- Docs/model-index.md and the DocC `ModelIndex.md` — one table row (entry class, network,
  configuration, registered name or "Swift API", the construction line, base backend). In the DocC file,
  the section also carries a copy-and-paste code block that is a per-FAMILY cheat-sheet — one
  construction line per model family, not a mirror of every table symbol — so every family in the table
  has at least one line there, and a code line never names a family the table omits. The drift to stop is
  a family present in one and absent from the other (a new family's table row with no cheat-sheet line,
  as SD3/FLUX/ControlNet were), not a per-symbol mismatch.
- Docs/model-parity.md and the DocC `ModelGallery.md` — the parity row (and gallery row) with the
  REAL recorded cosines at full precision, not rounded or a `> 0.999` threshold. Multiple seams are
  `seam A x; seam B y; final z`.
- **DocC `InferKitMLX.md`** — the symbol(s) in the modality's Topics list.
- **`MLXModelGalleryExamples.swift`** — a build-plus-forward example, for a REGISTERED `@objc` model.
  A generative pipeline (SD3/FLUX/Z-Image/…) is not registered and is absent here by design, so a new
  one that is also unregistered stays absent — consistently, not by omission.

When an edit touches a file, leave that file MORE consistent than you found it: if the neighbors already
share a shape, conform to it; if a genuinely better shape is warranted, do not introduce a third — raise
it so the whole file moves together rather than one more divergent entry accruing.
