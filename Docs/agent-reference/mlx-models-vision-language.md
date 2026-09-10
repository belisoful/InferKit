<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: vision-language models

- `NFKMLXSmolVLM` / `NFKMLXSigLIPNet` / `NFKMLXSmolVLMConnector` / `NFKMLXSmolVLMNet` — the package's
  **first vision-language model**, SmolVLM2-500M: an image and a question in, an answer out, which is
  the dominant 2026 on-device use (captioning, VQA, doc/screen understanding). Three parts: a **SigLIP
  vision encoder** (`NFKMLXSigLIPNet`: patch-embedding convolution + learned position embedding + 12
  bidirectional layers + post-norm; separate q/k/v/out projections with bias, gelu-tanh, LayerNorm eps
  1e-6), a **pixel-shuffle connector** (`NFKMLXSmolVLMConnector`: the Idefics3 pixel shuffle folds a
  4×4 patch neighborhood into one token at 16× the channels — 1024 patches per tile → 64 — then one
  bias-free `proj` to the decoder width), and a **Llama decoder** the dense `NFKMLXLanguageNet` already
  runs (`text_config` model_type llama, hidden 960, 32 layers, loaded from the checkpoint's
  `model.text_model.` subtree remapped onto `model.`). The **fusion** embeds the text, then splices the
  flattened projected vision tokens into the decoder's input embeddings at the `image_token_id` (49190)
  positions with a `where` over a gathered feature index (no per-row scatter), and the causal decoder
  reads the whole fused sequence. Reference parity against transformers' own
  `SmolVLMForConditionalGeneration` (`run_reference.py smolvlm`, the `llm` oracle — needs Pillow,
  torchvision, num2words), staged by the isolation harness: SigLIP embeddings 0.9999999999, layer 0
  0.9999999999, the full encoder 0.99999999997, the connector 0.9999999999, and the fused decoder
  logits predicting the reference's token at every one of the 1140 positions (last-position cosine
  0.9999999999) with the greedy continuation **token for token**. **Two bugs the isolation harness
  located, neither guessable:** SigLIP's position ids are not the row-major `0 … 1023` — the reference
  buckets each patch's fractional coordinate against `1/side … (side-1)/side` with a `1 - 1e-6` factor,
  so a full 32-patch row maps to `[0, 0, 1, …, 30]` (`positionIds` reproduces it, held as a Swift array
  so the stored constant does not enter `parameters()` — a stored `MLXArray` would, and the loader would
  then report it as an uncovered weight); and SmolVLM's `lm_head` is not tied to the embedding
  (byte-diff 0.85 despite the tied geometry), so the decoder loads its own `lm_head.weight` from the top
  level (`tiesWordEmbeddings: false`) rather than reusing the embedding — loading it tied scored logit
  cosine 0.83 and drifted after the sixth continuation token. **The consumer path:**
  `NFKMLXSmolVLMImageProcessor` tiles a `CGImage` the way SmolVLM does (longest edge scaled to 2048,
  split into `⌈h/512⌉ × ⌈w/512⌉` 512×512 sub-tiles plus a global 512×512 thumbnail appended last, each
  normalized to `-1 … 1`), and `NFKMLXSmolVLM.prompt(rows:cols:question:)` builds the expanded
  `User:` + per-tile `<fake_token_around_image><row_r_col_c>` + 64 `<image>` + `<global-img>` tile +
  question + `<end_of_utterance>\nAssistant:` string, which the byte-level BPE tokenizer (every added
  token registered as a special so the string segments on them) turns into ids **token-exactly** against
  the processor. The resize is CoreGraphics, not the reference's PIL LANCZOS, so a consumer caption is
  not token-identical to the reference — a documented approximation; the network is at parity on the
  reference's own pixel values, and end to end on the real Apollo-astronaut validation photo the model
  answers "a man is standing in front of a backdrop that resembles the moon. He is dressed in a white
  …". ObjC reaches it through `smolVLMWithDirectoryURL:error:` and `answerForImage:question:`.
- `NFKMLXQwen3VLVisionNet` / `NFKMLXQwen3VL` — the vision tower of a **second VLM**, Qwen3-VL-2B, and a
  second vision architecture beside SmolVLM's SigLIP. Qwen3-VL's encoder is a **2D-rotary ViT** whose
  patches are laid out in **2×2 merge blocks** (not row-major): a patch embedding (the reference's
  full-kernel `Conv3d` written as one `Linear` over the flattened `3·2·16·16` patch), a bilinearly
  interpolated position embedding (the learned 48×48 grid resampled to the image's grid, then
  reordered to merge-block order to line up with the patches), 24 blocks with 2D rotary and gelu-tanh
  MLP, a **merger** that folds each 2×2 block to `out_hidden` (2048), and a three-layer **deepstack**
  (feature maps from vision layers 5/11/17, each through its own merger with a post-shuffle norm). The
  2D rope pairs channel `i` with `i+32` over a `[row·freqs, col·freqs]` table; the merger groups four
  Consecutive patches, which is why the merge-block patch order is load-bearing. Reference parity on
  the first numeric run against transformers' own Qwen3-VL vision model (`run_reference.py qwen3vl`,
  the `llm` oracle — needs Pillow/torchvision): patch embedding 0.9999999999, position embedding
  0.9999999999, merged output 0.9999999997, and every one of the three deepstack features
  0.9999999999. The decoder is the Qwen3 dense stack `NFKMLXLanguageNet` already runs, loaded from the
  checkpoint's `model.language_model.` subtree. Qwen3-VL is much more than "reuses the Qwen3 decoder":
  the decoder adds interleaved M-RoPE (3D positions, `mrope_section [24,20,20]`) and deepstack injection
  at its first three layers, and `get_rope_index` computes the 3D positions from the token layout.
  That decoder integration is shipped, in the shared `NFKMLXLanguageNet` (the dense Qwen3, the
  embedders, SmolVLM, the music AR stage, and the Z-Image text step all reuse it, kept byte-identical
  through nil defaults): an opt-in M-RoPE (precomputed interleaved 3-D cos/sin in `NFKLMAttention`,
  rotate-half; the 64 frequency pairs interleave the T/H/W axes by `c % 3` for `c < 60`, then T, which is
  exactly `mrope_section [24, 20, 20]`) and deepstack injection (adding `features[i]` at the image-token
  positions after layers 0/1/2), both carried by an `NFKLMMultimodal` struct through
  `hiddenStates(fromEmbeddings:multimodal:)`. `NFKMLXQwen3VL` gained `decoder(directoryURL:)` (the
  `model.language_model.` subtree; Qwen3-1.7B geometry at `ropeTheta` 5e6), `ropePositionIds`
  (`get_rope_index`, single image), `mropeCosSin`, and `logits(...)`. Reference parity on the released
  4.25 GB Qwen3-VL-2B (`testQwen3VLDecoderMatchesTheReferenceOnReleasedWeights`, the recorded parity
  vision features fed in): logit cosine 0.99999999998, argmax 80/80, the first continuation token
  matching. The shared-decoder change was re-verified against the dense Qwen3 (0.99999999999), SmolVLM
  (argmax 1140/1140), and Qwen3-Embedding (0.99999999999) records. `NFKMLXQwen3VLImageProcessor` is the
  `smart_resize` + patchify input adapter (CoreGraphics resize, the documented approximation).
  The larger sizes are read from their own releases: `NFKMLXQwen3VLVisionConfiguration.configuration(fromHuggingFace:)`
  (model_type `qwen3_vl` or `qwen3_vl_moe`; the position grid's side is the square root of
  `num_position_embeddings`), `visionNet(directoryURL:)` (sharded through `NFKMLXReleaseWeights.arrays`),
  `decoderConfiguration(directoryURL:)` (the `text_config` through `NFKMLXLanguage.configuration(fromJSON:)`,
  tied unless the release ships a head), and `decoder(directoryURL:)`, which splits the 30B-A3B's fused
  `gate_up_proj [E, hidden, 2·inter]` into the module's `gate_proj` / `up_proj [E, inter, hidden]` and
  transposes `down_proj`. The 4B keeps the 2B's 24-block tower; the 8B, 32B, and 30B-A3B run the
  27-block one. Held to their released headers by shape, tower and decoder together: 713 / 750 /
  1058 / 930 tensors, 0 missing, 0 mismatched, 0 unaccounted.
