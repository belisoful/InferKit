<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: 3D reconstruction and generation

The modality is one image (or one prompt) in, an object out. The core carries the object as `NFKMesh`
(`Sources/InferKit/NFKMesh.m`): packed float32 positions, optional normals, vertex colors, and texture
coordinates, uint32 triangles, typed accessors, and OBJ, PLY, and GLB writers and readers. A backend
returns it under `NFKOutputMesh`; the catalog names the value type `NFKModelValueTypeMesh` and the task
`NFKModelTaskImageTo3D`. On the plugin and server wire a mesh is a JSON object of its counts carrying its
binary PLY, inline as base64 or beside the JSON as a file (`NFKWireValueObjects.m`, `NFKInferenceWireCoding.m`).
The candidate survey behind the modality is `Docs/porting-candidates.md` ("3D asset generation
candidates"); TripoSR is its first port, chosen because it is the smallest permissive model and forces
the mesh type, the surface extraction, and the writers every later port uses.

- `NFKMLXTripoSR` (`@objc`) — TripoSR (`stabilityai/TripoSR` at `5b521936`, MIT, 1.68 GB float32, 419M
  parameters), a single-forward reconstructor. `+register` under `triposr`; `backendWithWeightsURL:` reads
  the release's `model.ckpt` through the native torch reader; `network(weightsURL:configuration:)` is the
  public builder. `NFKMLXTripoSRNet` (`NFKMLXTripoSR.swift`) mirrors the checkpoint's keys:
  - `image_tokenizer.model.*` is transformers' `ViTModel` layout of DINO ViT-B/16 (`facebook/dino-vitb16`):
    `embeddings.{cls_token,position_embeddings,patch_embeddings.projection}`, `encoder.layer.N.{attention.attention.{query,key,value},attention.output.dense,intermediate.dense,output.dense,layernorm_before,layernorm_after}`,
    `layernorm`. LayerNorm eps is 1e-12 (BERT's), the GELU exact, the pooler dropped at load. Every token
    (class first, 1025 at 512 pixels) is the conditioning. The checkpoint stores the position embeddings at
    the 224-pixel grid (14 × 14 + 1); `ViTModel` interpolates them to 32 × 32 at run time
    (`interpolate_pos_encoding`, bicubic, `align_corners=False`, no antialias, `size=`), which the loader
    does once at load through `NFKMLXPhi4MMImageProcessor.bicubicResized` (torch's cubic, a = −0.75, clamped taps).
    transformers 5.x renames these keys (`layers.N.attention.q_proj`); the port keeps the 4.x names the
    checkpoint has, and the oracle runs on transformers 4.57.
  - `tokenizer.embeddings` `[3, 1024, 32, 32]` is the learned triplane token bank, flattened plane-major
    then row-major into 3072 tokens of 1024 channels.
  - `backbone` is diffusers' `Transformer1D`: a GroupNorm (32 groups, eps 1e-6) over the token channels,
    `proj_in`, 16 `BasicTransformerBlock`s (LayerNorm eps 1e-5; `attn1` self, `attn2` cross to the 768-wide
    image tokens; projections without bias, `to_out.0` with; GEGLU feed-forward `ff.net.0.proj` →
    `ff.net.2`), `proj_out`, and the residual from before the norm. Channels last, the GroupNorm groups run
    along the channel axis with statistics over the group's channels and every token, which is what the
    reference's `[N, C, tokens]` GroupNorm computes.
  - `post_processor.upsample` is a stride-2 `ConvTranspose2d` 1024 → 40 (PyTorch weight `[in, out, kH, kW]`,
    the transposed-convolution axis order at load), giving three 64 × 64 planes of 40 channels: the scene code.
  - `decoder.layers.{0,2,…,18}` is the NeRF MLP as `nn.Sequential` indices (SiLU between, no parameters),
    120 → 64 ×9 → 4; the module keeps parameter-free placeholders at the odd indices so the keys match.
    `query(_:sceneCode:)` is the reference's `query_triplane`: positions over the radius to (−1, 1), each plane
    sampled bilinearly (`grid_sample`, `align_corners=False`, zero padding) at its coordinate pair (xy, xz,
    yz), the three 40-vectors concatenated, density `exp(out₀ − 1)`, color `sigmoid(out₁…₃)`.
  - The mesh: density on a `resolution³` grid over `[−0.87, 0.87]³` (x-major, as `meshgrid(indexing="ij")`
    lays it out), marching cubes at threshold 25, the decoder's color at each vertex. `NFKMarchingCubes`
    derives its 256-case table at first use from the cube's topology (crossed edges paired around each
    face, an ambiguous face keeping its inside corners apart, the loops fanned and wound outward by a
    trilinear indicator test) instead of a typed table; the reference uses torchmcubes, so the two meshes
    are compared by nearest-vertex distance and face count, not vertex by vertex.
  - The frame: the reference reconstructs the picture with its up along +z, its right along +y, and its
    front toward +x. Measured on the released weights (2026-10-28) with a picture whose right and top
    carry their own colors: the right's vertices sit +0.378 along y, the top's +0.422 along z, and the
    depth is the thinnest extent (0.73 along x, against 0.91 and 1.01). `NFKMesh` holds glTF's frame (y
    up, a reconstructed object facing +z), so the backend turns the positions `(x, y, z) → (y, z, x)`
    after the color query, which reads the decoder in its own frame (`NFKMLXTripoSR.gltfFrame`). The map
    is a rotation, so the winding holds. The parity test turns the reference's vertices the same way.
    A later image-to-3D port states its reference's frame here and turns it into glTF's at the output.
  - Preparation mirrors the reference demo (`run.py`): an image with alpha below one is cropped to its
    foreground (the half-open `image[y1:y2, x1:x2]` drops the last row and column), padded to a square, padded
    again to `foregroundRatio` (0.85), composited on mid-gray; an opaque image is used as it is. The model's
    own resize is `F.interpolate(bilinear, align_corners=False, antialias=True)` to 512, a triangle filter
    whose support widens with the scale (`antialiasBilinearMatrix`).
  - Request keys (`NFKMLXTripoSRParameterKey`): `resolution` (256), `threshold` (25), `foregroundRatio`
    (0.85), `vertexColors` (true). The backend declares them, `NFKInputImage`, and `NFKOutputMesh`.
  - Parity: measured against the repository at 107cefd (`run_reference.py triposr --plate subject --size 320`,
    the `triposr` oracle environment), on the released weights: preparation 1.0 (exact), conditioning 0.99999999999996, position embeddings 0.99999999999989, image tokens 0.9999999999940, scene codes 0.99999999999941 (3 × 64 × 64 × 40), density on the 32³ grid 0.99999999997716, color 0.99999999999965; the 64³ mesh has 10,328 faces on both sides and its vertices lie 1.03e-5 of a cell from the reference's in either direction (`NFKMLXTripoSRTests`, 2026-10-28). The reference's own float32 run is the record, so these are float32 rounding.
  - **Customization: untrainable here.** The repository publishes inference code only; the paper's
    multi-view rendering objective has no training script to measure against. The network builder stays
    public for an offline recipe.
  - Runtime: one stage, resident; no streaming or paging applies (1.68 GB float32). The registration reads
    `NFKModelLoadOptionPrecision` (`bfloat16` casts the float32 release as it loads; `checkpoint` keeps it)
    and `NFKModelLoadOptionQuantizationBits` (every linear layer whose input width is a multiple of the
    group size packs; the decoder's 120-wide first layer stays as loaded; a group size other than 32, 64,
    or 128 is refused before MLX's quantizer can abort the process); `familyProfiles["triposr"]` declares both. Against the float32 record on the released weights, the scene codes measure 0.99992145 at the bfloat16 cast and 0.99184 packed to 4 bits at group size 64; the float32 default is the one held to the reference. The release is a
    `.ckpt`, so the measurement reads no parameter count from it; the catalog's memory figures come from
    the download size and the measured load.
