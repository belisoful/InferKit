<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: 3D reconstruction and generation

The modality is one image (or one prompt) in, an object out. The core carries the object as `NFKMesh`
(`Sources/InferKit/NFKMesh.m`): packed float32 positions, optional normals, vertex colors, and texture
coordinates, uint32 triangles, an optional `NFKMeshMaterial` (`NFKMeshMaterial.m`: glTF's metallic-roughness
model, each texture an `NFKMeshTexture` holding an encoded PNG or JPEG and its MIME type), typed
accessors, and OBJ, PLY, and GLB writers and readers. The GLB writes the material with its images in the
binary chunk (one sampler, linear with mipmaps, repeating) and the reader reads the first primitive's
material back; OBJ and PLY drop it. A backend returns the mesh under `NFKOutputMesh`; the catalog names
the value type `NFKModelValueTypeMesh` and the task `NFKModelTaskImageTo3D` (`NFKModelTaskTextTo3D` for a
prompt). Gaussians travel as `NFKGaussianSplats` under `NFKOutputGaussianSplats` (value type
`NFKModelValueTypeGaussianSplats`). On the plugin and server
wire a mesh is a JSON object of its counts carrying its binary PLY, inline as base64 or beside the JSON
as a file; a mesh with a material travels as its GLB instead, under `glb` or as a `.glb` file
(`NFKWireValueObjects.m`, `NFKInferenceWireCoding.m`).
Vertex colors are sRGB-encoded. TripoSR's decoder learns them against the input picture's pixels, and the
reference exports them through trimesh as 8-bit display colors; a Blender render of a known material
(linear 0.9, 0.4, 0.1, sRGB 0.955, 0.665, 0.349) came back from TripoSR with a median vertex color of
(0.816, 0.547, 0.328), near the sRGB values (2026-11-05). glTF defines `COLOR_0` as linear, so the GLB
writer converts sRGB to linear (IEC 61966-2-1) and the reader converts back. The round trip is not bit
exact: float32 sRGB → linear is not injective (adjacent inputs share an output in 16.5 % of [0, 1]), and
an enumeration of every float32 in [0, 1] puts the largest difference at 2.98e-8. A colored mesh with a
material therefore crosses the wire (its GLB form) within that difference; one without a material keeps
its PLY, which quantizes to 8 bits.
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
    again to `foregroundRatio` (0.85), composited on mid-gray. An opaque image is centered on a mid-gray
    square of its longer side (a square one is used as it is). The reference resizes an opaque image
    straight to the square, stretching a picture that is not square, and its `--no-remove-bg` help asks
    for one already framed on gray; the square keeps a photograph's proportions, which the stretch lost
    (an 800 × 533 photograph came out 1.02 × 0.966 across its front, measured 2026-10-28). The model's
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
- `NFKMLXStableFast3D` (`@objc`) — Stable Fast 3D (`stabilityai/stable-fast-3d` at `f0c9a8ff`, Stability AI
  Community License, gated, 4.02 GB float32, 1,004M parameters), a single-forward reconstructor of a
  UV-textured object with a PBR material. `+register` under `stable-fast-3d`; `backendWithDirectoryURL:`
  reads the release directory, `backendWithWeightsURL:gridURL:` the two files, `backendWithWeightsURL:grid:`
  a grid already read (`NFKMLXTetrahedralGrid`, `+latticeWithDivisions:` for a test or an example), and
  `backendWithRepo:revision:cacheDirectoryURL:` downloads the checkpoint from the Hub and the grid into the
  cache's URL downloads. `network(weightsURL:configuration:)` is the public builder. The surface stages
  live in `NFKMLXStableFast3DGeometry.swift`.
  - The tetrahedral grid `160_tets.npz` (535,882 vertices, 2,971,452 tetrahedra; a jittered lattice from a
    tetrahedral mesher, stored in no regular order) is not in the weights repository: it is
    `load/tets/160_tets.npz` of the code repository at ff21fc4 (15,408,790 bytes, sha256
    1f4be37efc604d28d55a1a78c2aabefeeab7e63149f541aa45f9dd858ee35bb9). It cannot be regenerated, so the
    catalog lists it as a URL file beside the checkpoint. The archive is deflated; `NFKMLXZipArchive` inflates
    it and a small `.npy` reader takes the float64 vertices (read as float32, as the reference reads them)
    and the int64 indices.
  - `camera_embedder.linear` maps the release's fixed camera (the camera-to-world matrix at distance 1.6
    along +x looking back with +z up, flattened row-major, then the 40° pinhole's intrinsics normalized by
    the side) to the 768-wide embedding that modulates the tokenizer.
  - `image_tokenizer.model.*` is transformers' `Dinov2Model` layout of DINOv2-large with both LayerNorms of
    every layer followed by a modulation, `normN_modulation.linear2` (768 → 2048, SiLU of the embedding
    first, no first projection in the release's single-layer form): `x · (1 + scale) + shift`. LayerNorm eps
    1e-6, exact GELU, `layer_scaleN.lambda1`. The checkpoint stores each modulation twice (under
    `image_tokenizer.modulations.N` and under the layer); safetensors keeps one copy and records the alias in
    its metadata, and the loader reads the layer's. The mask token is dropped. The position embeddings
    (37 × 37 + 1, the 518-pixel grid) are resized at load the way the release's copied DINOv2 resizes them
    on every forward: bicubic with `scale_factor` (36.1 / 37), whose reciprocal is torch's source step, not
    `size=` (`resampleMatrix(from:to:scale:)`). The patch convolution reads 36 × 36 patches of the 512-pixel
    image and drops its last 8 pixels; all 1,297 tokens are the conditioning.
  - `tokenizer.embeddings` `[3, 1024, 96, 96]` is the triplane token bank, flattened plane-major into 27,648
    tokens. `backbone` is the `TwoStreamInterleaveTransformer`: the triplane tokens group-normalized (32
    groups, eps 1e-6) and projected; the latents are the 1,297 projected image tokens followed by the 1,792
    normalized and projected learned latents; four `TwoStreamBlock`s each fuse the triplane into the
    latents (`fuse_block_in`), run three `BasicBlock`s over the latents with cross-attention to the raw
    image tokens, and fuse the latents back into the triplane (`fuse_block_out`); `proj_out` and the
    residual. The release sets `norm_x_input: false`, so the attended stream is not normalized and has no
    `norm_x`. Every attention runs through MLX's fused kernel, because the triplane stream's 27,648 queries
    over 3,089 latents would hold a 5.5 GB score matrix at float32. The GEGLU feed-forward is TripoSR's.
  - `post_processor.upsample.{0,2,4,6}`: four 3 × 3 convolutions (ReLU between) to 640 channels and a
    pixel shuffle by 4 (output channel c at sub-position (i, j) is input channel 16c + 4i + j), three
    384 × 384 planes of 40 channels: the scene code.
  - `decoder.heads.{density,features,perturb_normal,vertex_offset}` are SiLU stacks from the 120 sampled
    features (2, 3, 3, and 2 hidden layers of 64). Density is `exp(x − 1)` (the release's `trunc_exp`, an
    exponential in the forward), the albedo a sigmoid, the perturbed normal normalized (eps 1e-7), the
    vertex offset linear. The planes are sampled as `query_triplane` samples them: positions over the
    radius 0.87 to (−1, 1) and `grid_sample` with `align_corners=True` (TripoSR samples with `False`).
  - `image_estimator`: OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`, plain GELU) reads the masked conditioning
    image after a bilinear resize to 224 without antialiasing and OpenAI's normalization; its projected
    class token feeds, per material, a shared three-layer ReLU stack and two branches whose outputs plus 1,
    through a softplus, are the concentrations of a beta distribution. The value is torch's `Beta.mode`
    (`(α − 1)⁺ / ((α − 1)⁺ + (β − 1)⁺)`, and the first one-hot where both lie below one). The checkpoint's
    text tower is dropped at load. `global_estimator` (two stride-2 3 × 3 convolutions over the three
    1024-channel planes stacked along the channels, a spatial max, a ReLU stack, `softplus(x + 1)`) gives
    the 24 spherical-Gaussian amplitudes the release returns beside the mesh and never uses.
  - The surface: the decoder on the grid's vertices scaled to the cube, the grid deformed by
    `tanh(offset) / 160`, marching tetrahedra at density 10 (`NFKMarchingTetrahedra`: one vertex a crossed
    edge in sorted edge order, the one-triangle tetrahedra first, the release's case table). The extractor
    winds the triangles inward; the release's export reverses them (`trimesh.invert`).
  - The unwrapper ports `uv_unwrapper`: the mesh aligned to its principal axes by `torch.pca_lowrank`
    (reseeded at 0, so its sketch is torch's seed-0 `randn(3, 2)`, held as a constant; two power iterations
    orthonormalized by LAPACK's QR and the projection's SVD by LAPACK's `gesdd`, resolved at run time from
    Accelerate's `sgeqrf_`, `sorgqr_`, and `sgesdd_`, the routines torch links, so the axes take torch's
    signs without a deprecated call); each face box-projected onto the cube face its averaged vertex
    normal points to most, each corner slot scaled by its own largest coordinate along the axis; each cube
    face's charts rotated to a common tangent direction and rescaled; overlapping charts in one slot moved
    twice to the next tier (the extension's coplanar Guigue–Devillers test in the first chart's own frame,
    the clipped polygon's area at least 1e-10, an identical chart skipped); and the three tiers packed into
    a 3 × 2 grid, a lower-third grid of sixths, and a cell a face. The extension's per-slot BVH is part of
    the result, not a search structure alone: a face's query reaches a chart only through node boxes its
    triangle-box test admits, and that test rejects some charts that touch the query only in a sliver
    along a shared vertex, of area near 1e-8, which the chart test alone counts as an overlap. A uniform
    grid in its place moved 12 of 19,480 faces the reference keeps (17 of 124,658 candidate pairs, each
    overlapping in one direction only and box-rejected in that direction; the reference's own
    `triangle_triangle_intersection`, called through its symbol, agrees with the port's on all 17). The
    port builds the extension's BVH (`NFKSF3DTree`, `NFKSF3DChartTree`) and traverses it as `BVH::Intersect`
    does.
  - The bake: `texture_baker`'s CPU rasterizer (texel `(r, c)` samples `(c / side, 1 − r / side)`, its
    corner, through the extension's own BVH: a child is entered when its box holds the point, left
    before right, and the first triangle whose barycentric test passes is taken, so a texel two
    triangles both claim, or one only a box excludes, follows the reference) and interpolator; the albedo and the perturbed
    normal decoded at each covered texel; the normal turned into the texel's tangent frame from the split
    mesh's per-face normals and tangents; `dilate_fill` for `side / 150` iterations; and 8-bit
    quantization with a dither, `floor(256x + d)`. The release draws the dither from NumPy's unseeded
    generator; the backend seeds a SplitMix64 from `NFKParameterSeed`. The release passes the flat texels
    as the normal map's dither mask, so only they are dithered there, and the port does the same.
  - The export follows the release's trimesh: the split mesh's vertices welded where positions agree to
    1e-8 and texture coordinates to 1e-4 (the first kept, in order), the winding reversed, and
    angle-weighted vertex normals in float64 (trimesh's sparse path; the oracle environment carries scipy
    for it). The release's export turns `(x, y, z) → (−y, z, −x)`; its camera looks from +x with +z up, as
    TripoSR's does, so the port turns `(x, y, z) → (y, z, x)`, which faces the picture toward +z as
    `NFKMesh` documents. The two exports differ by a half turn about y.
  - Preparation mirrors `run.py` and `prepare_image`: `resize_foreground` at 0.85 (the box's sides are
    its last index minus its first, the crop's corner and side truncated float64 values, transparent black
    outside the image); Pillow's default bicubic resize of the RGBA image to 512, which premultiplies
    (`MULDIV255`), resamples in 8 bits (`NFKMLXPILResample`, now four-channel), and unpremultiplies with a
    truncating division; the composite on 0.5 by `torch.lerp` (its two-branch form); and the model's
    antialiased resize, the identity at 512. An opaque image skips the framing and is stretched to the
    square, as `prepare_image` stretches it; the release's demo removes its background first, which the
    backend does not.
  - Request keys (`NFKMLXStableFast3DParameterKey`): `textureResolution` (1024), `threshold` (10),
    `foregroundRatio` (0.85), `textureFormat` (PNG; the release writes JPEG), `estimateIllumination`
    (off), `referenceArithmetic` (off), and `NFKParameterSeed`. The result holds `NFKOutputMesh` and, when asked,
    `NFKMLXStableFast3DOutputKey.illuminationAmplitudes`.
  - The reference's two C++ extensions are built for the CPU with `-ffp-contract=off` in the `sf3d` oracle
    environment: Apple silicon's default build fuses multiply-adds (536 fused instructions in
    `uv_unwrapper`), which an x86 build and Swift's float arithmetic do not.
  - The release's downstream is chaotic: a scene code moved by 1.42e-6 relative noise changes 18 to 33
    atlas faces, because the unwrapper's chart overlaps and the atlas tiers flip on ulp-level changes.
    The GPU network cannot reproduce the release's mesh, so `referenceArithmetic` runs the network and its
    field decoder on the CPU in float32 as torch 2.14.1 runs them on Apple silicon
    (`NFKMLXStableFast3DReference.swift`). The checkpoint is mapped, not loaded. The reference is
    bit-identical at 1, 4, and 8 threads, so one order defines it. The kernels follow torch's:
    - `F.linear` is `addmm_impl_cpu_` onto Accelerate's legacy `sgemm_`. A contiguous input is
      `sgemm('T', 'N')` onto the bias with β = 1. A transposed view of a batch-of-one tensor (the
      group-normed triplane, the surface's sampled features) is `matmul`, then `add_`. A two-dimensional
      input takes `addmm` whatever its strides: the bake's features (`query_triplane(…)[0]`) do.
      `normalize_last_dims` sets the leading dimensions.
    - Accelerate's products with a dimension of one (the density head's last layer, the modulations'
      one-row `linear2`) read their matrix in an order that depends on its address modulo the 128-byte
      cache line; the output's address does not matter. torch's operands here all start on a line, so
      the port copies an operand that does not (Swift arrays and the mapped checkpoint's tensors need
      not).
    - Convolutions are `slow_conv2d`: `im2col` with rows `(channel, ki, kj)`, then `sgemm('N', 'N')`.
    - LayerNorm and GroupNorm use `RowwiseMoments`: four-lane Welford chunks of 16 with fused updates, a
      cascade by depth, and fused scalar merges. GroupNorm takes its rstd in float64.
    - Attention is `cpu_flash_attention`: query blocks of 256, 64, or 32 against key blocks of 512,
      `sgemm` for the scores and the values, and `exp_u20` (Arm's `expf`, Sleef's in a four-lane group
      past 87.3). The lane sums fold as `(l0 + l2) + (l1 + l3)`. The running sum updates with a fused
      multiply-add.
    - The vector math is Sleef's u10 `expf`, `tanhf`, `sinf`, and `cosf`, and torch's vector `erf`.
      `cpu_kernel_vec` runs eight-float groups in vector form and the tail through libm.
    - The position embeddings use the non-separable bicubic kernel, with clang's contractions.
    - Sampling is `grid_sample`'s vector bilinear kernel.
    The torch ops of the surface stages follow torch too, on either path:
    - `torch.cross` fuses its first product.
    - `F.normalize` accumulates the squares with fused multiply-adds.
    - A three-wide `sum(-1)` adds in order. `torch.dot` adds `(p0 + p2) + p1`.
    - The expected-tangent field's `F.normalize(x, -1)` is the p = −1 norm through libm's `powf`, which is
      not the reciprocal for 2% of values.
    - `einsum` rotations are single `sgemm` calls. An explicit multiply-add chain differs in one value.
    - The small tangent-frame `bmm` is a fused chain.
    - The deformation's `tanh` is Sleef's.
    Each kernel was verified bit for bit against torch, and
    `testTheReferenceKernelsReproduceTorchBitForBit` holds digests of torch's results.
  - Parity: measured against the release at f0c9a8ff and its repository at ff21fc4 (`run_reference.py sf3d
    --plate subject --size 512`, the `sf3d` oracle environment), on the released weights (`NFKMLXStableFast3DTests`,
    2026-10-29). The reference arithmetic matches every seam with 0 mismatches: the conditioning and its mask,
    the camera embedding, the position embeddings, the image tokens (1,297 × 1024), the triplane tokens
    (27,648 × 1024), the scene code (3 × 40 × 384 × 384), the density and vertex offset at the 535,882 grid
    vertices, and the albedo and perturbed normal at the 284,404 baked texels. From the photo, its mesh is
    the release's: 10,589 vertices, equal faces, positions 0 apart, vertex normals at cosine 1.0, and both
    textures within one dithered level on every channel. The surface and texture stages fed the reference's
    own inputs match bit for bit, from the alignment and the box, rotated, and placed UVs through every
    normal and tangent, the raster, the bake, the padding, and the export. The GPU path holds the image
    tokens at 0.99999999999366, the triplane tokens 0.99999999999797, the scene code 0.99999999999900, the
    grid density 0.99999999999983, the albedo 0.99999999999982, the roughness within 6.64e-6, and the
    metalness exactly; its mesh is not the release's, because the downstream is chaotic. The ladder against
    the same record: scene codes 0.99999851 at the bfloat16 cast, 0.99908074 at 4 bits.
  - **Customization: untrainable here.** The repository publishes inference code only; the paper's
    staged objective (rendering, material, and illumination losses through a differentiable rasterizer)
    has no training script to measure against. The network builder stays public.
  - Runtime: one stage, resident (4.02 GB float32); no streaming or paging applies. The registration reads
    `NFKModelLoadOptionPrecision` and `NFKModelLoadOptionQuantizationBits` as TripoSR's does;
    `familyProfiles["stable-fast-3d"]` declares both. The remeshers the release offers (`triangle` through
    gpytoolbox, `quad` through pynanoinstantmeshes, both off by default) are third-party libraries the
    backend does not carry.
- `NFKMLXTrellisFactory` (`@objc`) — TRELLIS (Microsoft, MIT), a generator of a 3D object in two rectified-flow
  stages, returned as a vertex-colored mesh and as 3D Gaussians. Four releases: `microsoft/TRELLIS-image-large` at
  `25e0d31f` (image in, conditioned on DINOv2 ViT-L/14 with four registers, `dinov2_vitl14_reg4_pretrain.pth`
  from Meta's server, Apache-2.0) and `microsoft/TRELLIS-text-base` / `-large` / `-xlarge` at `f8e8cf00` /
  `4aad9f4a` / `e0b00432` (prompt in, conditioned on the last hidden states of `openai/clip-vit-large-patch14`
  at `32bd6428`, read by the Stable Diffusion text tower and tokenizer; their `pipeline.json` names the image
  release's decoders). `+register` under `trellis-image-large`, `trellis-text-base`, `trellis-text-large`, and
  `trellis-text-xlarge`. The factories: `backendWithDirectoryURL:encoderWeightsURL:residency:error:`,
  `backendWithVariant:directoryURL:imageReleaseDirectoryURL:textEncoderDirectoryURL:residency:error:`,
  `backendWithDirectoryURL:encoderWeightsURL:structureFlowWeightsURL:latentFlowWeightsURL:residency:error:` (a
  fine-tune's flows in place of the release's), and the download `backendWithVariant:revision:cacheDirectoryURL:residency:`
  with its `completionHandler:` peer. `NFKMLXTrellisPipeline` is the Swift surface (`image(…)`, `text(…)`,
  `random(configuration:)`, the per-stage calls). `NFKMLXTrellisRelease` reads `pipeline.json` and every flow's
  JSON and refuses a positional mode, patch, shared modulation, width split, sampler, or conditioner the port
  does not read; the release geometry comes from those files (`NFKMLXTrellisConfiguration.textBase` and the other
  presets state it). Inputs `NFKInputImage` or `NFKInputPrompt`; outputs `NFKOutputMesh` and
  `NFKOutputGaussianSplats`; parameters `NFKMLXTrellisParameterKey` (steps and guidance per stage, `.outputs`) and
  `NFKParameterSeed` (42, the release's `run` default). The sources: `NFKMLXTrellis.swift` (configuration,
  encoder, structure flow and decoder), `NFKMLXTrellisSparse.swift` (voxel grid, sparse convolution, latent flow,
  decoders), `NFKMLXTrellisMesh.swift` (FlexiCubes), `NFKMLXTrellisPipeline.swift`, `NFKMLXTrellisBackend.swift`,
  and `NFKMLXTrellisTraining.swift`.
  - Image preparation (`preprocess_image`): the alpha's box over alpha > 0.8 · 255, centered, its longer side
    times 1.2 truncated, cropped with PIL's half-to-even rounding of a fractional box (outside the image reads
    transparent black), resized to 518 by Pillow's Lanczos over premultiplied `RGBa` and back
    (`NFKMLXPillowLanczos.resizeRGBA`: MULDIV255 premultiplication, truncating unpremultiplication with alpha 0
    and 255 passing through), then each color times its alpha, truncated to 8 bits. The release runs `rembg`
    on an opaque image; the backend uses an opaque image whole.
  - DINOv2: the hub model's layout (`patch_embed.proj`, `blocks.N.{norm1, attn.qkv, attn.proj, ls1, norm2,
    mlp.fc1, mlp.fc2, ls2}`); class token plus the 37 × 37 position grid, the four registers inserted after the
    class token without positions; the condition is the last block's tokens (`x_prenorm`) normalized without an
    affine transform: 1,374 × 1024. The `.pth` stores each tensor through `torch._tensor._rebuild_from_type_v2`
    with a `_has_been_cloned` attribute, which the native reader reads as the plain tensor.
  - Stage 1: `SparseStructureFlowModel` over the dense 16³ × 8 latent, one token per voxel (patch 1), absolute
    sinusoidal positions (sines then cosines per axis, 170 frequencies an axis, zero-padded to 1024), and 24
    modulated cross blocks: adaLN shift, scale, and gate from SiLU of the timestep embedding (cosines then sines
    of t · 1000 over 128 frequencies), self-attention with per-head RMS-normalized queries and keys (unit
    length times γ · √64), an affine-normalized cross-attention to the condition, tanh-approximated GELU.
    `SparseStructureDecoder`: Conv3d with channel LayerNorm residual blocks at 512, 128, and 32 channels and a
    3D pixel shuffle between them (output voxel (2x + a, 2y + b, 2z + d) reads channel c · 8 + a · 4 + b · 2 +
    d), to 64³ logits; a voxel is occupied where its logit is positive, x-major.
  - Stage 2: `SLatFlowModel` on the occupied voxels: residual blocks with submanifold 3³ convolution at 128
    channels (256 and three blocks a side for text-xlarge), a halving to 32³ before the transformer and its
    reversal after, with skip connections. The convolution is one gather and matrix product over 27 taps
    (`NFKTrellisSparseConvolution`, one product per tap past 2²⁷ gathered elements); a tap reads the input at
    the output's coordinate plus (offset − 1), x-major, as spconv 2.3.8's `calc_subm_conv_indices` pairs them,
    and the weights are stored `[out, 3, 3, 3, in]`. The halving averages each parent's children the way
    torch's `scatter_reduce(..., "mean")` does over the zero-filled output, which counts the zero: sum / (n + 1).
  - Decoders, on the 64³ voxels: 12 swin blocks (window 8, shifted by 4 on odd blocks, windows grouped
    x-major after the shift; `NFKTrellisWindowPartition` pads each window and masks the padding), LayerNorm
    without an affine transform. The Gaussian decoder normalizes and projects to 32 Gaussians a voxel: offsets
    tanh(o + the stored `offset_perturbation`) · 0.5 · 1.5 / 64 around the voxel center, scales softplus(s +
    softplus⁻¹(0.004)) widened by 9e-4 in quadrature, rotations (0.1 · r + (1, 0, 0, 0)) normalized, opacities
    σ(o + logit 0.1). The mesh decoder subdivides twice (GroupNorm of 32 groups, SiLU, the voxel split into its
    eight children x-major, two convolutions, a 1³ convolution skip; 768 → 192 → 96 channels) and projects to
    101 channels a voxel at 256³: eight corner SDFs, eight deformations, 21 FlexiCubes weights, eight 6-channel
    colors.
  - FlexiCubes (`NFKTrellisFlexiCubes`, the CPU, float32): the corner attributes average over the voxels sharing
    a vertex (SDF + −1/256; a vertex no voxel touches reads outside), the vertices deform by
    (1 − 1e-8) / 512 · tanh, the weights β and α read 1 + 0.99 tanh and γ 0.005 + 0.99 σ (zero where no voxel
    covers the cube). The dual marching cubes table is derived at first use (`NFKTrellisDualCubes`): one dual
    vertex per surface patch, crossing edges joined on every face where they share it, an ambiguous face keeping
    its two inside corners apart, edges ascending in a group and groups ordered by the minority-sign corners
    they enclose. A case with one ambiguous face and one dual vertex takes its complement's grouping when the cube
    across that face is such a case too. The derivation reproduces the release's `dmc_table`, `num_vd_table`, and
    `check_table` for all 256 cases (counts and spot cases are in `testTheDerivedDualTableHasTheReleasesCaseCounts`).
    The extraction runs over the surface cubes alone in the reference's order: cubes x-major, dual vertices by
    ascending count then cube, edges merged and sorted, quads from edges shared by four surface cubes (the
    flipped ones first), each split along the larger γ product. Colors are σ of the decoder's, interpolated as
    the vertices are. The mesh turns from the decoders' z up to glTF's frame, (x, y, z) → (x, z, −y), as the
    release's GLB export turns it; the Gaussians take the same quarter turn, their rotations premultiplied by
    (√½, −√½, 0, 0). The export's simplification, hole filling, and texture baking need a Gaussian rasterizer
    (nvdiffrast); the backend returns the mesh's vertex colors instead.
    The extraction leaves a surface open where the reference does. On a 12³ grid of voxels whose corner SDFs are
    (√(x² + y² + z²) − 4) / 12 in float32 about the grid's center (the eight SDFs per voxel, zeros in the other 93
    channels), the release's `SparseFeatures2Mesh(res=12, use_color=True)`, run on the CPU with the oracle's
    `trellis_shims` (2026-11-11), gives 416 vertices and 828 faces, 2,304 of whose 2,394 directed edges meet their
    reverse. `testFlexiCubesClosesASphere` holds the port to those figures.
  - Sampler (`FlowEulerGuidanceIntervalSampler`): the timetable is numpy's `linspace(1, 0, steps + 1)` bent by
    3t / (1 + 2t) in double precision, the flow evaluated at float32 t · 1000, guidance (1 + s) · v − s · v₀ where
    t lies in the interval, x ← x − (t − t_next) · v. The noise comes from MLX's generator, so a seed does not
    reproduce torch's sample.
  - Oracle: `run_reference.py trellis` (the image release at 25 steps on the subject plate with an elliptical
    alpha: every seam above, four flow forwards of each stage plus the last, 8,640 voxels, 247,172 vertices,
    492,012 faces, 276,480 Gaussians), `trellis_text` (each text release through the structured latent on "A
    wooden chair with a red cushion"), and `trellis_objectives`, all in the `sf3d` environment. The release's code
    runs on the CPU at float32 (`use_fp16` forced off) with `Tools/reference-parity/trellis_shims`: spconv's
    `SubMConv3d` as the same gather product (equal to a dense `Conv3d` within 2.1e-14), flash-attn as torch SDPA
    per sequence, `kaolin.check_tensor` and `easydict`, and `rembg`, `open3d`, `plyfile`, `utils3d`, and `cv2` as
    stand-ins that raise when called. FlexiCubes and the mesh extractor take the CPU device.
  - Parity (float32 on the GPU against the reference's float32 CPU run, `NFKMLXTrellisTests`, 2026-11-11):
    the condition 0.9999999999275 (max 3.2e-4); the structure flow 0.99999999999984, …88, …87, and
    0.99999999999799 at its last forward; the occupancy logits 0.9999999999996767 with the 8,640 voxels identical;
    the latent flow 0.99999999999966, …955, …981, and 0.99999999999762; the Gaussian decoder's features
    0.999999999999205 and every field ≥ 0.9999999999743; the mesh decoder's 552,960 × 101 features
    0.9999999999937 in the reference's voxel order; the prepared image byte for byte (804,972 bytes); FlexiCubes on the reference's
    features the reference's mesh (247,172 vertices, identical faces, positions within 3.0e-8, colors exact); both samplers from the recorded noise over 25
    steps 0.99999999998208 (the structure, its 8,640 voxels identical) and 0.99999999999985 (the structured
    latent; 0.99999999999987 denormalized). Text: (run 5, 2026-11-11; the CLIP condition 0.99999999999843 and the empty prompt's 0.99999999999863 for all three, the tokens identical) text-base: structure flow 0.99999999999988, 0.99999999999989, and 0.99999999999860 at its last forward, latent flow 0.99999999999956, 0.99999999999957, 0.99999999999492, samplers over 25 steps 0.99999999998588 and 0.99999999999934; text-large: structure flow 0.99999999999988, 0.99999999999988, 0.99999999999769, latent flow 0.99999999999953, 0.99999999999955, 0.99999999999591, samplers 0.99999999987970 and 0.99999999999927; text-xlarge: structure flow 0.99999999999989, 0.99999999999989, 0.99999999999867, latent flow 0.99999999999970, 0.99999999999970, 0.99999999999828, samplers 0.99999999986941 and 0.99999999999959. The tests hold each float32 cosine above a threshold,
    the convention of the float32 ports (TripoSR, Stable Fast 3D): 0.9999999999 for each network's forward,
    0.999999999 for the 25-step guided sampler chains (which compound each forward's float32 differences; the text
    releases' structure chains read 0.9999999998797 and 0.9999999998694), 0.9999999 for CLIP's condition. These are thresholds, not floors: the reference's own float32 against
    float64 spread is not measured. Bytes, voxels, faces, and the FlexiCubes sphere's counts are held exactly.
  - **Customization: LoRA (or full), both flows.** `NFKMLXTrellisObjective` is the trainers' rectified flow:
    x_t = (1 − t) x₀ + (1e-5 + (1 − 1e-5) t) ε, the target (1 − 1e-5) ε − x₀, mean squared error, times σ of a
    normal draw around 1 (`logitNormal`, mean 1, std 1), the condition replaced by zeros with probability 0.1, AdamW
    at 1e-4 without decay, the gradient clipped to norm 1 (the release's adaptive clipper at its ceiling). The
    release's EMA is not kept. `NFKMLXTrellisTraining.fineTuneStructureFlow` trains on device from image–mesh pairs:
    `occupancy(of:resolution:)` voxelizes a mesh's surface in the release's normalized cube and
    `structureLatent(of:encoder:)` encodes it with `ss_enc_conv3d_16l8_fp16` to the posterior's mean, as
    `encode_ss_latent.py` does. `fineTuneLatentFlow` trains on structured latents the release's toolkit encodes
    offline from 150 rendered views of DINOv2 features. Measured against `trellis_objectives`: the structure encoder's mean 0.99999999999696 and log-variance 0.99999999999837; the dense loss 0.05751151 against the trainer's 0.057511486 and 0.062691584 against 0.06269159 (x_t and the target at cosine 1.0); the sparse loss 5.9853306 against 5.985331 and 1.4090514 against 1.4090514, over 17,111 voxels.
  - Runtime: five stages (conditioner, structure flow, structure decoder, latent flow, decoders) staged or held
    resident (`NFKMLXResidency`; the image release holds 6.86 GB at float32); the two flows quantize to 2–8 bits
    (`NFKModelLoadOptionQuantizationBits`). The release is published at float16 and runs it on CUDA; float16 and
    bfloat16 are not offered until a measured floor holds them. Streaming and paging do not apply: no stage
    exceeds a tier that holds the next.
