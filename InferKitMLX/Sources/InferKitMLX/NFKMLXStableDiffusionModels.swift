//
//  NFKMLXStableDiffusionModels.swift
//  InferKitMLX
//
//  The Stable Diffusion UNet and autoencoder, in the diffusers layout.
//
//  One implementation serves every latent-diffusion model shipped here. The releases differ in scalars
//  a configuration carries — channel widths, which levels attend, the cross-attention width, whether
//  the transformer projects with a convolution or a linear layer, and whether a class embedding joins
//  the timestep — not in structure. Tensors flow NHWC, where the reference is NCHW.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: Configuration

/// The geometry of a released UNet. The three shipped models are three values of this type.
public struct NFKMLXSDUNetConfiguration: Sendable {
    public var inputChannels: Int = 4
    public var outputChannels: Int = 4
    /// The width at each resolution level, coarsening left to right.
    public var blockChannels: [Int] = [320, 640, 1280, 1280]
    public var layersPerBlock: Int = 2
    /// True where that level carries a cross-attention transformer. The reference spells this as its
    /// `down_block_types`, and the level the upscaler leaves plain is its first, not its last.
    public var attends: [Bool] = [true, true, true, false]
    public var crossAttentionDimensions: Int = 768
    /// Attention heads per level. A release states either one count for every level or one per level.
    public var attentionHeads: [Int] = [8, 8, 8, 8]
    /// The reference's `use_linear_projection`: false projects the transformer's ends with a 1×1
    /// convolution, true with a linear layer.
    public var usesLinearProjection: Bool = false
    /// The reference's `only_cross_attention`. Where it is true, the block's FIRST attention is a
    /// cross-attention as well, so its keys and values are the context's width rather than the
    /// feature's. Nothing in the checkpoint's key names says so — only the tensor shapes — and MLX
    /// adopts a checkpoint's shapes wholesale, so getting this wrong loads cleanly and fails later.
    public var onlyCrossAttention: [Bool] = [false, false, false, false]
    /// The reference's `num_class_embeds`. The ×4 upscaler conditions on a noise level through it.
    public var classEmbeddingCount: Int?
    /// The reference's `transformer_layers_per_block`: how many transformer blocks each level's
    /// attention runs. Every release before SDXL runs one everywhere.
    public var transformerLayers: [Int] = [1, 1, 1, 1]
    /// The reference's `addition_embed_type: "text_time"`. SDXL folds a pooled text embedding and a
    /// size-and-crop descriptor into the timestep through it; the other releases carry none.
    public var additionEmbedding: NFKSDAdditionEmbedding?
    /// The reference's `time_cond_proj_dim`: the width of the guidance embedding a latent consistency
    /// model adds to the timestep before the time embedding's first layer. A model that carries it
    /// conditions on the guidance scale and runs no classifier-free guidance. Nil for every release but
    /// the LCM distillations. Introduced in InferKit 0.4.0.
    public var timeConditionProjectionDimensions: Int?
    public var normalizationGroups: Int = 32

    public init() {}

    /// Stable Diffusion 1.5 inpainting: nine input channels — the noisy latent, the masked-image
    /// latent, and the mask.
    public static let inpainting: NFKMLXSDUNetConfiguration = {
        var c = NFKMLXSDUNetConfiguration()
        c.inputChannels = 9
        return c
    }()

    /// Marigold: Stable Diffusion 2 geometry, conditioned on an image latent rather than a mask.
    public static let marigold: NFKMLXSDUNetConfiguration = {
        var c = NFKMLXSDUNetConfiguration()
        c.inputChannels = 8
        c.crossAttentionDimensions = 1024
        c.attentionHeads = [5, 10, 20, 20]
        c.usesLinearProjection = true
        return c
    }()

    /// SDXL: three levels rather than four, ten transformer blocks deep at the coarsest, cross-attending
    /// to two towers concatenated, and conditioned on a pooled embedding and a size descriptor.
    public static let sdxl: NFKMLXSDUNetConfiguration = {
        var c = NFKMLXSDUNetConfiguration()
        c.blockChannels = [320, 640, 1280]
        c.attends = [false, true, true]
        c.attentionHeads = [5, 10, 20]
        c.transformerLayers = [1, 2, 10]
        c.crossAttentionDimensions = 2048
        c.usesLinearProjection = true
        c.onlyCrossAttention = [false, false, false]
        c.additionEmbedding = NFKSDAdditionEmbedding()
        return c
    }()

    /// The ×4 latent upscaler: narrower, its plain level is the first, and a noise level joins the
    /// timestep through a class embedding.
    public static let upscaler: NFKMLXSDUNetConfiguration = {
        var c = NFKMLXSDUNetConfiguration()
        c.inputChannels = 7
        c.blockChannels = [256, 512, 512, 1024]
        c.attends = [false, true, true, true]
        c.crossAttentionDimensions = 1024
        c.usesLinearProjection = true
        c.onlyCrossAttention = [true, true, true, false]
        c.classEmbeddingCount = 1000
        return c
    }()
}

/// The reference's `text_time` conditioning: a pooled text embedding and a size-and-crop descriptor,
/// projected together and added to the timestep embedding.
public struct NFKSDAdditionEmbedding: Sendable {
    /// The width each entry of the descriptor embeds to (`addition_time_embed_dim`).
    public var timeEmbeddingDimensions: Int = 256
    /// The concatenated width the projection takes (`projection_class_embeddings_input_dim`): the
    /// descriptor's entries plus the pooled embedding.
    public var projectionInputDimensions: Int = 2816

    public init(timeEmbeddingDimensions: Int = 256, projectionInputDimensions: Int = 2816) {
        self.timeEmbeddingDimensions = timeEmbeddingDimensions
        self.projectionInputDimensions = projectionInputDimensions
    }
}

/// What SDXL conditions on beside the prompt sequence and the timestep.
public struct NFKSDAddedConditioning {
    /// The second text tower's pooled embedding, `[batch, width]`.
    public let pooled: MLXArray
    /// The reference's `time_ids`, `[batch, 6]`: the original size, the crop's top-left corner, and
    /// the target size.
    public let timeIds: MLXArray

    public init(pooled: MLXArray, timeIds: MLXArray) {
        self.pooled = pooled
        self.timeIds = timeIds
    }
}

/// The geometry of a released autoencoder.
public struct NFKMLXSDVAEConfiguration: Sendable {
    public var latentChannels: Int = 4
    public var blockChannels: [Int] = [128, 256, 512, 512]
    /// The decoder's widths where they differ from the encoder's, the reference's
    /// `decoder_block_out_channels`. Nil means the decoder mirrors the encoder, which every release
    /// before FLUX.2's small decoder does.
    public var decoderBlockChannels: [Int]?
    public var layersPerBlock: Int = 2
    public var normalizationGroups: Int = 32
    /// Multiplies an encoded latent (and divides before decoding) so its variance suits the UNet —
    /// the diffusers `scaling_factor`, which each release trains with.
    public var scaleFactor: Float = 0.18215
    /// Added before decoding (and subtracted after encoding) — the diffusers `shift_factor`. The
    /// Flux-family autoencoders center their latents; the Stable Diffusion ones leave it at zero.
    public var shiftFactor: Float = 0.0
    /// Whether the `1×1` `quant_conv` / `post_quant_conv` are present. Stable Diffusion keeps them; the
    /// Flux-family autoencoders drop both, so the encoder's `conv_out` is the moments directly.
    public var useQuantConv: Bool = true

    public init() {}

    public static let stableDiffusion = NFKMLXSDVAEConfiguration()

    /// The upscaler's autoencoder is one level shallower and scales its latents differently.
    public static let upscaler: NFKMLXSDVAEConfiguration = {
        var c = NFKMLXSDVAEConfiguration()
        c.blockChannels = [128, 256, 512]
        c.scaleFactor = 0.08333
        return c
    }()

    /// FLUX.2's SMALL DECODER (`black-forest-labs/FLUX.2-small-decoder`): the same encoder at a
    /// narrower decoder, which is what `decoder_block_out_channels` expresses. 62M parameters against
    /// the full autoencoder's, for a cheaper decode at the same latent.
    public static let flux2SmallDecoder: NFKMLXSDVAEConfiguration = {
        var c = NFKMLXSDVAEConfiguration.flux2
        c.decoderBlockChannels = [96, 192, 384, 384]
        return c
    }()

    /// FLUX.2's autoencoder: 32 latent channels, the quantization convolutions kept, and no scalar
    /// scale or shift. FLUX.2 whitens its latent with a BatchNorm's running statistics after a 2x2
    /// patching instead, which ``NFKMLXFlux2LatentCodec`` carries. Architecturally the same
    /// `AutoencoderKL` as Stable Diffusion's.
    public static let flux2: NFKMLXSDVAEConfiguration = {
        var c = NFKMLXSDVAEConfiguration()
        c.latentChannels = 32
        c.scaleFactor = 1
        c.shiftFactor = 0
        c.useQuantConv = true
        return c
    }()

    /// The Flux autoencoder Z-Image encodes into: 16 latent channels, no quant convolutions, and a
    /// centering shift. Architecturally the same `AutoencoderKL` as Stable Diffusion's.
    public static let flux: NFKMLXSDVAEConfiguration = {
        var c = NFKMLXSDVAEConfiguration()
        c.latentChannels = 16
        c.scaleFactor = 0.3611
        c.shiftFactor = 0.1159
        c.useQuantConv = false
        return c
    }()
}

// MARK: Shared pieces

/// `nn.GroupNorm(groups, channels)` over a channels-last tensor.
///
/// MLX ships a `GroupNorm`, but the reference's two epsilons differ by layer — a resnet normalizes at
/// `1e-5` and a transformer's input norm at `1e-6` — so the value is explicit at every call site.
final class NFKSDGroupNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    let groups: Int
    let eps: Float

    init(groups: Int, channels: Int, eps: Float) {
        self.groups = groups
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
    }

    /// On a half-precision input the statistics, the normalization, and the affine run in float32 and the
    /// result rounds once, as torch's `group_norm` computes it.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else { return normalized(x) }
        return normalized(x.asType(.float32)).asType(x.dtype)
    }

    /// The statistics form over each group's values in one contiguous row; the normalization and the affine
    /// apply in `x`'s own layout, so no transposed copy of the whole tensor outlives the statistics.
    private func normalized(_ x: MLXArray) -> MLXArray {
        let shape = x.shape                                     // [batch, height, width, channel]
        let channels = shape[3]
        let perGroup = channels / groups
        // The reference groups consecutive channels and normalizes each group over its channels and
        // the whole spatial extent.
        let spatial = x.reshaped([shape[0], shape[1] * shape[2], groups, perGroup])
        let grouped = spatial.transposed(0, 2, 1, 3).reshaped([shape[0], groups, shape[1] * shape[2] * perGroup])
        let mean = grouped.mean(axis: 2, keepDims: true)
        let centered = grouped - mean
        let variance = (centered * centered).mean(axis: 2, keepDims: true)
        let normalized = (spatial - mean.reshaped([shape[0], 1, groups, 1]))
            * rsqrt(variance + eps).reshaped([shape[0], 1, groups, 1])
        return normalized.reshaped(shape) * weight.asType(x.dtype) + bias.asType(x.dtype)
    }
}

/// `get_timestep_embedding`: half the channels are cosines and half sines, cosines first, over
/// frequencies spaced logarithmically to `maxPeriod`.
enum NFKSDTimesteps {
    static func embedding(_ timesteps: MLXArray, channels: Int, maxPeriod: Float = 10_000) -> MLXArray {
        let half = channels / 2
        let exponents = (0 ..< half).map { -logf(maxPeriod) * Float($0) / Float(half) }
        let scale = NFKMLXHalfPrecision.arithmetic == .reference
            ? NFKReferenceRounding.exponential(MLXArray(exponents, [1, half]))
            : MLXArray(exponents.map { expf($0) }, [1, half])
        let angles = timesteps.reshaped([-1, 1]).asType(.float32) * scale
        // `flip_sin_to_cos` is true for every released Stable Diffusion UNet.
        return concatenated([cos(angles), sin(angles)], axis: 1)
    }

    /// The pipeline's `get_guidance_scale_embedding`: the guidance scale less one, times 1000, embedded
    /// sines first over frequencies spaced to `maxPeriod` with the end point included (`half - 1`),
    /// which is the opposite order and a different spacing from the timestep embedding.
    static func guidanceEmbedding(_ guidanceScale: Float, channels: Int, maxPeriod: Float = 10_000) -> MLXArray {
        let half = channels / 2
        // The reference forms one step, `-log(maxPeriod) / (half - 1)`, and multiplies it by the index.
        let step = -logf(maxPeriod) / Float(half - 1)
        let exponents = (0 ..< half).map { Float($0) * step }
        let scale = NFKMLXHalfPrecision.arithmetic == .reference
            ? NFKReferenceRounding.exponential(MLXArray(exponents, [1, half]))
            : MLXArray(exponents.map { expf($0) }, [1, half])
        let angles = MLXArray([(guidanceScale - 1) * 1000]).reshaped([1, 1]) * scale
        let embedding = concatenated([sin(angles), cos(angles)], axis: 1)
        guard channels % 2 == 1 else { return embedding }
        return concatenated([embedding, MLXArray.zeros([1, 1])], axis: 1)
    }
}

/// `TimestepEmbedding`: two linear layers with a SiLU between them, and for a latent consistency model
/// a projection of the guidance embedding added to the input first.
final class NFKSDTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "cond_proj") var conditionProjection: Linear?

    init(inputChannels: Int, timeChannels: Int, conditionChannels: Int? = nil) {
        self._linear1.wrappedValue = Linear(inputChannels, timeChannels)
        self._linear2.wrappedValue = Linear(timeChannels, timeChannels)
        self._conditionProjection.wrappedValue = conditionChannels.map { Linear($0, inputChannels, bias: false) }
    }

    func callAsFunction(_ x: MLXArray, condition: MLXArray? = nil) -> MLXArray {
        var input = x
        if let conditionProjection, let condition {
            input = input + conditionProjection(condition.asType(x.dtype))
        }
        return linear2(NFKReferenceRounding.silu(linear1(input)))
    }
}

/// Whether an autoencoder's decode runs as one graph or evaluates each large intermediate before the next step
/// reads it. The two give the same bits.
///
/// Measured under ``NFKMLXGPU/applyStandingLimits()`` for FLUX.1's decoder at 1024 × 1024 in bfloat16:
/// - one graph → 3.32 s, 13.5 GB beyond the weights.
/// - step by step → 3.93 s, 8.4 GB.
enum NFKSDDecodeStaging {
    /// The size from which a staged decode evaluates an intermediate: 256 MiB.
    static let stagedEvaluationBytes = 1 << 28

    /// The evaluation size in force for the decode in progress, or nil when it runs as one graph.
    @TaskLocal static var evaluationBytes: Int?

    /// `x`, evaluated first when the decode in progress is staged and `x` holds at least the evaluation size.
    static func evaluated(_ x: MLXArray) -> MLXArray {
        if let evaluationBytes, x.nbytes >= evaluationBytes {
            eval(x)
        }
        return x
    }

    /// The one-graph decode's peak beyond its weights, in bytes: 1.4 GB plus 50.4 bytes per element of the
    /// widest map the decode holds at the output size, for each image.
    ///
    /// @discussion Measured under the standing limits for the SD, FLUX.1, FLUX.2, and FLUX.2 small decoders at
    /// 512 × 512 to 1024 × 1024, in float16, bfloat16, and float32, the peak runs 46 to 70 bytes per element of
    /// that map. The estimate sits 0.06 to 2.5 GB above each measured peak.
    static func estimatedWholeGraphPeak(images: Int, outputPixels: Int, widestOutputChannels: Int) -> Int {
        images * (1_400_000_000 + outputPixels * widestOutputChannels * 504 / 10)
    }

    /// The memory a decode can take before MLX frees its cache or Metal starts evicting: the smaller of
    /// Metal's recommended working set and MLX's memory limit, less the memory already active.
    static var budget: Int {
        min(NFKMLXGPU.recommendedWorkingSetSize, NFKMLXGPU.memoryLimit) - NFKMLXGPU.activeMemory
    }

    /// The evaluation size for a decode whose one-graph peak is `estimatedPeak` against `budget`: nil while
    /// the peak fits, ``stagedEvaluationBytes`` past it.
    static func evaluationBytes(estimatedPeak: Int, budget: Int) -> Int? {
        estimatedPeak <= budget ? nil : stagedEvaluationBytes
    }
}

/// `ResnetBlock2D`: two normalized convolutions with the timestep embedding added between them, over a
/// residual that projects when the width changes.
final class NFKSDResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: NFKSDGroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "time_emb_proj") var timeProjection: Linear?
    @ModuleInfo(key: "norm2") var norm2: NFKSDGroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var shortcut: Conv2d?

    /// `eps` is the UNet's `1e-5` by default. The autoencoder's reference passes `resnet_eps=1e-6`
    /// to every block it builds, and the two normalizations are otherwise identical, so the value is
    /// the caller's rather than a constant here.
    init(inputChannels: Int, outputChannels: Int, timeChannels: Int?, groups: Int,
         eps: Float = 1e-5) {
        self._norm1.wrappedValue = NFKSDGroupNorm(groups: groups, channels: inputChannels, eps: eps)
        self._conv1.wrappedValue = NFKConv2d(inputChannels: inputChannels, outputChannels: outputChannels,
                                             kernelSize: 3, padding: 1)
        self._timeProjection.wrappedValue = timeChannels.map { Linear($0, outputChannels) }
        self._norm2.wrappedValue = NFKSDGroupNorm(groups: groups, channels: outputChannels, eps: eps)
        self._conv2.wrappedValue = NFKConv2d(inputChannels: outputChannels, outputChannels: outputChannels,
                                             kernelSize: 3, padding: 1)
        self._shortcut.wrappedValue = inputChannels == outputChannels
            ? nil
            : NFKConv2d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray, time: MLXArray?) -> MLXArray {
        let staged = NFKSDDecodeStaging.evaluated
        var h = staged(conv1(staged(NFKReferenceRounding.silu(norm1(x)))))
        if let timeProjection, let time {
            h = h + timeProjection(NFKReferenceRounding.silu(time)).reshaped([time.shape[0], 1, 1, -1])
        }
        h = staged(conv2(staged(NFKReferenceRounding.silu(norm2(h)))))
        return (shortcut.map { staged($0(x)) } ?? x) + h
    }
}

/// The gated feed-forward the reference stores as `ff.net.0.proj` and `ff.net.2`: one projection to
/// twice the inner width, split into a value and a gate.
final class NFKSDFeedForward: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(dimensions: Int, multiplier: Int = 4) {
        let inner = dimensions * multiplier
        self._proj.wrappedValue = Linear(dimensions, inner * 2)
        self._out.wrappedValue = Linear(inner, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = proj(x)
        let inner = projected.shape[projected.ndim - 1] / 2
        let value = projected[.ellipsis, 0 ..< inner]
        let gate = projected[.ellipsis, inner ..< (2 * inner)]
        return out(value * NFKReferenceRounding.gelu(gate))
    }
}

/// The transformer's attention. Queries, keys, and values carry no bias; only the output projection
/// The IP-Adapter conditioning the UNet threads to its cross-attention: the projected image tokens and
/// the scale the image-conditioned attention is blended in at. A run with no adapter passes `nil`, so
/// the cross-attention is the text one unchanged.
public struct NFKSDImageConditioning {
    public let tokens: MLXArray
    public let scale: Float
    public init(tokens: MLXArray, scale: Float) {
        self.tokens = tokens
        self.scale = scale
    }
}

/// does. Passing `context` makes it cross-attention.
final class NFKSDAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    // The IP-Adapter key/value projections exist only on a cross-attention an adapter is attached to;
    // a base UNet checkpoint carries no such weights, so they stay nil and out of `parameters()`.
    @ModuleInfo(key: "to_k_ip") var toKIP: Linear?
    @ModuleInfo(key: "to_v_ip") var toVIP: Linear?

    let heads: Int
    let dimensions: Int
    let contextDimensions: Int
    /// The release's `upcast_attention`, which diffusers' classic processor honors.
    var upcasts = false

    init(dimensions: Int, contextDimensions: Int, heads: Int) {
        self.heads = heads
        self.dimensions = dimensions
        self.contextDimensions = contextDimensions
        self._toQ.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._toK.wrappedValue = Linear(contextDimensions, dimensions, bias: false)
        self._toV.wrappedValue = Linear(contextDimensions, dimensions, bias: false)
        self._toOut.wrappedValue = Linear(dimensions, dimensions)
    }

    /// Adds the IP-Adapter image key/value projections, loaded from the adapter's weights, so this
    /// cross-attention blends an image-conditioned attention beside the text one. They read the context
    /// width and write the query width, matching the text `to_k`/`to_v`. The projections are attached
    /// through `update(modules:)`, the sanctioned way to register a `@ModuleInfo` child after init.
    func attachIPAdapter(keyWeight: MLXArray, valueWeight: MLXArray) throws {
        let toK = Linear(contextDimensions, dimensions, bias: false)
        let toV = Linear(contextDimensions, dimensions, bias: false)
        toK.update(parameters: ModuleParameters.unflattened([("weight", keyWeight)]))
        toV.update(parameters: ModuleParameters.unflattened([("weight", valueWeight)]))
        try update(modules: ModuleChildren.unflattened([("to_k_ip", toK), ("to_v_ip", toV)]),
                   verify: .none)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil,
                        ip: NFKSDImageConditioning? = nil) -> MLXArray {
        let source = context ?? x
        let batch = x.shape[0], dimensions = x.shape[2]
        let headDimensions = dimensions / heads
        func split(_ value: MLXArray) -> MLXArray {
            value.reshaped([value.shape[0], value.shape[1], heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        let queries = split(toQ(x))
        var out = upcasts
            ? NFKReferenceRounding.upcastAttention(queries: queries, keys: split(toK(source)), values: split(toV(source)),
                                                   scale: 1 / sqrt(Float(headDimensions)))
            : NFKReferenceRounding.flashAttention(queries: queries, keys: split(toK(source)), values: split(toV(source)),
                                                  scale: 1 / sqrt(Float(headDimensions)), mask: nil)
        // The image-conditioned attention shares the query and reads its own projections of the image
        // tokens, added at the adapter's scale — the IP-Adapter's decoupled cross-attention.
        if let ip, let toKIP, let toVIP {
            let tokens = ip.tokens.shape[0] == batch ? ip.tokens
                : broadcast(ip.tokens, to: [batch, ip.tokens.shape[1], ip.tokens.shape[2]])
            let imageAttention = NFKReferenceRounding.flashAttention(
                queries: queries, keys: split(toKIP(tokens)), values: split(toVIP(tokens)),
                scale: 1 / sqrt(Float(headDimensions)), mask: nil)
            out = out + NFKReferenceRounding.scaled(imageAttention, by: ip.scale)
        }
        return toOut(out.transposed(0, 2, 1, 3).reshaped([batch, x.shape[1], dimensions]))
    }
}

/// `BasicTransformerBlock`: pre-norm self-attention, pre-norm cross-attention, pre-norm feed-forward.
final class NFKSDTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn1") var attn1: NFKSDAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn2") var attn2: NFKSDAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff") var ff: NFKSDFeedForward

    let onlyCrossAttention: Bool

    init(dimensions: Int, contextDimensions: Int, heads: Int, onlyCrossAttention: Bool) {
        self.onlyCrossAttention = onlyCrossAttention
        self._norm1.wrappedValue = NFKLayerNorm(dimensions: dimensions)
        self._attn1.wrappedValue = NFKSDAttention(
            dimensions: dimensions,
            contextDimensions: onlyCrossAttention ? contextDimensions : dimensions, heads: heads)
        self._norm2.wrappedValue = NFKLayerNorm(dimensions: dimensions)
        self._attn2.wrappedValue = NFKSDAttention(dimensions: dimensions,
                                                  contextDimensions: contextDimensions, heads: heads)
        self._norm3.wrappedValue = NFKLayerNorm(dimensions: dimensions)
        self._ff.wrappedValue = NFKSDFeedForward(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray?,
                        ip: NFKSDImageConditioning? = nil) -> MLXArray {
        var out = x + attn1(norm1(x), context: onlyCrossAttention ? context : nil)
        out = out + attn2(norm2(out), context: context, ip: ip)
        return out + ff(norm3(out))
    }
}

/// `Transformer2DModel`: normalize, project into tokens, run the transformer blocks, project back, and
/// add the whole thing to what came in.
final class NFKSDTransformer2D: Module {
    @ModuleInfo(key: "norm") var norm: NFKSDGroupNorm
    @ModuleInfo(key: "proj_in_conv") var projInConv: Conv2d?
    @ModuleInfo(key: "proj_out_conv") var projOutConv: Conv2d?
    @ModuleInfo(key: "proj_in_linear") var projInLinear: Linear?
    @ModuleInfo(key: "proj_out_linear") var projOutLinear: Linear?
    @ModuleInfo(key: "transformer_blocks") var blocks: [NFKSDTransformerBlock]

    init(channels: Int, contextDimensions: Int, heads: Int, depth: Int, groups: Int,
         usesLinearProjection: Bool, onlyCrossAttention: Bool = false) {
        // The reference's input norm is the one place its epsilon is 1e-6 rather than 1e-5.
        self._norm.wrappedValue = NFKSDGroupNorm(groups: groups, channels: channels, eps: 1e-6)
        if usesLinearProjection {
            self._projInLinear.wrappedValue = Linear(channels, channels)
            self._projOutLinear.wrappedValue = Linear(channels, channels)
        } else {
            self._projInConv.wrappedValue = NFKConv2d(inputChannels: channels, outputChannels: channels,
                                                      kernelSize: 1)
            self._projOutConv.wrappedValue = NFKConv2d(inputChannels: channels, outputChannels: channels,
                                                       kernelSize: 1)
        }
        self._blocks.wrappedValue = (0 ..< depth).map { _ in
            NFKSDTransformerBlock(dimensions: channels, contextDimensions: contextDimensions,
                                  heads: heads, onlyCrossAttention: onlyCrossAttention)
        }
    }

    /// The linear projection into tokens. diffusers flattens the normalized map channels-first through a
    /// permute that leaves a strided view, and torch's half-precision `linear` on a strided input rounds
    /// the product and then the bias.
    func projectedIn(_ tokens: MLXArray, _ projection: Linear) -> MLXArray {
        guard NFKReferenceRounding.isReduced(tokens), let bias = projection.bias else { return projection(tokens) }
        return matmul(tokens, projection.weight.T) + bias
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray?,
                        ip: NFKSDImageConditioning? = nil) -> MLXArray {
        let shape = x.shape
        var h = norm(x)
        // A convolution projects before the tokens are flattened; a linear layer projects after.
        if let projInConv { h = projInConv(h) }
        h = h.reshaped([shape[0], shape[1] * shape[2], shape[3]])
        if let projInLinear { h = projectedIn(h, projInLinear) }
        for block in blocks { h = block(h, context: context, ip: ip) }
        if let projOutLinear { h = projOutLinear(h) }
        h = h.reshaped(shape)
        if let projOutConv { h = projOutConv(h) }
        return h + x
    }
}

/// `Downsample2D` with the reference's padding. A UNet pads its convolution symmetrically; the
/// autoencoder pads zero and prepends an explicit right-and-bottom row instead.
final class NFKSDDownsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    let padsAsymmetrically: Bool

    init(channels: Int, padsAsymmetrically: Bool) {
        self.padsAsymmetrically = padsAsymmetrically
        self._conv.wrappedValue = NFKConv2d(inputChannels: channels, outputChannels: channels,
                                            kernelSize: 3, stride: 2,
                                            padding: padsAsymmetrically ? 0 : 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard padsAsymmetrically else { return conv(x) }
        return conv(padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))]))
    }
}

/// `Upsample2D`: nearest ×2 then a 3×3 convolution.
final class NFKSDUpsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        self._conv.wrappedValue = NFKConv2d(inputChannels: channels, outputChannels: channels,
                                            kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(NFKSDDecodeStaging.evaluated(NFKMLXResample.upsampleNearest(x, scale: 2)))
    }
}

// MARK: UNet blocks

final class NFKSDDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKSDResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [NFKSDTransformer2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [NFKSDDownsample]

    init(inputChannels: Int, outputChannels: Int, timeChannels: Int, layers: Int, groups: Int,
         attention: (context: Int, heads: Int, onlyCross: Bool, depth: Int)?, downsamples: Bool,
         usesLinearProjection: Bool) {
        self._resnets.wrappedValue = (0 ..< layers).map { index in
            NFKSDResnetBlock(inputChannels: index == 0 ? inputChannels : outputChannels,
                             outputChannels: outputChannels, timeChannels: timeChannels, groups: groups)
        }
        self._attentions.wrappedValue = attention.map { attention in
            (0 ..< layers).map { _ in
                NFKSDTransformer2D(channels: outputChannels, contextDimensions: attention.context,
                                   heads: attention.heads, depth: attention.depth, groups: groups,
                                   usesLinearProjection: usesLinearProjection,
                                   onlyCrossAttention: attention.onlyCross)
            }
        } ?? []
        self._downsamplers.wrappedValue = downsamples
            ? [NFKSDDownsample(channels: outputChannels, padsAsymmetrically: false)]
            : []
    }

    /// Returns the block's output together with every tensor the decoder will consume as a skip.
    func callAsFunction(_ x: MLXArray, time: MLXArray, context: MLXArray?,
                        ip: NFKSDImageConditioning? = nil) -> (MLXArray, [MLXArray]) {
        var h = x
        var skips = [MLXArray]()
        for (index, resnet) in resnets.enumerated() {
            h = resnet(h, time: time)
            if index < attentions.count { h = attentions[index](h, context: context, ip: ip) }
            skips.append(h)
        }
        for downsampler in downsamplers {
            h = downsampler(h)
            skips.append(h)
        }
        return (h, skips)
    }
}

final class NFKSDUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKSDResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [NFKSDTransformer2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [NFKSDUpsample]

    init(inputChannels: Int, previousChannels: Int, outputChannels: Int, timeChannels: Int,
         layers: Int, groups: Int, attention: (context: Int, heads: Int, onlyCross: Bool, depth: Int)?,
         upsamples: Bool, usesLinearProjection: Bool) {
        self._resnets.wrappedValue = (0 ..< layers).map { index in
            // Each layer consumes one skip, and the last one takes the shallower level's.
            let skip = index == layers - 1 ? inputChannels : outputChannels
            let from = index == 0 ? previousChannels : outputChannels
            return NFKSDResnetBlock(inputChannels: from + skip, outputChannels: outputChannels,
                                    timeChannels: timeChannels, groups: groups)
        }
        self._attentions.wrappedValue = attention.map { attention in
            (0 ..< layers).map { _ in
                NFKSDTransformer2D(channels: outputChannels, contextDimensions: attention.context,
                                   heads: attention.heads, depth: attention.depth, groups: groups,
                                   usesLinearProjection: usesLinearProjection,
                                   onlyCrossAttention: attention.onlyCross)
            }
        } ?? []
        self._upsamplers.wrappedValue = upsamples ? [NFKSDUpsample(channels: outputChannels)] : []
    }

    func callAsFunction(_ x: MLXArray, skips: inout [MLXArray], time: MLXArray,
                        context: MLXArray?, ip: NFKSDImageConditioning? = nil) -> MLXArray {
        var h = x
        for (index, resnet) in resnets.enumerated() {
            h = resnet(concatenated([h, skips.removeLast()], axis: 3), time: time)
            if index < attentions.count { h = attentions[index](h, context: context, ip: ip) }
        }
        for upsampler in upsamplers { h = upsampler(h) }
        return h
    }
}

final class NFKSDMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKSDResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [NFKSDTransformer2D]

    init(channels: Int, timeChannels: Int, groups: Int, context: Int, heads: Int, depth: Int,
         usesLinearProjection: Bool) {
        self._resnets.wrappedValue = (0 ..< 2).map { _ in
            NFKSDResnetBlock(inputChannels: channels, outputChannels: channels,
                             timeChannels: timeChannels, groups: groups)
        }
        self._attentions.wrappedValue = [
            NFKSDTransformer2D(channels: channels, contextDimensions: context, heads: heads,
                               depth: depth, groups: groups, usesLinearProjection: usesLinearProjection)
        ]
    }

    func callAsFunction(_ x: MLXArray, time: MLXArray, context: MLXArray?,
                        ip: NFKSDImageConditioning? = nil) -> MLXArray {
        var h = resnets[0](x, time: time)
        h = attentions[0](h, context: context, ip: ip)
        return resnets[1](h, time: time)
    }
}

// MARK: The UNet

/// `UNet2DConditionModel` — the denoising network every latent-diffusion model here runs.
public final class NFKMLXSDUNet: Module {
    @ModuleInfo(key: "time_embedding") var timeEmbedding: NFKSDTimestepEmbedding
    @ModuleInfo(key: "class_embedding") var classEmbedding: Embedding?
    @ModuleInfo(key: "add_embedding") var addEmbedding: NFKSDTimestepEmbedding?
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [NFKSDDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: NFKSDMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [NFKSDUpBlock]
    @ModuleInfo(key: "conv_norm_out") var normOut: NFKSDGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public let configuration: NFKMLXSDUNetConfiguration

    public init(configuration: NFKMLXSDUNetConfiguration = .inpainting) {
        self.configuration = configuration
        let c = configuration
        let levels = c.blockChannels.count
        let timeChannels = c.blockChannels[0] * 4

        self._timeEmbedding.wrappedValue = NFKSDTimestepEmbedding(
            inputChannels: c.blockChannels[0], timeChannels: timeChannels,
            conditionChannels: c.timeConditionProjectionDimensions)
        self._classEmbedding.wrappedValue = c.classEmbeddingCount.map {
            Embedding(embeddingCount: $0, dimensions: timeChannels)
        }
        self._addEmbedding.wrappedValue = c.additionEmbedding.map {
            NFKSDTimestepEmbedding(inputChannels: $0.projectionInputDimensions, timeChannels: timeChannels)
        }
        self._convIn.wrappedValue = NFKConv2d(inputChannels: c.inputChannels,
                                              outputChannels: c.blockChannels[0],
                                              kernelSize: 3, padding: 1)

        var down = [NFKSDDownBlock]()
        for level in 0 ..< levels {
            down.append(NFKSDDownBlock(
                inputChannels: level == 0 ? c.blockChannels[0] : c.blockChannels[level - 1],
                outputChannels: c.blockChannels[level], timeChannels: timeChannels,
                layers: c.layersPerBlock, groups: c.normalizationGroups,
                attention: c.attends[level]
                    ? (c.crossAttentionDimensions, c.attentionHeads[level], c.onlyCrossAttention[level],
                       c.transformerLayers[level])
                    : nil,
                downsamples: level < levels - 1, usesLinearProjection: c.usesLinearProjection))
        }
        self._downBlocks.wrappedValue = down

        self._midBlock.wrappedValue = NFKSDMidBlock(
            channels: c.blockChannels[levels - 1], timeChannels: timeChannels,
            groups: c.normalizationGroups, context: c.crossAttentionDimensions,
            heads: c.attentionHeads[levels - 1], depth: c.transformerLayers[levels - 1],
            usesLinearProjection: c.usesLinearProjection)

        let reversedChannels = c.blockChannels.reversed().map { $0 }
        let reversedAttends = c.attends.reversed().map { $0 }
        let reversedHeads = c.attentionHeads.reversed().map { $0 }
        let reversedOnlyCross = c.onlyCrossAttention.reversed().map { $0 }
        let reversedDepths = c.transformerLayers.reversed().map { $0 }
        var up = [NFKSDUpBlock]()
        for level in 0 ..< levels {
            let output = reversedChannels[level]
            let previous = level == 0 ? reversedChannels[0] : reversedChannels[level - 1]
            let shallower = reversedChannels[min(level + 1, levels - 1)]
            up.append(NFKSDUpBlock(
                inputChannels: shallower, previousChannels: previous, outputChannels: output,
                timeChannels: timeChannels, layers: c.layersPerBlock + 1,
                groups: c.normalizationGroups,
                attention: reversedAttends[level]
                    ? (c.crossAttentionDimensions, reversedHeads[level], reversedOnlyCross[level],
                       reversedDepths[level])
                    : nil,
                upsamples: level < levels - 1, usesLinearProjection: c.usesLinearProjection))
        }
        self._upBlocks.wrappedValue = up

        self._normOut.wrappedValue = NFKSDGroupNorm(groups: c.normalizationGroups,
                                                    channels: c.blockChannels[0], eps: 1e-5)
        self._convOut.wrappedValue = NFKConv2d(inputChannels: c.blockChannels[0],
                                               outputChannels: c.outputChannels,
                                               kernelSize: 3, padding: 1)
    }

    /// Whether every attention computes its scores in float32, as a release that sets `upcast_attention`
    /// asks. ``NFKMLXStableDiffusionModels/loadUNetWeights(into:from:precision:)`` reads it from the release.
    var upcastsAttention: Bool {
        get { modules().contains { ($0 as? NFKSDAttention)?.upcasts == true } }
        set {
            for case let attention as NFKSDAttention in modules() {
                attention.upcasts = newValue
            }
        }
    }

    /// The cross-attention layers, in the diffusers `attn_processors` order — down blocks, then up
    /// blocks, then the mid block, because `UNet2DConditionModel` registers the mid block last. This is
    /// the order an IP-Adapter's per-layer key/value weights are stored in.
    var crossAttentions: [NFKSDAttention] {
        var attentions = [NFKSDAttention]()
        func collect(_ transformers: [NFKSDTransformer2D]) {
            for transformer in transformers {
                for block in transformer.blocks { attentions.append(block.attn2) }
            }
        }
        for block in downBlocks { collect(block.attentions) }
        for block in upBlocks { collect(block.attentions) }
        collect(midBlock.attentions)
        return attentions
    }

    /// The guidance embedding a latent consistency model conditions on, `[1, channels]`: the pipeline's
    /// `get_guidance_scale_embedding` of `guidanceScale - 1`, sines first. `channels` is the model's
    /// ``NFKMLXSDUNetConfiguration/timeConditionProjectionDimensions``. Introduced in InferKit 0.4.0.
    public static func guidanceEmbedding(scale guidanceScale: Float, channels: Int) -> MLXArray {
        NFKSDTimesteps.guidanceEmbedding(guidanceScale, channels: channels)
    }

    /// - Parameters:
    ///   - x: the latent, `[batch, height, width, inputChannels]`.
    ///   - timestep: the diffusion step.
    ///   - context: the text (or other) conditioning, `[batch, tokens, crossAttentionDimensions]`.
    ///   - classLabel: the noise level, for a model that carries a class embedding.
    ///   - added: the pooled embedding and size descriptor, for a model that carries a `text_time`
    ///     addition embedding.
    ///   - imageConditioning: an IP-Adapter's image tokens and blend, for a UNet with the adapter attached.
    ///   - guidanceEmbedding: the guidance-scale embedding `[batch, timeConditionProjectionDimensions]`
    ///     (``guidanceEmbedding(scale:channels:)``), for a latent consistency model that conditions on
    ///     its guidance scale. A model without the projection ignores it.
    public func callAsFunction(_ x: MLXArray, timestep: MLXArray, context: MLXArray?,
                               classLabel: MLXArray? = nil,
                               added: NFKSDAddedConditioning? = nil,
                               imageConditioning: NFKSDImageConditioning? = nil,
                               guidanceEmbedding: MLXArray? = nil) -> MLXArray {
        // The float32 sinusoids take the sample's type before the embedding, as diffusers casts them; a
        // float32 table would promote the time path, and through every resnet the whole network.
        var time = timeEmbedding(NFKSDTimesteps.embedding(timestep, channels: configuration.blockChannels[0])
            .asType(x.dtype), condition: guidanceEmbedding)
        if let classEmbedding, let classLabel {
            time = time + classEmbedding(classLabel).asType(x.dtype)
        }
        if let addEmbedding, let added, let addition = configuration.additionEmbedding {
            // Every entry of the descriptor embeds separately and the results run together, so six
            // numbers become one vector the width of six timestep embeddings.
            let batch = added.pooled.shape[0]
            let descriptor = NFKSDTimesteps.embedding(added.timeIds.reshaped([-1]),
                                                      channels: addition.timeEmbeddingDimensions)
                .reshaped([batch, -1])
            let joined = concatenated([added.pooled.asType(.float32), descriptor], axis: -1).asType(time.dtype)
            time = time + addEmbedding(joined)
        }

        var h = convIn(x)
        var skips = [h]
        for block in downBlocks {
            let (out, produced) = block(h, time: time, context: context, ip: imageConditioning)
            h = out
            skips += produced
        }
        h = midBlock(h, time: time, context: context, ip: imageConditioning)
        for block in upBlocks {
            h = block(h, skips: &skips, time: time, context: context, ip: imageConditioning)
        }
        return convOut(NFKReferenceRounding.silu(normOut(h)))
    }
}

// MARK: The autoencoder

/// A resnet block of the autoencoder: the UNet's, without the timestep.
final class NFKSDVAEResnet: Module {
    @ModuleInfo(key: "block") var block: NFKSDResnetBlock

    init(inputChannels: Int, outputChannels: Int, groups: Int) {
        // diffusers builds every autoencoder block with `resnet_eps=1e-6`, where the UNet's blocks
        // take 1e-5. The difference is invisible at the released widths and shows at a small one.
        self._block.wrappedValue = NFKSDResnetBlock(inputChannels: inputChannels,
                                                    outputChannels: outputChannels,
                                                    timeChannels: nil, groups: groups, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block(x, time: nil) }
}

/// The autoencoder's bottleneck attention: single-head, over the flattened spatial grid, with biases
/// on every projection.
final class NFKSDVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: NFKSDGroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(channels: Int, groups: Int) {
        self._groupNorm.wrappedValue = NFKSDGroupNorm(groups: groups, channels: channels, eps: 1e-6)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = Linear(channels, channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let tokens = groupNorm(x).reshaped([shape[0], shape[1] * shape[2], shape[3]])
        // diffusers hands the projections the normalized map transposed in place. torch's half-precision
        // `linear` on such a strided input rounds the product and then the bias, where on a contiguous one
        // (`to_out`'s) it rounds once.
        func projected(_ layer: Linear) -> MLXArray {
            guard NFKReferenceRounding.isReduced(tokens), let bias = layer.bias else { return layer(tokens) }
            return matmul(tokens, layer.weight.T) + bias
        }
        // At half precision diffusers' `scaled_dot_product_attention` takes torch's CPU flash kernel,
        // which `flashAttention` follows; a float32 input takes MLX's fused kernel either way.
        let out = NFKReferenceRounding.flashAttention(
            queries: projected(toQ).expandedDimensions(axis: 1),
            keys: projected(toK).expandedDimensions(axis: 1),
            values: projected(toV).expandedDimensions(axis: 1),
            scale: 1 / sqrt(Float(shape[3])), mask: nil, queryBlock: 1024)
        return x + toOut(out.squeezed(axis: 1)).reshaped(shape)
    }
}

final class NFKSDVAEDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKSDVAEResnet]
    @ModuleInfo(key: "downsamplers") var downsamplers: [NFKSDDownsample]

    init(inputChannels: Int, outputChannels: Int, layers: Int, groups: Int, downsamples: Bool) {
        self._resnets.wrappedValue = (0 ..< layers).map { index in
            NFKSDVAEResnet(inputChannels: index == 0 ? inputChannels : outputChannels,
                           outputChannels: outputChannels, groups: groups)
        }
        self._downsamplers.wrappedValue = downsamples
            ? [NFKSDDownsample(channels: outputChannels, padsAsymmetrically: true)]
            : []
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        for downsampler in downsamplers { h = downsampler(h) }
        return h
    }
}

final class NFKSDVAEUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKSDVAEResnet]
    @ModuleInfo(key: "upsamplers") var upsamplers: [NFKSDUpsample]

    init(inputChannels: Int, outputChannels: Int, layers: Int, groups: Int, upsamples: Bool) {
        self._resnets.wrappedValue = (0 ..< layers).map { index in
            NFKSDVAEResnet(inputChannels: index == 0 ? inputChannels : outputChannels,
                           outputChannels: outputChannels, groups: groups)
        }
        self._upsamplers.wrappedValue = upsamples ? [NFKSDUpsample(channels: outputChannels)] : []
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        for upsampler in upsamplers { h = upsampler(h) }
        return h
    }
}

final class NFKSDVAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKSDVAEResnet]
    @ModuleInfo(key: "attentions") var attentions: [NFKSDVAEAttention]

    init(channels: Int, groups: Int) {
        self._resnets.wrappedValue = (0 ..< 2).map { _ in
            NFKSDVAEResnet(inputChannels: channels, outputChannels: channels, groups: groups)
        }
        self._attentions.wrappedValue = [NFKSDVAEAttention(channels: channels, groups: groups)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

final class NFKSDVAEEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [NFKSDVAEDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: NFKSDVAEMidBlock
    @ModuleInfo(key: "conv_norm_out") var normOut: NFKSDGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(_ c: NFKMLXSDVAEConfiguration) {
        let levels = c.blockChannels.count
        self._convIn.wrappedValue = NFKConv2d(inputChannels: 3, outputChannels: c.blockChannels[0],
                                              kernelSize: 3, padding: 1)
        self._downBlocks.wrappedValue = (0 ..< levels).map { level in
            NFKSDVAEDownBlock(inputChannels: level == 0 ? c.blockChannels[0] : c.blockChannels[level - 1],
                              outputChannels: c.blockChannels[level], layers: c.layersPerBlock,
                              groups: c.normalizationGroups, downsamples: level < levels - 1)
        }
        self._midBlock.wrappedValue = NFKSDVAEMidBlock(channels: c.blockChannels[levels - 1],
                                                       groups: c.normalizationGroups)
        self._normOut.wrappedValue = NFKSDGroupNorm(groups: c.normalizationGroups,
                                                    channels: c.blockChannels[levels - 1], eps: 1e-6)
        // The encoder emits a mean and a log-variance, so twice the latent width.
        self._convOut.wrappedValue = NFKConv2d(inputChannels: c.blockChannels[levels - 1],
                                               outputChannels: 2 * c.latentChannels,
                                               kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in downBlocks { h = block(h) }
        h = midBlock(h)
        return convOut(NFKReferenceRounding.silu(normOut(h)))
    }
}

final class NFKSDVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: NFKSDVAEMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [NFKSDVAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var normOut: NFKSDGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    /// The output's size over the latent's.
    let outputScale: Int
    /// The widest map the decode holds at the output size, in channels.
    let widestOutputChannels: Int

    init(_ c: NFKMLXSDVAEConfiguration) {
        let widths = c.decoderBlockChannels ?? c.blockChannels
        let levels = widths.count
        let reversed = widths.reversed().map { $0 }
        self.outputScale = 1 << (levels - 1)
        self.widestOutputChannels = reversed.suffix(2).max() ?? reversed[0]
        self._convIn.wrappedValue = NFKConv2d(inputChannels: c.latentChannels,
                                              outputChannels: reversed[0], kernelSize: 3, padding: 1)
        self._midBlock.wrappedValue = NFKSDVAEMidBlock(channels: reversed[0],
                                                       groups: c.normalizationGroups)
        self._upBlocks.wrappedValue = (0 ..< levels).map { level in
            NFKSDVAEUpBlock(inputChannels: level == 0 ? reversed[0] : reversed[level - 1],
                            outputChannels: reversed[level], layers: c.layersPerBlock + 1,
                            groups: c.normalizationGroups, upsamples: level < levels - 1)
        }
        self._normOut.wrappedValue = NFKSDGroupNorm(groups: c.normalizationGroups,
                                                    channels: reversed[levels - 1], eps: 1e-6)
        self._convOut.wrappedValue = NFKConv2d(inputChannels: reversed[levels - 1], outputChannels: 3,
                                               kernelSize: 3, padding: 1)
    }

    /// The decode of `x`, staged when its estimated one-graph peak exceeds ``NFKSDDecodeStaging/budget``.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let peak = NFKSDDecodeStaging.estimatedWholeGraphPeak(
            images: x.shape[0], outputPixels: x.shape[1] * x.shape[2] * outputScale * outputScale,
            widestOutputChannels: widestOutputChannels)
        return decoded(x, evaluationBytes: NFKSDDecodeStaging.evaluationBytes(estimatedPeak: peak,
                                                                               budget: NFKSDDecodeStaging.budget))
    }

    /// The decode of `x`, evaluating each intermediate of at least `evaluationBytes` before the next step reads
    /// it, or as one graph when `evaluationBytes` is nil.
    func decoded(_ x: MLXArray, evaluationBytes: Int?) -> MLXArray {
        NFKSDDecodeStaging.$evaluationBytes.withValue(evaluationBytes) {
            var h = midBlock(convIn(x))
            for block in upBlocks { h = NFKSDDecodeStaging.evaluated(block(h)) }
            return convOut(NFKSDDecodeStaging.evaluated(NFKReferenceRounding.silu(normOut(h))))
        }
    }
}

/// `AutoencoderKL` — the image-to-latent transform every latent-diffusion model here shares.
public final class NFKMLXSDAutoencoder: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKSDVAEEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKSDVAEDecoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d?
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d?

    public let configuration: NFKMLXSDVAEConfiguration

    public init(configuration: NFKMLXSDVAEConfiguration = .stableDiffusion) {
        self.configuration = configuration
        self._encoder.wrappedValue = NFKSDVAEEncoder(configuration)
        self._decoder.wrappedValue = NFKSDVAEDecoder(configuration)
        if configuration.useQuantConv {
            self._quantConv.wrappedValue = NFKConv2d(inputChannels: 2 * configuration.latentChannels,
                                                     outputChannels: 2 * configuration.latentChannels,
                                                     kernelSize: 1)
            self._postQuantConv.wrappedValue = NFKConv2d(inputChannels: configuration.latentChannels,
                                                         outputChannels: configuration.latentChannels,
                                                         kernelSize: 1)
        } else {
            self._quantConv.wrappedValue = nil
            self._postQuantConv.wrappedValue = nil
        }
    }

    /// The latent distribution's mean and log-variance, each `[batch, height/8, width/8, latent]`.
    /// A caller that wants the deterministic latent takes the mean.
    public func encode(_ image: MLXArray) -> (mean: MLXArray, logVariance: MLXArray) {
        var moments = encoder(image)
        if let quantConv {
            moments = quantConv(moments)
        }
        let latent = configuration.latentChannels
        return (moments[.ellipsis, 0 ..< latent], moments[.ellipsis, latent ..< (2 * latent)])
    }

    /// A draw from the latent distribution ``encode(_:)`` describes, as diffusers'
    /// `DiagonalGaussianDistribution.sample` takes it: `mean + exp(0.5 · logVariance) · noise`, the log-variance
    /// clamped to [−30, 20] and each operation rounded to the mean's type. `noise` is standard normal noise of the
    /// mean's shape.
    static func latentSample(mean: MLXArray, logVariance: MLXArray, noise: MLXArray) -> MLXArray {
        let halved = NFKReferenceRounding.scaled(clip(logVariance, min: -30, max: 20), by: 0.5)
        return mean + NFKReferenceRounding.exponential(halved) * noise.asType(mean.dtype)
    }

    public func decode(_ latent: MLXArray) -> MLXArray {
        decoder(postQuantConv?(latent) ?? latent)
    }
}

// MARK: Weights

/// Registration and weight loading for the shared Stable Diffusion networks.
@objc(NFKMLXStableDiffusionModels)
public final class NFKMLXStableDiffusionModels: NSObject {

    /// Translates a diffusers UNet checkpoint key to this module's.
    public static func remapUNetKey(_ key: String) -> String {
        var name = key
        // The reference stores the transformer's ends under one name whose type depends on
        // `use_linear_projection`; MLX needs the two shapes in separate properties.
        name = name.replacingOccurrences(of: ".proj_in.", with: ".proj_in_conv.")
        name = name.replacingOccurrences(of: ".proj_out.", with: ".proj_out_conv.")
        name = name.replacingOccurrences(of: ".attn1.to_out.0.", with: ".attn1.to_out.")
        name = name.replacingOccurrences(of: ".attn2.to_out.0.", with: ".attn2.to_out.")
        name = name.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj.")
        name = name.replacingOccurrences(of: ".ff.net.2.", with: ".ff.out.")
        return name
    }

    /// Translates a diffusers autoencoder checkpoint key to this module's.
    public static func remapVAEKey(_ key: String) -> String {
        var name = key
        name = name.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
        // A release from before diffusers renamed its attention projections spells them `query`,
        // `key`, `value`, and `proj_attn`; the SD 1.5 autoencoders are all of that vintage.
        name = name.replacingOccurrences(of: ".attentions.0.query.", with: ".attentions.0.to_q.")
        name = name.replacingOccurrences(of: ".attentions.0.key.", with: ".attentions.0.to_k.")
        name = name.replacingOccurrences(of: ".attentions.0.value.", with: ".attentions.0.to_v.")
        name = name.replacingOccurrences(of: ".attentions.0.proj_attn.", with: ".attentions.0.to_out.")
        // Every autoencoder resnet is wrapped so it can share the UNet's block.
        for part in ["norm1", "conv1", "norm2", "conv2", "conv_shortcut"] {
            name = name.replacingOccurrences(of: ".resnets.0.\(part).", with: ".resnets.0.block.\(part).")
            name = name.replacingOccurrences(of: ".resnets.1.\(part).", with: ".resnets.1.block.\(part).")
            name = name.replacingOccurrences(of: ".resnets.2.\(part).", with: ".resnets.2.block.\(part).")
        }
        return name
    }

    /// Loads a diffusers UNet checkpoint into `net`.
    public static func loadUNetWeights(into net: NFKMLXSDUNet, from url: URL,
                                       precision: NFKMLXWeightPrecision = .float32) throws {
        try loadUNetWeights(into: net, from: url, precision: precision, castingTo: nil)
    }

    /// Loads a diffusers UNet checkpoint into `net`, its floating tensors converted to `dtype` as they are
    /// read where `dtype` is set, else held at `precision`.
    static func loadUNetWeights(into net: NFKMLXSDUNet, from url: URL, precision: NFKMLXWeightPrecision,
                                castingTo dtype: DType?) throws {
        let checkpoint = try checkpointArrays(url: url, precision: precision, castingTo: dtype)
        let linear = net.configuration.usesLinearProjection
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            var name = remapUNetKey(key)
            if linear {
                name = name.replacingOccurrences(of: ".proj_in_conv.", with: ".proj_in_linear.")
                name = name.replacingOccurrences(of: ".proj_out_conv.", with: ".proj_out_linear.")
            }
            return (name, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
        net.upcastsAttention = upcastsAttention(unetWeightsURL: url)
    }

    /// A diffusers checkpoint's arrays at the precision a load asks for, and whether its 4-D weights need
    /// the transpose. A `dtype` converts every floating tensor to it as the file is read, in groups, so the
    /// stored copy never sits whole beside the converted one; without one the arrays are held at `precision`.
    static func checkpointArrays(url: URL, precision: NFKMLXWeightPrecision, castingTo dtype: DType?) throws
        -> (arrays: [(String, MLXArray)], needsConvTranspose: Bool) {
        if let dtype {
            return try NFKMLXReleaseWeights.convertedCheckpoint(url: url, to: dtype)
        }
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        return (NFKMLXWeights.converted(checkpoint.arrays.map { ($0.key, $0.value) }, to: precision),
                checkpoint.needsConvTranspose)
    }

    /// Whether the release's `config.json` beside a UNet checkpoint sets `upcast_attention` (Stable
    /// Diffusion 2.1), under which diffusers' classic attention forms its scores in float32. A half-precision
    /// run that rounds the raw scores overflows on this release.
    static func upcastsAttention(unetWeightsURL url: URL) -> Bool {
        let configURL = url.deletingLastPathComponent().appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return json["upcast_attention"] as? Bool ?? false
    }

    /// The autoencoder geometry a diffusers `vae/config.json` describes: its widths, depth, groups,
    /// latent width, latent scale and shift, and whether the `1×1` quantization convolutions are present.
    ///
    /// @discussion A config keeping one quantization convolution and dropping the other is refused; the
    /// module holds both or neither. Introduced in InferKit 0.4.0.
    public static func vaeConfiguration(fromHuggingFace url: URL) throws -> NFKMLXSDVAEConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        let quant = json["use_quant_conv"] as? Bool ?? true
        guard quant == (json["use_post_quant_conv"] as? Bool ?? true) else {
            throw NFKMLXError.unsupportedConfiguration("the autoencoder keeps one quantization convolution of two")
        }
        var configuration = NFKMLXSDVAEConfiguration()
        configuration.latentChannels = json["latent_channels"] as? Int ?? configuration.latentChannels
        configuration.blockChannels = json["block_out_channels"] as? [Int] ?? configuration.blockChannels
        configuration.layersPerBlock = json["layers_per_block"] as? Int ?? configuration.layersPerBlock
        configuration.normalizationGroups = json["norm_num_groups"] as? Int ?? configuration.normalizationGroups
        configuration.scaleFactor = (json["scaling_factor"] as? NSNumber)?.floatValue ?? configuration.scaleFactor
        configuration.shiftFactor = (json["shift_factor"] as? NSNumber)?.floatValue ?? 0
        configuration.useQuantConv = quant
        return configuration
    }

    /// Whether the release's `config.json` beside an autoencoder checkpoint sets `force_upcast`, which a
    /// diffusers SDXL pipeline reads to decode a float16 autoencoder in float32. The reference's default
    /// is true, so a release without the file or the key upcasts.
    static func forcesUpcast(vaeWeightsURL url: URL) -> Bool {
        let configURL = url.deletingLastPathComponent().appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return true }
        return json["force_upcast"] as? Bool ?? true
    }

    /// Loads a diffusers autoencoder checkpoint into `net`.
    public static func loadVAEWeights(into net: NFKMLXSDAutoencoder, from url: URL,
                                      precision: NFKMLXWeightPrecision = .float32) throws {
        try loadVAEWeights(into: net, from: url, precision: precision, castingTo: nil)
    }

    /// Loads a diffusers autoencoder checkpoint into `net`, its floating tensors converted to `dtype` as they
    /// are read where `dtype` is set, else held at `precision`.
    static func loadVAEWeights(into net: NFKMLXSDAutoencoder, from url: URL, precision: NFKMLXWeightPrecision,
                               castingTo dtype: DType?) throws {
        let checkpoint = try checkpointArrays(url: url, precision: precision, castingTo: dtype)
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            (remapVAEKey(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

// MARK: The pipeline the three shipped models share

/// A UNet and an autoencoder together, with the text conditioning the released checkpoints require.
///
/// The three latent-diffusion models here differ in how they build the UNet's input and what they do
/// with the decoded result; they share this. Text conditioning arrives as a tensor rather than from an
/// encoder — the caller brings the tower, as with every other conditioned model in this package.
public final class NFKMLXSDPipeline: Module {
    @ModuleInfo(key: "unet") var unet: NFKMLXSDUNet
    @ModuleInfo(key: "vae") var vae: NFKMLXSDAutoencoder

    /// `[tokens, crossAttentionDimensions]`. Every released checkpoint here cross-attends, so a run
    /// without this is not a run of the trained model; the models that take no prompt still expect the
    /// embedding of an empty one.
    public var textContext: MLXArray?

    public init(unet: NFKMLXSDUNetConfiguration, vae: NFKMLXSDVAEConfiguration) {
        self._unet.wrappedValue = NFKMLXSDUNet(configuration: unet)
        self._vae.wrappedValue = NFKMLXSDAutoencoder(configuration: vae)
    }

    /// Loads the released two-file layout: one diffusers checkpoint per network.
    public func loadWeights(unetURL: URL, vaeURL: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        try loadWeights(unetURL: unetURL, vaeURL: vaeURL, precision: precision, castingTo: nil)
    }

    /// Loads the released two-file layout with both networks' floating tensors converted to `dtype` as they
    /// are read where `dtype` is set, else held at `precision`.
    func loadWeights(unetURL: URL, vaeURL: URL, precision: NFKMLXWeightPrecision, castingTo dtype: DType?) throws {
        try NFKMLXStableDiffusionModels.loadUNetWeights(into: unet, from: unetURL, precision: precision,
                                                        castingTo: dtype)
        try NFKMLXStableDiffusionModels.loadVAEWeights(into: vae, from: vaeURL, precision: precision,
                                                       castingTo: dtype)
    }

    /// Loads a single checkpoint holding both networks under `unet.` and `vae.` — the layout
    /// `NFKMLXWeights.save` writes, so a fine-tuned pipeline reloads through one file.
    public func loadWeights(from url: URL) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            let name = key.hasPrefix("vae.")
                ? "vae." + NFKMLXStableDiffusionModels.remapVAEKey(String(key.dropFirst(4)))
                : (key.hasPrefix("unet.")
                    ? "unet." + NFKMLXStableDiffusionModels.remapUNetKey(String(key.dropFirst(5)))
                    : key)
            guard checkpoint.needsConvTranspose, value.ndim == 4 else { return (name, value) }
            return (name, value.transposed(0, 2, 3, 1))
        }
        try NFKMLXWeights.apply(mapped, to: self)
    }

    /// Reads a `[tokens, dimensions]` embedding stored under `context` and installs it.
    public func loadTextContext(from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let arrays = checkpoint.arrays
        guard let context = arrays["context"] ?? arrays.values.first else {
            throw NFKMLXError.weightsMismatch("the text-context file carries no `context` tensor")
        }
        textContext = context.ndim == 3 ? context[0] : context
    }

    /// The conditioning the UNet takes, batched. A model built without weights runs with zeros so that
    /// a shape-only test needs no embedding file.
    func batchedContext(dimensions: Int) -> MLXArray {
        guard let textContext else { return MLXArray.zeros([1, 77, dimensions]) }
        return textContext.reshaped([1] + textContext.shape)
    }

    /// The deterministic latent of `image` (`[1, h, w, 3]` in `-1...1`), already scaled.
    func latent(of image: MLXArray, scale: Float) -> MLXArray {
        vae.encode(image).mean * scale
    }

    /// A latent drawn from the autoencoder's distribution over `image` (`[1, h, w, 3]` in `-1...1`) and scaled, as
    /// diffusers' image-to-image and inpainting take it (`retrieve_latents`), its noise from the run's seed
    /// (``NFKMLXDiffusionBackend/encodingNoise(height:width:channels:stream:)``).
    func sampledLatent(of image: MLXArray, scale: Float, stream: UInt64) -> MLXArray {
        let (mean, logVariance) = vae.encode(image)
        let noise = NFKMLXDiffusionBackend.encodingNoise(height: mean.dim(1), width: mean.dim(2), channels: mean.dim(3),
                                                         stream: stream).reshaped(mean.shape)
        let drawn = NFKMLXSDAutoencoder.latentSample(mean: mean, logVariance: logVariance, noise: noise)
        return NFKReferenceRounding.scaled(drawn, by: scale)
    }

    /// The image a latent decodes to. The latent takes the autoencoder's type before it is unscaled, as a
    /// diffusers pipeline casts it, so an autoencoder upcast to float32 decodes a float32 latent.
    func image(from latent: MLXArray, scale: Float) -> MLXArray {
        let typed = latent.asType(NFKReferenceRounding.parameterType(of: vae))
        return vae.decode(NFKReferenceRounding.divided(typed, by: scale))
    }
}
