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
    public var layersPerBlock: Int = 2
    public var normalizationGroups: Int = 32
    /// Multiplies an encoded latent (and divides before decoding) so its variance suits the UNet —
    /// the diffusers `scaling_factor`, which each release trains with.
    public var scaleFactor: Float = 0.18215

    public init() {}

    public static let stableDiffusion = NFKMLXSDVAEConfiguration()

    /// The upscaler's autoencoder is one level shallower and scales its latents differently.
    public static let upscaler: NFKMLXSDVAEConfiguration = {
        var c = NFKMLXSDVAEConfiguration()
        c.blockChannels = [128, 256, 512]
        c.scaleFactor = 0.08333
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape                                     // [batch, height, width, channel]
        let channels = shape[3]
        let perGroup = channels / groups
        // The reference groups consecutive channels and normalizes each group over its channels and
        // the whole spatial extent.
        let grouped = x.reshaped([shape[0], shape[1] * shape[2], groups, perGroup])
            .transposed(0, 2, 1, 3)
            .reshaped([shape[0], groups, shape[1] * shape[2] * perGroup])
        let mean = grouped.mean(axis: 2, keepDims: true)
        let centered = grouped - mean
        let variance = (centered * centered).mean(axis: 2, keepDims: true)
        let normalized = (centered * rsqrt(variance + eps))
            .reshaped([shape[0], groups, shape[1] * shape[2], perGroup])
            .transposed(0, 2, 1, 3)
            .reshaped(shape)
        return normalized * weight + bias
    }
}

/// `get_timestep_embedding`: half the channels are cosines and half sines, cosines first, over
/// frequencies spaced logarithmically to `maxPeriod`.
enum NFKSDTimesteps {
    static func embedding(_ timesteps: MLXArray, channels: Int, maxPeriod: Float = 10_000) -> MLXArray {
        let half = channels / 2
        var frequencies = [Float](repeating: 0, count: half)
        for i in 0 ..< half {
            frequencies[i] = expf(-logf(maxPeriod) * Float(i) / Float(half))
        }
        let scale = frequencies.withUnsafeBufferPointer { MLXArray($0, [1, half]) }
        let angles = timesteps.reshaped([-1, 1]).asType(.float32) * scale
        // `flip_sin_to_cos` is true for every released Stable Diffusion UNet.
        return concatenated([cos(angles), sin(angles)], axis: 1)
    }
}

/// `TimestepEmbedding`: two linear layers with a SiLU between them.
final class NFKSDTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputChannels: Int, timeChannels: Int) {
        self._linear1.wrappedValue = Linear(inputChannels, timeChannels)
        self._linear2.wrappedValue = Linear(timeChannels, timeChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
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

    init(inputChannels: Int, outputChannels: Int, timeChannels: Int?, groups: Int) {
        self._norm1.wrappedValue = NFKSDGroupNorm(groups: groups, channels: inputChannels, eps: 1e-5)
        self._conv1.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels,
                                          kernelSize: 3, padding: 1)
        self._timeProjection.wrappedValue = timeChannels.map { Linear($0, outputChannels) }
        self._norm2.wrappedValue = NFKSDGroupNorm(groups: groups, channels: outputChannels, eps: 1e-5)
        self._conv2.wrappedValue = Conv2d(inputChannels: outputChannels, outputChannels: outputChannels,
                                          kernelSize: 3, padding: 1)
        self._shortcut.wrappedValue = inputChannels == outputChannels
            ? nil
            : Conv2d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray, time: MLXArray?) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        if let timeProjection, let time {
            h = h + timeProjection(silu(time)).reshaped([time.shape[0], 1, 1, -1])
        }
        h = conv2(silu(norm2(h)))
        return (shortcut.map { $0(x) } ?? x) + h
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
        return out(value * gelu(gate))
    }
}

/// The transformer's attention. Queries, keys, and values carry no bias; only the output projection
/// does. Passing `context` makes it cross-attention.
final class NFKSDAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    let heads: Int

    init(dimensions: Int, contextDimensions: Int, heads: Int) {
        self.heads = heads
        self._toQ.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._toK.wrappedValue = Linear(contextDimensions, dimensions, bias: false)
        self._toV.wrappedValue = Linear(contextDimensions, dimensions, bias: false)
        self._toOut.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let source = context ?? x
        let batch = x.shape[0], dimensions = x.shape[2]
        let headDimensions = dimensions / heads
        func split(_ value: MLXArray) -> MLXArray {
            value.reshaped([batch, value.shape[1], heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: split(toQ(x)), keys: split(toK(source)), values: split(toV(source)),
            scale: 1 / sqrt(Float(headDimensions)), mask: nil)
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
        self._norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        self._attn1.wrappedValue = NFKSDAttention(
            dimensions: dimensions,
            contextDimensions: onlyCrossAttention ? contextDimensions : dimensions, heads: heads)
        self._norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        self._attn2.wrappedValue = NFKSDAttention(dimensions: dimensions,
                                                  contextDimensions: contextDimensions, heads: heads)
        self._norm3.wrappedValue = LayerNorm(dimensions: dimensions)
        self._ff.wrappedValue = NFKSDFeedForward(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray?) -> MLXArray {
        var out = x + attn1(norm1(x), context: onlyCrossAttention ? context : nil)
        out = out + attn2(norm2(out), context: context)
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
            self._projInConv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels,
                                                   kernelSize: 1)
            self._projOutConv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels,
                                                    kernelSize: 1)
        }
        self._blocks.wrappedValue = (0 ..< depth).map { _ in
            NFKSDTransformerBlock(dimensions: channels, contextDimensions: contextDimensions,
                                  heads: heads, onlyCrossAttention: onlyCrossAttention)
        }
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray?) -> MLXArray {
        let shape = x.shape
        var h = norm(x)
        // A convolution projects before the tokens are flattened; a linear layer projects after.
        if let projInConv { h = projInConv(h) }
        h = h.reshaped([shape[0], shape[1] * shape[2], shape[3]])
        if let projInLinear { h = projInLinear(h) }
        for block in blocks { h = block(h, context: context) }
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
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels,
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
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels,
                                         kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(NFKMLXResample.upsampleNearest(x, scale: 2))
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
    func callAsFunction(_ x: MLXArray, time: MLXArray, context: MLXArray?) -> (MLXArray, [MLXArray]) {
        var h = x
        var skips = [MLXArray]()
        for (index, resnet) in resnets.enumerated() {
            h = resnet(h, time: time)
            if index < attentions.count { h = attentions[index](h, context: context) }
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
                        context: MLXArray?) -> MLXArray {
        var h = x
        for (index, resnet) in resnets.enumerated() {
            h = resnet(concatenated([h, skips.removeLast()], axis: 3), time: time)
            if index < attentions.count { h = attentions[index](h, context: context) }
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

    func callAsFunction(_ x: MLXArray, time: MLXArray, context: MLXArray?) -> MLXArray {
        var h = resnets[0](x, time: time)
        h = attentions[0](h, context: context)
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

        self._timeEmbedding.wrappedValue = NFKSDTimestepEmbedding(inputChannels: c.blockChannels[0],
                                                                  timeChannels: timeChannels)
        self._classEmbedding.wrappedValue = c.classEmbeddingCount.map {
            Embedding(embeddingCount: $0, dimensions: timeChannels)
        }
        self._addEmbedding.wrappedValue = c.additionEmbedding.map {
            NFKSDTimestepEmbedding(inputChannels: $0.projectionInputDimensions, timeChannels: timeChannels)
        }
        self._convIn.wrappedValue = Conv2d(inputChannels: c.inputChannels,
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
        self._convOut.wrappedValue = Conv2d(inputChannels: c.blockChannels[0],
                                            outputChannels: c.outputChannels,
                                            kernelSize: 3, padding: 1)
    }

    /// - Parameters:
    ///   - x: the latent, `[batch, height, width, inputChannels]`.
    ///   - timestep: the diffusion step.
    ///   - context: the text (or other) conditioning, `[batch, tokens, crossAttentionDimensions]`.
    ///   - classLabel: the noise level, for a model that carries a class embedding.
    ///   - added: the pooled embedding and size descriptor, for a model that carries a `text_time`
    ///     addition embedding.
    public func callAsFunction(_ x: MLXArray, timestep: MLXArray, context: MLXArray?,
                               classLabel: MLXArray? = nil,
                               added: NFKSDAddedConditioning? = nil) -> MLXArray {
        var time = timeEmbedding(NFKSDTimesteps.embedding(timestep,
                                                          channels: configuration.blockChannels[0]))
        if let classEmbedding, let classLabel {
            time = time + classEmbedding(classLabel)
        }
        if let addEmbedding, let added, let addition = configuration.additionEmbedding {
            // Every entry of the descriptor embeds separately and the results run together, so six
            // numbers become one vector the width of six timestep embeddings.
            let batch = added.pooled.shape[0]
            let descriptor = NFKSDTimesteps.embedding(added.timeIds.reshaped([-1]),
                                                      channels: addition.timeEmbeddingDimensions)
                .reshaped([batch, -1])
            time = time + addEmbedding(concatenated([added.pooled, descriptor], axis: -1))
        }

        var h = convIn(x)
        var skips = [h]
        for block in downBlocks {
            let (out, produced) = block(h, time: time, context: context)
            h = out
            skips += produced
        }
        h = midBlock(h, time: time, context: context)
        for block in upBlocks {
            h = block(h, skips: &skips, time: time, context: context)
        }
        return convOut(silu(normOut(h)))
    }
}

// MARK: The autoencoder

/// A resnet block of the autoencoder: the UNet's, without the timestep.
final class NFKSDVAEResnet: Module {
    @ModuleInfo(key: "block") var block: NFKSDResnetBlock

    init(inputChannels: Int, outputChannels: Int, groups: Int) {
        self._block.wrappedValue = NFKSDResnetBlock(inputChannels: inputChannels,
                                                    outputChannels: outputChannels,
                                                    timeChannels: nil, groups: groups)
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
        let out = MLXFast.scaledDotProductAttention(
            queries: toQ(tokens).expandedDimensions(axis: 1),
            keys: toK(tokens).expandedDimensions(axis: 1),
            values: toV(tokens).expandedDimensions(axis: 1),
            scale: 1 / sqrt(Float(shape[3])), mask: nil)
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
        self._convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.blockChannels[0],
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
        self._convOut.wrappedValue = Conv2d(inputChannels: c.blockChannels[levels - 1],
                                            outputChannels: 2 * c.latentChannels,
                                            kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in downBlocks { h = block(h) }
        h = midBlock(h)
        return convOut(silu(normOut(h)))
    }
}

final class NFKSDVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: NFKSDVAEMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [NFKSDVAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var normOut: NFKSDGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(_ c: NFKMLXSDVAEConfiguration) {
        let levels = c.blockChannels.count
        let reversed = c.blockChannels.reversed().map { $0 }
        self._convIn.wrappedValue = Conv2d(inputChannels: c.latentChannels,
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
        self._convOut.wrappedValue = Conv2d(inputChannels: reversed[levels - 1], outputChannels: 3,
                                            kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = midBlock(convIn(x))
        for block in upBlocks { h = block(h) }
        return convOut(silu(normOut(h)))
    }
}

/// `AutoencoderKL` — the image-to-latent transform every latent-diffusion model here shares.
public final class NFKMLXSDAutoencoder: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKSDVAEEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKSDVAEDecoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d

    public let configuration: NFKMLXSDVAEConfiguration

    public init(configuration: NFKMLXSDVAEConfiguration = .stableDiffusion) {
        self.configuration = configuration
        self._encoder.wrappedValue = NFKSDVAEEncoder(configuration)
        self._decoder.wrappedValue = NFKSDVAEDecoder(configuration)
        self._quantConv.wrappedValue = Conv2d(inputChannels: 2 * configuration.latentChannels,
                                              outputChannels: 2 * configuration.latentChannels,
                                              kernelSize: 1)
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: configuration.latentChannels,
                                                  outputChannels: configuration.latentChannels,
                                                  kernelSize: 1)
    }

    /// The latent distribution's mean and log-variance, each `[batch, height/8, width/8, latent]`.
    /// A caller that wants the deterministic latent takes the mean.
    public func encode(_ image: MLXArray) -> (mean: MLXArray, logVariance: MLXArray) {
        let moments = quantConv(encoder(image))
        let latent = configuration.latentChannels
        return (moments[.ellipsis, 0 ..< latent], moments[.ellipsis, latent ..< (2 * latent)])
    }

    public func decode(_ latent: MLXArray) -> MLXArray {
        decoder(postQuantConv(latent))
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
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let linear = net.configuration.usesLinearProjection
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            var name = remapUNetKey(key)
            if linear {
                name = name.replacingOccurrences(of: ".proj_in_conv.", with: ".proj_in_linear.")
                name = name.replacingOccurrences(of: ".proj_out_conv.", with: ".proj_out_linear.")
            }
            guard checkpoint.needsConvTranspose else { return (name, value) }
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(NFKMLXWeights.converted(mapped, to: precision), to: net)
    }

    /// Loads a diffusers autoencoder checkpoint into `net`.
    public static func loadVAEWeights(into net: NFKMLXSDAutoencoder, from url: URL,
                                      precision: NFKMLXWeightPrecision = .float32) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            let name = remapVAEKey(key)
            guard checkpoint.needsConvTranspose else { return (name, value) }
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(NFKMLXWeights.converted(mapped, to: precision), to: net)
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
        try NFKMLXStableDiffusionModels.loadUNetWeights(into: unet, from: unetURL, precision: precision)
        try NFKMLXStableDiffusionModels.loadVAEWeights(into: vae, from: vaeURL, precision: precision)
    }

    /// Loads a single checkpoint holding both networks under `unet.` and `vae.` — the layout
    /// `NFKMLXWeights.save` writes, so a fine-tuned pipeline reloads through one file.
    public func loadWeights(from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
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

    func image(from latent: MLXArray, scale: Float) -> MLXArray {
        vae.decode(latent / scale)
    }
}
