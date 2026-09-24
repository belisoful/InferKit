//
//  NFKMLXQwenImageTransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The Qwen-Image 2.1 denoising transformer (`QwenImage21Transformer2DModel`, Qwen). It is SINGLE-STREAM
// like the Z-Image DiT: the caption tokens and the image latents share one sequence. What is new here is
// how that sequence is read.
//
// Attention is BLOCK-CAUSAL. A token attends to everything before it, and every image block — each
// condition image, and the target image — is additionally bidirectional within itself, so the rule is
// `q >= kv or same image block`. The caption is therefore strictly causal, which is what lets the text
// and condition-image prefix be computed once and cached across denoising steps.
//
// The timestep reaches the two halves of the sequence differently. Under `causal_condition` the text and
// condition-image tokens modulate from `t = 0` while the target image's tokens modulate from the sampled
// timestep, which is what makes the prefix independent of the step and so cacheable. The port carries
// both rows and selects per token.
//
// The condition images do not sit beside the caption; they sit INSIDE it, at the slots the
// vision-language encoder reserved, and each slot stands for a 2x2 group of latent tokens.

/// Qwen-Image 2.1 transformer geometry. Defaults are the released 7B model.
public struct NFKMLXQwenImageConfiguration: Sendable {
    public var patchSize: Int
    public var inChannels: Int
    public var outChannels: Int
    public var layers: Int
    public var headDimensions: Int
    public var heads: Int
    public var contextInDimensions: Int
    public var mlpRatio: Int
    public var axesDimensionsRope: [Int]
    public var eps: Float
    /// Whether the text and condition-image tokens modulate from `t = 0` rather than from the sampled
    /// timestep. The release sets it, and the prefix cache is only valid because of it.
    public var causalCondition: Bool
    public var ropeTheta: Float

    public init(patchSize: Int = 1, inChannels: Int = 64, outChannels: Int = 64, layers: Int = 32,
                headDimensions: Int = 128, heads: Int = 32, contextInDimensions: Int = 4096,
                mlpRatio: Int = 3, axesDimensionsRope: [Int] = [16, 56, 56], eps: Float = 1e-6,
                causalCondition: Bool = true, ropeTheta: Float = 10_000) {
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.layers = layers
        self.headDimensions = headDimensions
        self.heads = heads
        self.contextInDimensions = contextInDimensions
        self.mlpRatio = mlpRatio
        self.axesDimensionsRope = axesDimensionsRope
        self.eps = eps
        self.causalCondition = causalCondition
        self.ropeTheta = ropeTheta
    }

    /// `Qwen/Qwen-Image-2.1`.
    public static let base = NFKMLXQwenImageConfiguration()

    /// A small configuration for tests and examples. The rotary axes keep the released proportions so
    /// the three-axis split is exercised.
    public static let tiny = NFKMLXQwenImageConfiguration(
        inChannels: 8, outChannels: 8, layers: 2, headDimensions: 16, heads: 2,
        contextInDimensions: 12, mlpRatio: 3, axesDimensionsRope: [4, 6, 6])

    /// The model width, `heads · headDimensions`.
    public var dimensions: Int { heads * headDimensions }
    /// The feed-forward width.
    public var feedForwardDimensions: Int { dimensions * mlpRatio }
    /// What one image token carries into `img_in`.
    public var patchInputDimensions: Int { inChannels * patchSize * patchSize }
}

/// RMS normalization whose stored weight is zero-centered: the scale is `weight + 1`.
///
/// @discussion A checkpoint written this way holds `scale - 1`, so loading it into an ordinary
/// `RMSNorm` would apply a scale near zero rather than near one.
public final class NFKQwenImageZeroCenterRMSNorm: Module, UnaryLayer {
    public let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The reference normalizes in float32 and casts back, which is what a bfloat16 release needs to
        // agree with it: the statistics of a 4096-wide row lose several digits in bfloat16.
        let wide = x.asType(.float32)
        let inverse = rsqrt((wide * wide).mean(axis: -1, keepDims: true) + eps)
        return (wide * inverse * (weight.asType(.float32) + 1)).asType(x.dtype)
    }
}

/// RMS normalization that computes in float32, the way the reference's does.
public final class NFKQwenImageRMSNorm: Module, UnaryLayer {
    public let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let wide = x.asType(.float32)
        let inverse = rsqrt((wide * wide).mean(axis: -1, keepDims: true) + eps)
        return (wide * inverse * weight.asType(.float32)).asType(x.dtype)
    }
}

/// The caption projection: a zero-centered RMS norm, then a two-layer tanh-GELU projection to the model
/// width.
final class NFKQwenImageTextProjection: Module {
    @ModuleInfo(key: "text_norm") var textNorm: NFKQwenImageZeroCenterRMSNorm
    @ModuleInfo(key: "in_layer") var inLayer: Linear
    @ModuleInfo(key: "out_layer") var outLayer: Linear

    init(_ configuration: NFKMLXQwenImageConfiguration) {
        _textNorm.wrappedValue = NFKQwenImageZeroCenterRMSNorm(
            dimensions: configuration.contextInDimensions, eps: configuration.eps)
        _inLayer.wrappedValue = Linear(configuration.contextInDimensions, configuration.dimensions,
                                       bias: false)
        _outLayer.wrappedValue = Linear(configuration.dimensions, configuration.dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outLayer(NFKReferenceRounding.geluTanh(inLayer(textNorm(x))))
    }
}

/// The SwiGLU feed-forward, `out(silu(gate_layer(x)) · proj(x))`.
final class NFKQwenImageFeedForward: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "gate_layer") var gateLayer: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(dimensions: Int, hidden: Int) {
        _proj.wrappedValue = Linear(dimensions, hidden, bias: false)
        _gateLayer.wrappedValue = Linear(dimensions, hidden, bias: false)
        _out.wrappedValue = Linear(hidden, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        out(NFKReferenceRounding.silu(gateLayer(x)) * proj(x))
    }
}

/// The timestep embedder: a sinusoidal projection, then two linear layers around a SiLU.
final class NFKQwenImageTimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(input: Int, dimensions: Int) {
        _linear1.wrappedValue = Linear(input, dimensions, bias: false)
        _linear2.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(NFKReferenceRounding.silu(linear1(x))) }
}

/// Holds the timestep embedder under the release's own nesting.
final class NFKQwenImageTimeTextEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: NFKQwenImageTimestepEmbedder

    init(dimensions: Int) {
        _timestepEmbedder.wrappedValue = NFKQwenImageTimestepEmbedder(input: 256, dimensions: dimensions)
    }

    /// The sinusoidal timestep projection, `[batch]` → `[batch, 256]`.
    ///
    /// @discussion COSINE occupies the first half of the channels and sine the second, which is the
    /// opposite of the order the diffusion models here usually use, and the timestep is scaled by 1000
    /// before the frequencies are applied.
    static func projection(_ timestep: MLXArray, dimensions: Int = 256, maxPeriod: Float = 10_000,
                           timeFactor: Float = 1000) -> MLXArray {
        let half = dimensions / 2
        let exponents = MLXArray((0 ..< half).map { Float($0) / Float(half) })
        let frequencies = exp(-log(maxPeriod) * exponents)
        let arguments = (timeFactor * timestep).reshaped([-1, 1]) * frequencies.reshaped([1, -1])
        return concatenated([cos(arguments), sin(arguments)], axis: -1)
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        timestepEmbedder(Self.projection(timestep))
    }
}

/// The three-axis rotary table over the joint sequence.
///
/// @discussion Each axis carries positions 0…8191 followed by −1024…−1, and an image block's height and
/// width positions are centered on zero, so a block's spatial coordinates do not depend on where the
/// block sits in the sequence. The negative half is why the table is built this way rather than indexed
/// directly: the reference reaches it by Python's negative indexing on the concatenated table.
struct NFKQwenImageRope {
    static let positiveCount = 8192
    static let negativeCount = 1024

    let cosines: [MLXArray]                                   // axis a: [9216, axisDim/2]
    let sines: [MLXArray]

    init(theta: Float, axesDimensions: [Int]) {
        var positions = (0 ..< Self.positiveCount).map { Float($0) }
        positions.append(contentsOf: (0 ..< Self.negativeCount).map { Float($0 - Self.negativeCount) })
        let index = MLXArray(positions).reshaped([-1, 1])
        var cosines = [MLXArray](), sines = [MLXArray]()
        for dimensions in axesDimensions {
            let k = MLXArray(stride(from: 0, to: dimensions, by: 2).map { Float($0) })
            let inverse = pow(MLXArray(theta), -(k / Float(dimensions))).reshaped([1, -1])
            let angles = index * inverse
            cosines.append(cos(angles))
            sines.append(sin(angles))
        }
        self.cosines = cosines
        self.sines = sines
    }

    /// The row a signed position occupies in the table.
    static func row(_ position: Int) -> Int32 {
        position >= 0 ? Int32(position) : Int32(positiveCount + negativeCount + position)
    }

    /// The `(cos, sin)` tables `[sequence, headDimensions/2]` for per-axis position lists.
    func table(_ axes: [[Int]]) -> (cos: MLXArray, sin: MLXArray) {
        var cosParts = [MLXArray](), sinParts = [MLXArray]()
        for (axis, positions) in axes.enumerated() {
            let index = MLXArray(positions.map { Self.row($0) })
            cosParts.append(cosines[axis].take(index, axis: 0))
            sinParts.append(sines[axis].take(index, axis: 0))
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }

    /// The per-axis positions of the joint sequence.
    ///
    /// Text advances one shared position on all three axes. An image block holds the frame axis at the
    /// position the preceding text reached, lays its tokens on a height/width grid centered on zero,
    /// and then advances the shared position by the block's longer side.
    static func positions(imageShapes: [(frame: Int, height: Int, width: Int)],
                          imagePadMask: [Bool]) -> [[Int]] {
        var frame = [Int](), blockHeights = [Int](), blockWidths = [Int]()
        var cursor = 0, position = 0
        for shape in imageShapes {
            guard let blockStart = (cursor ..< imagePadMask.count).first(where: { imagePadMask[$0] }) else {
                break
            }
            for offset in 0 ..< (blockStart - cursor) { frame.append(position + offset) }
            position += blockStart - cursor

            let tokens = shape.height * shape.width
            cursor = blockStart + tokens
            frame.append(contentsOf: [Int](repeating: position, count: tokens))
            position += Swift.max(shape.height, shape.width)

            for row in -(shape.height - shape.height / 2) ..< (shape.height / 2) {
                blockHeights.append(contentsOf: [Int](repeating: row, count: shape.width))
            }
            for _ in 0 ..< shape.height {
                for column in -(shape.width - shape.width / 2) ..< (shape.width / 2) {
                    blockWidths.append(column)
                }
            }
        }
        if cursor < imagePadMask.count {
            for offset in 0 ..< (imagePadMask.count - cursor) { frame.append(position + offset) }
        }

        var height = frame, width = frame
        var taken = 0
        for (index, isImage) in imagePadMask.enumerated() where isImage {
            height[index] = blockHeights[taken]
            width[index] = blockWidths[taken]
            taken += 1
        }
        return [frame, height, width]
    }
}

/// Applies the complex rotary to `x` `[sequence, heads, headDimensions]`, pairing adjacent channels.
func qwenImageApplyRope(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
    let sequence = x.dim(0), heads = x.dim(1), headDimensions = x.dim(2)
    // The reference builds complex tensors, which torch has only at float32, so the rotation is a
    // float32 step in the middle of a bfloat16 forward.
    let wide = x.asType(.float32)
    let pairs = wide.reshaped([sequence, heads, headDimensions / 2, 2])
    let real = pairs[0..., 0..., 0..., 0]
    let imaginary = pairs[0..., 0..., 0..., 1]
    let cc = c.reshaped([sequence, 1, headDimensions / 2])
    let ss = s.reshaped([sequence, 1, headDimensions / 2])
    return stacked([real * cc - imaginary * ss, real * ss + imaginary * cc], axis: -1)
        .reshaped([sequence, heads, headDimensions]).asType(x.dtype)
}

/// The block's attention: per-head RMS query and key norms, the complex rotary, and one masked
/// softmax over the joint sequence.
final class NFKQwenImageAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]
    @ModuleInfo(key: "norm_q") var normQ: NFKQwenImageRMSNorm
    @ModuleInfo(key: "norm_k") var normK: NFKQwenImageRMSNorm

    let heads: Int
    let headDimensions: Int

    init(_ configuration: NFKMLXQwenImageConfiguration) {
        self.heads = configuration.heads
        self.headDimensions = configuration.headDimensions
        let dimensions = configuration.dimensions
        _toQ.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toK.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toV.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toOut.wrappedValue = [Linear(dimensions, dimensions, bias: false)]
        _normQ.wrappedValue = NFKQwenImageRMSNorm(dimensions: configuration.headDimensions,
                                                  eps: configuration.eps)
        _normK.wrappedValue = NFKQwenImageRMSNorm(dimensions: configuration.headDimensions,
                                                  eps: configuration.eps)
    }

    func callAsFunction(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray, mask: MLXArray?) -> MLXArray {
        let sequence = x.dim(0)
        var q = normQ(toQ(x).reshaped([sequence, heads, headDimensions]))
        var k = normK(toK(x).reshaped([sequence, heads, headDimensions]))
        let v = toV(x).reshaped([sequence, heads, headDimensions])
        q = qwenImageApplyRope(q, cos: c, sin: s)
        k = qwenImageApplyRope(k, cos: c, sin: s)

        let queries = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let keys = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let values = v.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let attended = NFKReferenceRounding.flashAttention(
            queries: queries, keys: keys, values: values, scale: 1 / sqrt(Float(headDimensions)),
            mask: mask.map { $0.asType(queries.dtype) })
        let merged = attended[0].transposed(1, 0, 2).reshaped([sequence, heads * headDimensions])
        return (toOut[0] as! Linear)(merged)
    }
}

/// A single-stream block. The modulation is not learned per block: the model computes one shared
/// tensor and every block slices its own scales and gates out of it.
final class NFKQwenImageBlock: Module {
    @ModuleInfo(key: "attn") var attention: NFKQwenImageAttention
    @ModuleInfo(key: "img_mlp") var feedForward: NFKQwenImageFeedForward

    let eps: Float
    let dimensions: Int

    init(_ configuration: NFKMLXQwenImageConfiguration) {
        self.eps = configuration.eps
        self.dimensions = configuration.dimensions
        _attention.wrappedValue = NFKQwenImageAttention(configuration)
        _feedForward.wrappedValue = NFKQwenImageFeedForward(
            dimensions: configuration.dimensions, hidden: configuration.feedForwardDimensions)
        super.init()
    }

    /// The affine-free layer norm both halves of the block apply, computed in float32 the way
    /// `torch.nn.LayerNorm` does for a half-precision input.
    static func normalized(_ x: MLXArray, eps: Float) -> MLXArray {
        let wide = x.asType(.float32)
        let centered = wide - wide.mean(axis: -1, keepDims: true)
        let scaled = centered * rsqrt(centered.square().mean(axis: -1, keepDims: true) + eps)
        return scaled.asType(x.dtype)
    }

    /// `x` `[sequence, dimensions]`, `modulation` `[sequence, 4 · dimensions]` → `[sequence, dimensions]`.
    func callAsFunction(_ x: MLXArray, modulation: MLXArray, cos c: MLXArray, sin s: MLXArray,
                        mask: MLXArray?) -> MLXArray {
        let scaleAttention = modulation[0..., 0 ..< dimensions]
        let gateAttention = modulation[0..., dimensions ..< (2 * dimensions)]
        let scaleFeedForward = modulation[0..., (2 * dimensions) ..< (3 * dimensions)]
        let gateFeedForward = modulation[0..., (3 * dimensions)...]

        var hidden = x
        let attended = attention(Self.normalized(hidden, eps: eps) * (1 + scaleAttention),
                                 cos: c, sin: s, mask: mask)
        hidden = hidden + tanh(gateAttention) * attended
        let projected = feedForward(Self.normalized(hidden, eps: eps) * (1 + scaleFeedForward))
        return hidden + tanh(gateFeedForward) * projected
    }
}

/// The Qwen-Image 2.1 transformer.
public final class NFKMLXQwenImageNet: Module {
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: NFKQwenImageTimeTextEmbed
    @ModuleInfo(key: "txt_in") var textIn: NFKQwenImageTextProjection
    @ModuleInfo(key: "img_in") var imageIn: Linear
    @ModuleInfo(key: "modulation") var modulation: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [NFKQwenImageBlock]
    @ModuleInfo(key: "norm_out") var normOut: Linear
    @ModuleInfo(key: "proj_out") var projectionOut: Linear

    public let configuration: NFKMLXQwenImageConfiguration
    let rope: NFKQwenImageRope

    public init(_ configuration: NFKMLXQwenImageConfiguration) {
        self.configuration = configuration
        self.rope = NFKQwenImageRope(theta: configuration.ropeTheta,
                                     axesDimensions: configuration.axesDimensionsRope)
        let dimensions = configuration.dimensions
        _timeTextEmbed.wrappedValue = NFKQwenImageTimeTextEmbed(dimensions: dimensions)
        _textIn.wrappedValue = NFKQwenImageTextProjection(configuration)
        _imageIn.wrappedValue = Linear(configuration.patchInputDimensions, dimensions, bias: false)
        _modulation.wrappedValue = Linear(dimensions, 4 * dimensions, bias: false)
        _blocks.wrappedValue = (0 ..< configuration.layers).map { _ in NFKQwenImageBlock(configuration) }
        _normOut.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _projectionOut.wrappedValue = Linear(
            dimensions, configuration.patchSize * configuration.patchSize * configuration.outChannels,
            bias: false)
        super.init()
    }

    /// Where each token's latents and each image block sit in the joint sequence.
    ///
    /// - Parameters:
    ///   - imageMask: `true` at the vision-language encoder's image slots, over the caption followed by
    ///     the target image's slots. Each slot stands for a 2x2 group of latent tokens.
    ///   - imageShapes: per image `(frame, height, width)` in latent tokens, the condition images first
    ///     and the target image last.
    /// - Returns: the expanded per-token image mask, the block id of every token (`-1` for text), and
    ///   the mask marking the target image's tokens.
    public static func tokenMetadata(imageMask: [Bool],
                                     imageShapes: [(frame: Int, height: Int, width: Int)])
        -> (imagePadMask: [Bool], imageIds: [Int], targetMask: [Bool]) {
        var padMask = [Bool]()
        for slot in imageMask {
            padMask.append(contentsOf: [Bool](repeating: slot, count: slot ? 4 : 1))
        }
        var ids = [Int](repeating: -1, count: padMask.count)
        let lengths = imageShapes.map { $0.frame * $0.height * $0.width }
        var block = 0, remaining = lengths.first ?? 0
        for (index, isImage) in padMask.enumerated() where isImage {
            while remaining == 0 && block + 1 < lengths.count {
                block += 1
                remaining = lengths[block]
            }
            ids[index] = block
            remaining -= 1
        }
        var targetMask = [Bool](repeating: false, count: padMask.count)
        let lastBlock = lengths.count - 1
        for (index, id) in ids.enumerated() where id == lastBlock { targetMask[index] = true }
        return (padMask, ids, targetMask)
    }

    /// The block-causal mask: a token attends to everything before it, and to every token of its own
    /// image block. A caption position the prompt does not fill is excluded as a key.
    static func attentionMask(imageIds: [Int], keyValid: [Bool]?) -> MLXArray {
        let sequence = imageIds.count
        let ids = MLXArray(imageIds.map { Int32($0) })
        let index = MLXArray((0 ..< sequence).map { Int32($0) })
        let causal = (index.reshaped([sequence, 1]) .>= index.reshaped([1, sequence]))
        let sameBlock = (ids.reshaped([sequence, 1]) .== ids.reshaped([1, sequence]))
            .asType(.float32) * (ids .>= 0).asType(.float32).reshaped([sequence, 1])
        var allowed = maximum(causal.asType(.float32), sameBlock)
        if let keyValid {
            allowed = allowed * MLXArray(keyValid.map { $0 ? Float(1) : 0 }).reshaped([1, sequence])
        }
        return MLX.where(allowed .> 0, MLXArray(Float(0)), MLXArray(Float(-1e9)))
    }

    /// One denoising step over the joint sequence.
    ///
    /// - Parameters:
    ///   - latents: `[imageTokens, inChannels]`, the condition images' tokens first and the target
    ///     image's last, in the order `imageShapes` names them.
    ///   - encoderHidden: `[captionTokens, contextInDimensions]` from the vision-language encoder,
    ///     including the slots it reserved for the condition images.
    ///   - timestep: the denoising step, scaled to `0…1`.
    ///   - imageShapes: per image `(frame, height, width)` in latent tokens.
    ///   - imageMask: `true` at the encoder's image slots, over the caption followed by the target
    ///     image's own slots (one per 2x2 group of its latent tokens).
    ///   - encoderMask: `true` at the caption positions the prompt fills, or nil when it fills them all.
    /// - Returns: the projection of the whole joint sequence, `[sequence, outChannels]`. The target
    ///   image's rows are the ones a sampler steps with.
    public func callAsFunction(latents: MLXArray, encoderHidden: MLXArray, timestep: Float,
                               imageShapes: [(frame: Int, height: Int, width: Int)],
                               imageMask: [Bool], encoderMask: [Bool]? = nil) -> MLXArray {
        callAsFunction(latents: latents, encoderHidden: encoderHidden, timestep: timestep,
                       imageShapes: imageShapes, imageMask: imageMask, encoderMask: encoderMask,
                       observer: nil)
    }

    /// The same forward, reporting each stage to `observer`. The parity harness localizes a divergence
    /// to a stage with it rather than reading it off the output.
    func callAsFunction(latents: MLXArray, encoderHidden: MLXArray, timestep: Float,
                        imageShapes: [(frame: Int, height: Int, width: Int)],
                        imageMask: [Bool], encoderMask: [Bool]?,
                        observer: ((String, MLXArray) -> Void)?) -> MLXArray {
        let (padMask, imageIds, targetMask) = Self.tokenMetadata(imageMask: imageMask,
                                                                 imageShapes: imageShapes)
        let sequence = padMask.count

        // The caption, then a zero row per 2x2 group of the target image, expanded four-fold at every
        // image slot and filled with the projected latents.
        let projectedText = textIn(encoderHidden)
        observer?("txt_in", projectedText)
        let targetTokens = imageShapes[imageShapes.count - 1].frame
            * imageShapes[imageShapes.count - 1].height * imageShapes[imageShapes.count - 1].width
        // The zero rows are the PROJECTED width: the caption is projected before the target image's
        // slots are appended to it.
        let padded = concatenated(
            [projectedText, MLXArray.zeros([targetTokens / 4, projectedText.dim(1)])
                .asType(projectedText.dtype)], axis: 0)
        var source = [Int32]()
        for (slot, isImage) in imageMask.enumerated() {
            source.append(contentsOf: [Int32](repeating: Int32(slot), count: isImage ? 4 : 1))
        }
        var expanded = padded.take(MLXArray(source), axis: 0)
        // The latents take the image positions. A gather and a select rather than a scatter, which is
        // how the vision splice is written elsewhere here.
        var latentIndex = [Int32](repeating: 0, count: sequence)
        var taken: Int32 = 0
        for (index, isImage) in padMask.enumerated() where isImage {
            latentIndex[index] = taken
            taken += 1
        }
        let projectedLatents = imageIn(latents)
        observer?("img_in", projectedLatents)
        let placed = projectedLatents.take(MLXArray(latentIndex), axis: 0)
        let isImageToken = MLXArray(padMask.map { $0 ? Float(1) : 0 }).reshaped([sequence, 1]) .> 0
        expanded = MLX.where(isImageToken, placed, expanded)

        let (cosine, sine) = rope.table(NFKQwenImageRope.positions(imageShapes: imageShapes,
                                                                  imagePadMask: padMask))

        // Two timestep rows: the sampled step, and zero. Text and condition-image tokens read the zero
        // row, the target image's tokens their own.
        // The reference casts the timestep to the model's dtype BEFORE the sinusoidal projection, so a
        // bfloat16 forward modulates from the rounded step rather than the caller's float32 one. The
        // projection itself is float32 either way, and its result enters the embedder at the model's
        // dtype.
        let dtype = latents.dtype
        let steps = (configuration.causalCondition ? MLXArray([timestep, 0]) : MLXArray([timestep]))
            .asType(dtype).asType(.float32)
        let temb = timeTextEmbed.timestepEmbedder(
            NFKQwenImageTimeTextEmbed.projection(steps).asType(dtype))
        observer?("temb", temb[0])
        let rows = configuration.causalCondition
            ? MLXArray(targetMask.map { $0 ? Int32(0) : Int32(1) })
            : MLXArray([Int32](repeating: 0, count: sequence))
        let modulated = modulation(NFKReferenceRounding.silu(temb))
        observer?("modulation", modulated)
        let perToken = modulated.take(rows, axis: 0)                            // [sequence, 4·dim]

        var keyValid: [Bool]?
        if let encoderMask {
            // The mask is over the vision-language sequence, which carries the condition images' slots
            // among the caption's. Only its TEXT entries reach the joint sequence, in order, because
            // the image slots of the joint sequence take their validity from the latents rather than
            // from the prompt.
            var textValid = [Bool]()
            for (slot, valid) in encoderMask.enumerated() where !(slot < imageMask.count && imageMask[slot]) {
                textValid.append(valid)
            }
            var valid = [Bool](repeating: true, count: sequence)
            var caption = 0
            for (index, isImage) in padMask.enumerated() where !isImage {
                if caption < textValid.count {
                    valid[index] = textValid[caption]
                    caption += 1
                }
            }
            keyValid = valid
        }
        let mask = Self.attentionMask(imageIds: imageIds, keyValid: keyValid)

        var hidden = expanded
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden, modulation: perToken, cos: cosine, sin: sine, mask: mask)
            observer?("block_\(index)", hidden)
        }

        let scale = normOut(NFKReferenceRounding.silu(temb).asType(dtype)).take(rows, axis: 0)       // [sequence, dim]
        let normalized = NFKQwenImageBlock.normalized(hidden, eps: configuration.eps) * (1 + scale)
        observer?("norm_out", normalized)
        return projectionOut(normalized)
    }
}

/// Building and loading the Qwen-Image 2.1 transformer.
@objc(NFKMLXQwenImage)
public final class NFKMLXQwenImage: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "qwen-image-2.1"

    /// Builds the transformer at a geometry.
    public static func makeNet(_ configuration: NFKMLXQwenImageConfiguration = .base)
        -> NFKMLXQwenImageNet {
        NFKMLXQwenImageNet(configuration)
    }

    /// The geometry a release's `transformer/config.json` describes.
    public static func configuration(fromHuggingFace url: URL) throws
        -> NFKMLXQwenImageConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("the transformer config is not an object")
        }
        var configuration = NFKMLXQwenImageConfiguration()
        configuration.patchSize = json["patch_size"] as? Int ?? configuration.patchSize
        configuration.inChannels = json["in_channels"] as? Int ?? configuration.inChannels
        configuration.outChannels = json["out_channels"] as? Int ?? configuration.inChannels
        configuration.layers = json["num_layers"] as? Int ?? configuration.layers
        configuration.headDimensions = json["attention_head_dim"] as? Int ?? configuration.headDimensions
        configuration.heads = json["num_attention_heads"] as? Int ?? configuration.heads
        configuration.contextInDimensions = json["context_in_dim"] as? Int
            ?? configuration.contextInDimensions
        configuration.mlpRatio = json["mlp_ratio"] as? Int ?? configuration.mlpRatio
        if let axes = json["axes_dims_rope"] as? [Int] { configuration.axesDimensionsRope = axes }
        if let eps = json["eps"] as? NSNumber { configuration.eps = eps.floatValue }
        configuration.causalCondition = (json["causal_condition"] as? NSNumber)?.boolValue
            ?? configuration.causalCondition
        return configuration
    }

    /// The release's keys are the module names, with two exceptions: the shared modulation is the
    /// second element of a `Sequential` whose first is an activation, and the final adaptive norm wraps
    /// its projection in a `linear`.
    public static func remapReferenceKey(_ key: String) -> String? {
        if key == "modulation.1.weight" { return "modulation.weight" }
        if key == "norm_out.linear.weight" { return "norm_out.weight" }
        return key
    }

    /// Loads a release's `transformer/` weights, sharded or not.
    ///
    /// @discussion The release is 7.1B parameters in bfloat16. A `.float32` load doubles that to 28 GB,
    /// which a 32 GB machine does not hold beside its activations, so a consumer loads `.checkpoint`
    /// and runs at the precision the weights ship in.
    public static func loadWeights(into net: NFKMLXQwenImageNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .checkpoint) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision,
                                                     remap: remapReferenceKey)
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
    }
}
