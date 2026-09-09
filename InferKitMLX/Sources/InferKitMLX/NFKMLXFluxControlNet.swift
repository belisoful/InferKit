//
//  NFKMLXFluxControlNet.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// The FLUX.1 ControlNet (`FluxControlNetModel`, Black Forest Labs / InstantX / Shakker-Labs), a partial
// copy of the FLUX transformer that steers a generation with a spatial control image. Like the SD3
// ControlNet it is not a standalone denoiser: it runs a few double-stream and single-stream blocks over
// the packed noisy latent PLUS a control latent, and emits one zero-initialized residual PER block — a
// double-block list and a single-block list — that the base ``NFKMLXFluxTransformerNet`` adds into its
// own two block stacks. The residual projections start at zero, so an untrained ControlNet is a no-op.
//
// The control image reaches the ControlNet one of two ways. Most released InstantX / Shakker-Labs
// ControlNets take a VAE-encoded, packed control latent, added to the (embedded) noisy latent through
// the zero-initialized `controlnet_x_embedder`. The `input_hint_block` shape instead takes a
// FULL-RESOLUTION control image and runs it through a small convolutional pyramid (four 3×3 stages, three
// stride-2 downsamples matching the VAE's 8× reduction) before the linear embed. The union variant
// (`num_mode`) adds a learned mode embedding prepended to the text sequence, so one ControlNet serves
// several control types (Canny, depth, pose).

/// The `ControlNetConditioningEmbedding` a `input_hint_block` FLUX ControlNet carries: a `conv_in`, three
/// stride-2 downsampling stages (a `(16, 16, 16, 16)` channel pyramid, each stage a same-size 3×3 then a
/// stride-2 3×3), and a zero-initialized `conv_out`, SiLU between every convolution. A full-resolution
/// control image `[B, C, H·8, W·8]` → `[B, H·W, condEmbedChannels]` tokens.
final class NFKFluxControlNetHintEmbedding: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "blocks") var blocks: [Conv2d]
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(conditioningChannels: Int, conditioningEmbeddingChannels: Int, blockOutChannels: [Int] = [16, 16, 16, 16]) {
        _convIn.wrappedValue = Conv2d(inputChannels: conditioningChannels, outputChannels: blockOutChannels[0],
                                      kernelSize: IntOrPair(3), padding: IntOrPair(1))
        var convs: [Conv2d] = []
        for i in 0 ..< (blockOutChannels.count - 1) {
            let cin = blockOutChannels[i], cout = blockOutChannels[i + 1]
            convs.append(Conv2d(inputChannels: cin, outputChannels: cin, kernelSize: IntOrPair(3), padding: IntOrPair(1)))
            convs.append(Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: IntOrPair(3),
                                stride: IntOrPair(2), padding: IntOrPair(1)))
        }
        _blocks.wrappedValue = convs
        _convOut.wrappedValue = Conv2d(inputChannels: blockOutChannels.last!,
                                       outputChannels: conditioningEmbeddingChannels,
                                       kernelSize: IntOrPair(3), padding: IntOrPair(1))
    }

    /// `image` `[B, C, H, W]` (NCHW) → `[B, (H/8)·(W/8), condEmbedChannels]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        var embedding = silu(convIn(image.transposed(0, 2, 3, 1)))          // NHWC
        for block in blocks {
            embedding = silu(block(embedding))
        }
        embedding = convOut(embedding)                                     // [B, H', W', cec]
        return embedding.reshaped([embedding.dim(0), embedding.dim(1) * embedding.dim(2), embedding.dim(3)])
    }
}

/// FLUX ControlNet geometry. The base fields mirror ``NFKMLXFluxConfiguration``; `numMode` selects the
/// union variant (a learned control-type embedding).
public struct NFKMLXFluxControlNetConfiguration: Sendable {
    public var inChannels: Int
    public var numLayers: Int
    public var numSingleLayers: Int
    public var attentionHeadDim: Int
    public var numAttentionHeads: Int
    public var jointAttentionDim: Int
    public var pooledProjectionDim: Int
    public var guidanceEmbeds: Bool
    public var axesDimsRope: [Int]
    public var numMode: Int?
    public var conditioningEmbeddingChannels: Int?
    public var conditioningChannels: Int

    public init(inChannels: Int = 64, numLayers: Int = 5, numSingleLayers: Int = 0,
                attentionHeadDim: Int = 128, numAttentionHeads: Int = 24, jointAttentionDim: Int = 4096,
                pooledProjectionDim: Int = 768, guidanceEmbeds: Bool = false,
                axesDimsRope: [Int] = [16, 56, 56], numMode: Int? = nil,
                conditioningEmbeddingChannels: Int? = nil, conditioningChannels: Int = 3) {
        self.inChannels = inChannels
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.attentionHeadDim = attentionHeadDim
        self.numAttentionHeads = numAttentionHeads
        self.jointAttentionDim = jointAttentionDim
        self.pooledProjectionDim = pooledProjectionDim
        self.guidanceEmbeds = guidanceEmbeds
        self.axesDimsRope = axesDimsRope
        self.numMode = numMode
        self.conditioningEmbeddingChannels = conditioningEmbeddingChannels
        self.conditioningChannels = conditioningChannels
    }

    /// The InstantX / Shakker-Labs FLUX.1-dev ControlNet-Union-Pro: five double blocks, no single
    /// blocks, guidance-distilled, ten control modes.
    public static let unionPro = NFKMLXFluxControlNetConfiguration(
        numLayers: 5, numSingleLayers: 0, guidanceEmbeds: true, numMode: 10)

    /// A single-control-type FLUX.1-dev ControlNet (Canny, depth), guidance-distilled, no union embedding.
    public static let single = NFKMLXFluxControlNetConfiguration(
        numLayers: 6, numSingleLayers: 0, guidanceEmbeds: true)

    /// A tiny random configuration for reference parity: two double and two single blocks (so the base's
    /// `ceil` interval striding is exercised with two residuals over a three-block base), two heads of
    /// six, the three-axis rope, guidance-distilled.
    public static let tiny = NFKMLXFluxControlNetConfiguration(
        inChannels: 8, numLayers: 2, numSingleLayers: 2, attentionHeadDim: 6, numAttentionHeads: 2,
        jointAttentionDim: 24, pooledProjectionDim: 10, guidanceEmbeds: true, axesDimsRope: [2, 2, 2])

    /// A tiny random configuration exercising the `input_hint_block` (a full-resolution control image is
    /// downsampled 8× through the conditioning pyramid before the linear embed).
    public static let tinyHint = NFKMLXFluxControlNetConfiguration(
        inChannels: 8, numLayers: 2, numSingleLayers: 2, attentionHeadDim: 6, numAttentionHeads: 2,
        jointAttentionDim: 24, pooledProjectionDim: 10, guidanceEmbeds: true, axesDimsRope: [2, 2, 2],
        conditioningEmbeddingChannels: 8)

    var innerDim: Int { numAttentionHeads * attentionHeadDim }
}

/// The FLUX.1 ControlNet. `callAsFunction` returns the double-block and single-block residuals to feed
/// the base transformer's `controlnetBlockSamples` / `controlnetSingleBlockSamples`.
public final class NFKMLXFluxControlNetNet: Module {
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "controlnet_x_embedder") var controlnetXEmbedder: Linear
    @ModuleInfo(key: "controlnet_mode_embedder") var controlnetModeEmbedder: Embedding?
    @ModuleInfo(key: "input_hint_block") var inputHintBlock: NFKFluxControlNetHintEmbedding?
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: NFKFluxTimeTextEmbed
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [NFKFluxDoubleBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [NFKFluxSingleBlock]
    @ModuleInfo(key: "controlnet_blocks") var controlnetBlocks: [Linear]
    @ModuleInfo(key: "controlnet_single_blocks") var controlnetSingleBlocks: [Linear]

    public let config: NFKMLXFluxControlNetConfiguration
    let rope: NFKFluxRope

    public init(_ config: NFKMLXFluxControlNetConfiguration) {
        self.config = config
        self.rope = NFKFluxRope(axesDim: config.axesDimsRope, theta: 10000)
        let dim = config.innerDim
        let base = NFKMLXFluxConfiguration(
            inChannels: config.inChannels, outChannels: config.inChannels, numLayers: config.numLayers,
            numSingleLayers: config.numSingleLayers, attentionHeadDim: config.attentionHeadDim,
            numAttentionHeads: config.numAttentionHeads, jointAttentionDim: config.jointAttentionDim,
            pooledProjectionDim: config.pooledProjectionDim, guidanceEmbeds: config.guidanceEmbeds,
            axesDimsRope: config.axesDimsRope)
        _xEmbedder.wrappedValue = Linear(config.inChannels, dim)
        _contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, dim)
        _controlnetXEmbedder.wrappedValue = Linear(config.inChannels, dim)
        _controlnetModeEmbedder.wrappedValue = config.numMode.map { Embedding(embeddingCount: $0, dimensions: dim) }
        _inputHintBlock.wrappedValue = config.conditioningEmbeddingChannels.map {
            NFKFluxControlNetHintEmbedding(conditioningChannels: config.conditioningChannels,
                                           conditioningEmbeddingChannels: $0)
        }
        _timeTextEmbed.wrappedValue = NFKFluxTimeTextEmbed(base)
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in NFKFluxDoubleBlock(base) }
        _singleTransformerBlocks.wrappedValue = (0 ..< config.numSingleLayers).map { _ in NFKFluxSingleBlock(base) }
        _controlnetBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in Linear(dim, dim) }
        _controlnetSingleBlocks.wrappedValue = (0 ..< config.numSingleLayers).map { _ in Linear(dim, dim) }
    }

    /// Runs the ControlNet and returns `(double, single)` residual lists, scaled by `conditioningScale`.
    ///
    /// `hidden` is the packed noisy latent `[B, imgSeq, inChannels]`. `controlnetCond` is the packed
    /// control latent `[B, imgSeq, inChannels]`, or — when `input_hint_block` is present — the
    /// full-resolution control image `[B, C, H·8, W·8]` the pyramid downsamples. `controlnetMode` `[B]`
    /// selects the control type for a union ControlNet (nil otherwise).
    public func callAsFunction(_ hidden: MLXArray, controlnetCond: MLXArray, encoder: MLXArray,
                               pooled: MLXArray, timestep: MLXArray, guidance: MLXArray?, imageIds: MLXArray,
                               controlnetMode: MLXArray? = nil, conditioningScale: Float = 1)
        -> (double: [MLXArray], single: [MLXArray]) {
        var image = xEmbedder(hidden)                                      // [B, imgSeq, inner]
        let controlTokens = inputHintBlock.map { $0(controlnetCond) } ?? controlnetCond
        image = image + controlnetXEmbedder(controlTokens)
        let temb = timeTextEmbed(timestep: timestep * 1000, guidance: guidance.map { $0 * 1000 }, pooled: pooled)
        var context = contextEmbedder(encoder)                             // [B, txtSeq, inner]

        var textLength = encoder.dim(1)
        var textIds = MLXArray.zeros([textLength, imageIds.dim(1)])
        if let modeEmbedder = controlnetModeEmbedder, let controlnetMode {
            let modeEmb = modeEmbedder(controlnetMode).reshaped([context.dim(0), 1, context.dim(2)])
            context = concatenated([modeEmb, context], axis: 1)            // prepend the mode token
            textIds = concatenated([textIds[0 ..< 1, 0...], textIds], axis: 0)
            textLength += 1
        }
        let ids = concatenated([textIds, imageIds], axis: 0)
        let (cos, sin) = rope.table(ids: ids)

        var doubleResiduals: [MLXArray] = []
        for block in transformerBlocks {
            let (newContext, newImage) = block(image, encoder: context, temb: temb, cos: cos, sin: sin)
            context = newContext
            image = newImage
            doubleResiduals.append(image)
        }
        var singleResiduals: [MLXArray] = []
        for block in singleTransformerBlocks {
            let (newContext, newImage) = block(image, encoder: context, temb: temb, cos: cos, sin: sin)
            context = newContext
            image = newImage
            singleResiduals.append(image)
        }

        let double = doubleResiduals.enumerated().map { index, residual in
            controlnetBlocks[index](residual) * conditioningScale
        }
        let single = singleResiduals.enumerated().map { index, residual in
            controlnetSingleBlocks[index](residual) * conditioningScale
        }
        return (double, single)
    }

    /// The geometry of a released FLUX ControlNet, from its `config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXFluxControlNetConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "FluxControlNetModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads a FLUX ControlNet, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let axes = (json["axes_dims_rope"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue } ?? [16, 56, 56]
        return NFKMLXFluxControlNetConfiguration(
            inChannels: integer("in_channels", 64), numLayers: integer("num_layers", 5),
            numSingleLayers: integer("num_single_layers", 0),
            attentionHeadDim: integer("attention_head_dim", 128),
            numAttentionHeads: integer("num_attention_heads", 24),
            jointAttentionDim: integer("joint_attention_dim", 4096),
            pooledProjectionDim: integer("pooled_projection_dim", 768),
            guidanceEmbeds: (json["guidance_embeds"] as? NSNumber)?.boolValue ?? false,
            axesDimsRope: axes, numMode: (json["num_mode"] as? NSNumber)?.intValue,
            conditioningEmbeddingChannels: (json["conditioning_embedding_channels"] as? NSNumber)?.intValue,
            conditioningChannels: integer("conditioning_channels", 3))
    }

    /// Loads a released FLUX ControlNet directory into `net`. Only the `input_hint_block`'s 4-D
    /// convolutions transpose to MLX's NHWC; every other weight is at most 2-D.
    public static func loadWeights(into net: NFKMLXFluxControlNetNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        let weights = arrays.map { name, value -> (String, MLXArray) in
            (name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)
    }
}
