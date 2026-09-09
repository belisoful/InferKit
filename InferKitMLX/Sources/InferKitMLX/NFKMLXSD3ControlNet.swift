//
//  NFKMLXSD3ControlNet.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// The Stable Diffusion 3 ControlNet (`SD3ControlNetModel`, Stability AI / InstantX), a partial copy of
// the MMDiT that steers a generation with a spatial control image (Canny, depth, pose, blur, tile). It
// is NOT a standalone denoiser: it runs the first N joint blocks over the noisy latent PLUS a control
// latent, and emits one per-block residual (a zero-initialized linear projection of each block's output)
// that the base ``NFKMLXSD3TransformerNet`` adds into its own blocks. Because the projections start at
// zero, an untrained ControlNet is a no-op, and the base transformer is byte-identical without one.
//
// Two released shapes, both handled here:
//
//  - The InstantX SD3-medium / SD3.5-large ControlNets carry a `context_embedder` and reuse the full
//    dual-stream `JointTransformerBlock`, so they read the text conditioning like the base transformer.
//  - Stability's official SD3.5-large 8B ControlNets (Blur, Canny, Depth) drop the position embedding
//    and the context embedder and run single-stream `SD3SingleTransformerBlock`s over the image tokens
//    alone. Their `hidden_states` arrive already patch-embedded (the base transformer's `pos_embed`
//    supplies them), so the pipeline runs the base patch embed once and hands the 3-D tokens in.
//
// The control image is a VAE-encoded latent, patch-embedded through the zero-initialized `pos_embed_input`
// (which carries NO positional table) and added to the (patch-embedded) noisy latent before the blocks.

/// A stride-`patch` patch embedding with NO positional table, the reference's `PatchEmbed(pos_embed_type:
/// None)` for `pos_embed_input`. A `[B, C, H, W]` latent → `[B, (H/p)·(W/p), inner]` tokens.
final class NFKSD3InputPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    let patchSize: Int

    init(inChannels: Int, innerDim: Int, patchSize: Int) {
        self.patchSize = patchSize
        _proj.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: innerDim,
                                    kernelSize: IntOrPair(patchSize), stride: IntOrPair(patchSize))
    }

    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let h = latent.dim(2) / patchSize, w = latent.dim(3) / patchSize
        let convolved = proj(latent.transposed(0, 2, 3, 1))                // [B, H/p, W/p, inner]
        return convolved.reshaped([convolved.dim(0), h * w, convolved.dim(3)])
    }
}

/// One `SD3SingleTransformerBlock`: image-only adaptive-norm self-attention and a feed-forward, no text
/// stream, no qk-norm, no dual attention. The modulation is `AdaLayerNormZero`'s six chunks, the same
/// layout as the joint block's image stream.
final class NFKSD3SingleBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: NFKSD3AdaLinear                   // 6·dim
    @ModuleInfo(key: "attn") var attn: NFKSD3Attention                    // image-only, no qk-norm
    @ModuleInfo(key: "ff") var ff: NFKSD3FeedForward

    let dim: Int

    init(dim: Int, heads: Int, headDim: Int) {
        self.dim = dim
        _norm1.wrappedValue = NFKSD3AdaLinear(dim, 6 * dim)
        _attn.wrappedValue = NFKSD3Attention(dim: dim, heads: heads, headDim: headDim,
                                             qkNorm: false, added: false, contextPreOnly: false)
        _ff.wrappedValue = NFKSD3FeedForward(dim)
    }

    func callAsFunction(_ hidden: MLXArray, temb: MLXArray) -> MLXArray {
        var hiddenStates = hidden
        let mod = norm1(temb)                                             // [B, 6·dim]
        func chunk(_ i: Int) -> MLXArray { mod[0..., (i * dim) ..< ((i + 1) * dim)][0..., .newAxis, 0...] }
        let shiftMSA = chunk(0), scaleMSA = chunk(1), gateMSA = chunk(2)
        let shiftMLP = chunk(3), scaleMLP = chunk(4), gateMLP = chunk(5)
        let normHidden = sd3AffineFreeLayerNorm(hiddenStates) * (1 + scaleMSA) + shiftMSA
        let (attnOut, _) = attn(normHidden, encoder: nil)
        hiddenStates = hiddenStates + gateMSA * attnOut
        let normFF = sd3AffineFreeLayerNorm(hiddenStates) * (1 + scaleMLP) + shiftMLP
        hiddenStates = hiddenStates + gateMLP * ff(normFF)
        return hiddenStates
    }
}

/// SD3 ControlNet geometry. The base fields mirror ``NFKMLXSD3Configuration``; `extraConditioningChannels`
/// widens `pos_embed_input`, and `usePosEmbed`/`useContextEmbedder` select the InstantX (dual-stream) or
/// the Stability 8B (single-stream) shape.
public struct NFKMLXSD3ControlNetConfiguration: Sendable {
    public var sampleSize: Int
    public var patchSize: Int
    public var inChannels: Int
    public var numLayers: Int
    public var attentionHeadDim: Int
    public var numAttentionHeads: Int
    public var jointAttentionDim: Int
    public var pooledProjectionDim: Int
    public var posEmbedMaxSize: Int
    public var dualAttentionLayers: [Int]
    public var qkNorm: Bool
    public var extraConditioningChannels: Int
    public var usePosEmbed: Bool
    public var useContextEmbedder: Bool

    public init(sampleSize: Int = 128, patchSize: Int = 2, inChannels: Int = 16, numLayers: Int = 18,
                attentionHeadDim: Int = 64, numAttentionHeads: Int = 18, jointAttentionDim: Int = 4096,
                pooledProjectionDim: Int = 2048, posEmbedMaxSize: Int = 96, dualAttentionLayers: [Int] = [],
                qkNorm: Bool = false, extraConditioningChannels: Int = 0, usePosEmbed: Bool = true,
                useContextEmbedder: Bool = true) {
        self.sampleSize = sampleSize
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.numLayers = numLayers
        self.attentionHeadDim = attentionHeadDim
        self.numAttentionHeads = numAttentionHeads
        self.jointAttentionDim = jointAttentionDim
        self.pooledProjectionDim = pooledProjectionDim
        self.posEmbedMaxSize = posEmbedMaxSize
        self.dualAttentionLayers = dualAttentionLayers
        self.qkNorm = qkNorm
        self.extraConditioningChannels = extraConditioningChannels
        self.usePosEmbed = usePosEmbed
        self.useContextEmbedder = useContextEmbedder
    }

    /// The InstantX SD3-medium ControlNets (Canny, Pose, Tile): a dozen dual-stream blocks reading the
    /// text conditioning, no qk-norm.
    public static let instantXMedium = NFKMLXSD3ControlNetConfiguration(numLayers: 12, numAttentionHeads: 24)

    /// Stability's official SD3.5-large 8B ControlNets (Blur, Canny, Depth): single-stream blocks over
    /// the image tokens alone, no position embedding, no context embedder, RMS qk-norm, one extra
    /// conditioning channel.
    public static let stabilitySD35Large = NFKMLXSD3ControlNetConfiguration(
        numLayers: 18, attentionHeadDim: 64, numAttentionHeads: 38, qkNorm: true,
        extraConditioningChannels: 1, usePosEmbed: false, useContextEmbedder: false)

    /// A tiny random configuration for reference parity: four dual-stream blocks (so the base's
    /// `interval_control` striding is exercised with two residuals over a four-block base), two heads of
    /// eight, RMS qk-norm, a small position grid the crop is exercised on.
    public static let tiny = NFKMLXSD3ControlNetConfiguration(
        sampleSize: 16, inChannels: 4, numLayers: 2, attentionHeadDim: 8, numAttentionHeads: 2,
        jointAttentionDim: 24, pooledProjectionDim: 20, posEmbedMaxSize: 8, qkNorm: true)

    /// A tiny single-stream configuration for the Stability 8B path: two single blocks, no position
    /// embedding, no context embedder.
    public static let tinySingle = NFKMLXSD3ControlNetConfiguration(
        sampleSize: 16, inChannels: 4, numLayers: 2, attentionHeadDim: 8, numAttentionHeads: 2,
        pooledProjectionDim: 20, posEmbedMaxSize: 8, qkNorm: false, extraConditioningChannels: 1,
        usePosEmbed: false, useContextEmbedder: false)

    var innerDim: Int { numAttentionHeads * attentionHeadDim }
}

/// The Stable Diffusion 3 ControlNet. `callAsFunction` returns the per-block residuals to feed the base
/// transformer's `blockControlnetHiddenStates`.
public final class NFKMLXSD3ControlNetNet: Module {
    @ModuleInfo(key: "pos_embed") var posEmbed: NFKSD3PatchEmbed?
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: NFKSD3TimeTextEmbed
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear?
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Module]
    @ModuleInfo(key: "controlnet_blocks") var controlnetBlocks: [Linear]
    @ModuleInfo(key: "pos_embed_input") var posEmbedInput: NFKSD3InputPatchEmbed

    public let config: NFKMLXSD3ControlNetConfiguration

    public init(_ config: NFKMLXSD3ControlNetConfiguration) {
        self.config = config
        let dim = config.innerDim
        if config.usePosEmbed {
            _posEmbed.wrappedValue = NFKSD3PatchEmbed(NFKMLXSD3Configuration(
                sampleSize: config.sampleSize, patchSize: config.patchSize, inChannels: config.inChannels,
                numLayers: config.numLayers, attentionHeadDim: config.attentionHeadDim,
                numAttentionHeads: config.numAttentionHeads, jointAttentionDim: config.jointAttentionDim,
                captionProjectionDim: dim, pooledProjectionDim: config.pooledProjectionDim,
                posEmbedMaxSize: config.posEmbedMaxSize, dualAttentionLayers: config.dualAttentionLayers,
                qkNorm: config.qkNorm))
        }
        _timeTextEmbed.wrappedValue = NFKSD3TimeTextEmbed(NFKMLXSD3Configuration(
            attentionHeadDim: config.attentionHeadDim, numAttentionHeads: config.numAttentionHeads,
            pooledProjectionDim: config.pooledProjectionDim))
        if config.useContextEmbedder {
            _contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, dim)
        }
        // The base config for the joint blocks (all context_pre_only=false in a ControlNet).
        let blockConfig = NFKMLXSD3Configuration(
            attentionHeadDim: config.attentionHeadDim, numAttentionHeads: config.numAttentionHeads,
            captionProjectionDim: dim, qkNorm: config.qkNorm)
        if config.useContextEmbedder {
            _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { index in
                NFKSD3JointBlock(blockConfig, contextPreOnly: false,
                                 useDualAttention: config.dualAttentionLayers.contains(index))
            }
        } else {
            _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in
                NFKSD3SingleBlock(dim: dim, heads: config.numAttentionHeads, headDim: config.attentionHeadDim)
            }
        }
        _controlnetBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in Linear(dim, dim) }
        _posEmbedInput.wrappedValue = NFKSD3InputPatchEmbed(
            inChannels: config.inChannels + config.extraConditioningChannels, innerDim: dim,
            patchSize: config.patchSize)
    }

    /// Runs the ControlNet and returns its per-block residuals, scaled by `conditioningScale`.
    ///
    /// `hidden` is the noisy latent — `[B, C, H, W]` for the InstantX (position-embedded) shape, or the
    /// already patch-embedded `[B, S, inner]` tokens for the Stability 8B shape. `controlnetCond` is the
    /// VAE-encoded control latent `[B, C, H, W]`. `encoder`/`pooled` are the caption and pooled text
    /// projections (the InstantX shape reads the caption; the 8B shape ignores it).
    public func callAsFunction(_ hidden: MLXArray, controlnetCond: MLXArray, encoder: MLXArray?,
                               pooled: MLXArray, timestep: MLXArray, conditioningScale: Float = 1) -> [MLXArray] {
        var image = posEmbed.map { $0(hidden) } ?? hidden                  // [B, S, inner]
        let temb = timeTextEmbed(timestep: timestep, pooled: pooled)
        var context = contextEmbedder.map { $0(encoder!) }                 // [B, L, inner]?
        image = image + posEmbedInput(controlnetCond)

        var residuals: [MLXArray] = []
        for block in transformerBlocks {
            if let joint = block as? NFKSD3JointBlock {
                let (newContext, newImage) = joint(image, encoder: context!, temb: temb)
                image = newImage
                context = newContext
            } else if let single = block as? NFKSD3SingleBlock {
                image = single(image, temb: temb)
            }
            residuals.append(image)
        }
        return residuals.enumerated().map { index, residual in
            controlnetBlocks[index](residual) * conditioningScale
        }
    }

    /// The geometry of a released SD3 ControlNet, from its `config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXSD3ControlNetConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "SD3ControlNetModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads an SD3 ControlNet, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let dual = (json["dual_attention_layers"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue } ?? []
        return NFKMLXSD3ControlNetConfiguration(
            sampleSize: integer("sample_size", 128), patchSize: integer("patch_size", 2),
            inChannels: integer("in_channels", 16), numLayers: integer("num_layers", 18),
            attentionHeadDim: integer("attention_head_dim", 64),
            numAttentionHeads: integer("num_attention_heads", 18),
            jointAttentionDim: integer("joint_attention_dim", 4096),
            pooledProjectionDim: integer("pooled_projection_dim", 2048),
            posEmbedMaxSize: integer("pos_embed_max_size", 96), dualAttentionLayers: dual,
            qkNorm: (json["qk_norm"] as? String) != nil,
            extraConditioningChannels: integer("extra_conditioning_channels", 0),
            usePosEmbed: (json["use_pos_embed"] as? NSNumber)?.boolValue ?? true,
            useContextEmbedder: (json["joint_attention_dim"] as? NSNumber) != nil)
    }

    /// Loads a released SD3 ControlNet directory into `net`. Only the 4-D patch-embed convolutions
    /// transpose to MLX's NHWC.
    public static func loadWeights(into net: NFKMLXSD3ControlNetNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        let weights = arrays.map { name, value -> (String, MLXArray) in
            (name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)
    }
}
