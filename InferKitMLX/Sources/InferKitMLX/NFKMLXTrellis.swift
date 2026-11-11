//
//  NFKMLXTrellis.swift
//  InferKitMLX
//
//  TRELLIS (microsoft/TRELLIS-image-large): the DINOv2 image encoder with registers, the sparse-structure
//  flow transformer over the dense 16³ latent, and the sparse-structure decoder to 64³ occupancy logits.
//  Tensors are channels last: a dense volume is `[N, X, Y, Z, C]`.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// The geometry of a TRELLIS image-to-3D release. `.released` is microsoft/TRELLIS-image-large.
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrellisConfiguration: Sendable {
    /// The image encoder's input side; the release encodes 518 × 518.
    public var imageSize = 518
    public var patchSize = 14
    public var encoderWidth = 1024
    public var encoderDepth = 24
    public var encoderHeads = 16
    public var encoderHidden = 4096
    public var registerTokens = 4
    /// The side of the patch grid the encoder's position embeddings were trained at.
    public var encoderPositionGrid = 37

    /// The dense structure latent's side and channels.
    public var structureResolution = 16
    public var structureChannels = 8
    public var flowWidth = 1024
    public var flowBlocks = 24
    public var flowHeads = 16
    public var flowHidden = 4096
    public var frequencyWidth = 256
    /// The condition tokens' width: DINOv2's 1024 for the image release, CLIP's 768 for the text ones.
    public var conditionWidth = 1024
    /// Whether the structure flow's cross-attention normalizes its queries and keys per head (text-xlarge's
    /// does; its latent flow's does not).
    public var crossAttentionHeadNorm = false
    /// Whether the latent flow's cross-attention normalizes its queries and keys per head.
    public var latentCrossAttentionHeadNorm = false

    /// The structure decoder's widths from the latent side up; each step after the first doubles the side.
    public var decoderChannels = [512, 128, 32]
    public var decoderResidualBlocks = 2
    public var decoderMiddleBlocks = 2

    /// The structured latent's channels, and the latent flow's width on the full-resolution voxels.
    public var latentChannels = 8
    public var latentFlowIOWidth = 128
    /// Residual blocks on the full-resolution voxels on each side of the transformer.
    public var latentFlowIOBlocks = 2

    /// The structured-latent decoders' swin trunk.
    public var decoderWidth = 768
    public var decoderBlocks = 12
    public var decoderHeads = 12
    public var decoderWindow = 8
    /// The mesh decoder's features per voxel at four times the latent resolution: SDF (8 corners),
    /// deformations (8 × 3), FlexiCubes weights (21), and colors (8 × 6).
    public var meshFeatureChannels = 101
    public var gaussiansPerVoxel = 32
    /// How far a Gaussian may sit from its voxel center, in voxels across.
    public var gaussianVoxelSpan: Float = 1.5
    /// The scale a zero output decodes to, the smallest scale, and the opacity a zero output decodes to.
    public var gaussianScaleBias: Float = 0.004
    public var gaussianMinimumScale: Float = 0.0009
    public var gaussianOpacityBias: Float = 0.1

    public init() {}

    public static let released = NFKMLXTrellisConfiguration()

    /// microsoft/TRELLIS-text-base: 768-wide flows of 12 blocks on CLIP ViT-L/14 text tokens.
    public static let textBase: NFKMLXTrellisConfiguration = {
        var configuration = NFKMLXTrellisConfiguration()
        configuration.setFlows(width: 768, blocks: 12, heads: 12)
        return configuration
    }()

    /// microsoft/TRELLIS-text-large: 1024-wide flows of 24 blocks on CLIP ViT-L/14 text tokens.
    public static let textLarge: NFKMLXTrellisConfiguration = {
        var configuration = NFKMLXTrellisConfiguration()
        configuration.setFlows(width: 1024, blocks: 24, heads: 16)
        return configuration
    }()

    /// microsoft/TRELLIS-text-xlarge: 1280-wide flows of 28 blocks, per-head normalized cross-attention in
    /// the structure flow, and three 256-wide residual blocks on each side of the latent flow's transformer.
    public static let textXLarge: NFKMLXTrellisConfiguration = {
        var configuration = NFKMLXTrellisConfiguration()
        configuration.setFlows(width: 1280, blocks: 28, heads: 16)
        configuration.crossAttentionHeadNorm = true
        configuration.latentFlowIOWidth = 256
        configuration.latentFlowIOBlocks = 3
        return configuration
    }()

    mutating func setFlows(width: Int, blocks: Int, heads: Int) {
        flowWidth = width
        flowBlocks = blocks
        flowHeads = heads
        flowHidden = width * 4
        conditionWidth = 768
    }

    /// The occupancy grid's side, `structureResolution · 2^(decoderChannels.count − 1)`.
    public var occupancyResolution: Int { structureResolution << (decoderChannels.count - 1) }
    var patchGrid: Int { imageSize / patchSize }
}

/// A sequential slot that holds no parameters, so checkpoint indices line up with the layers around it.
final class NFKTrellisSlot: Module {}

// MARK: - Attention

enum NFKTrellisAttention {
    /// Scaled dot-product attention over `[N, L, H, D]` queries and `[N, S, H, D]` keys and values,
    /// returned as `[N, L, H · D]`.
    static func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let (batch, length, heads, width) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
            scale: 1 / Float(width).squareRoot(), mask: nil)
        return out.transposed(0, 2, 1, 3).reshaped([batch, length, heads * width])
    }

    /// LayerNorm without an affine transform.
    static func normalized(_ x: MLXArray, eps: Float) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
    }
}

/// Per-head RMS normalization of queries or keys: each head vector scaled to unit length (at least
/// 1e-12), times a learned per-head gain and √width.
final class NFKTrellisHeadNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float

    init(heads: Int, width: Int) {
        _gamma.wrappedValue = MLXArray.ones([heads, width])
        scale = Float(width).squareRoot()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = maximum(sqrt(sum(x * x, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return x / length * gamma * scale
    }
}

/// Multi-head self-attention with a fused query-key-value projection and optional per-head RMS
/// normalization of the queries and keys.
final class NFKTrellisSelfAttention: Module {
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_rms_norm") var queryNorm: NFKTrellisHeadNorm?
    @ModuleInfo(key: "k_rms_norm") var keyNorm: NFKTrellisHeadNorm?
    let heads: Int

    init(width: Int, heads: Int, headNorm: Bool) {
        self.heads = heads
        _toQKV.wrappedValue = Linear(width, width * 3)
        _toOut.wrappedValue = Linear(width, width)
        _queryNorm.wrappedValue = headNorm ? NFKTrellisHeadNorm(heads: heads, width: width / heads) : nil
        _keyNorm.wrappedValue = headNorm ? NFKTrellisHeadNorm(heads: heads, width: width / heads) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let qkv = toQKV(x).reshaped([batch, length, 3, heads, width / heads])
        var q = qkv[0..., 0..., 0], k = qkv[0..., 0..., 1]
        if let queryNorm, let keyNorm {
            q = queryNorm(q)
            k = keyNorm(k)
        }
        return toOut(NFKTrellisAttention.attend(q, k, qkv[0..., 0..., 2]))
    }
}

/// Multi-head cross-attention from the tokens to a condition sequence, with optional per-head RMS
/// normalization of the queries and keys.
final class NFKTrellisCrossAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_rms_norm") var queryNorm: NFKTrellisHeadNorm?
    @ModuleInfo(key: "k_rms_norm") var keyNorm: NFKTrellisHeadNorm?
    let heads: Int

    init(width: Int, contextWidth: Int, heads: Int, headNorm: Bool) {
        self.heads = heads
        _toQ.wrappedValue = Linear(width, width)
        _toKV.wrappedValue = Linear(contextWidth, width * 2)
        _toOut.wrappedValue = Linear(width, width)
        _queryNorm.wrappedValue = headNorm ? NFKTrellisHeadNorm(heads: heads, width: width / heads) : nil
        _keyNorm.wrappedValue = headNorm ? NFKTrellisHeadNorm(heads: heads, width: width / heads) : nil
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let headWidth = width / heads
        var q = toQ(x).reshaped([batch, length, heads, headWidth])
        let kv = toKV(context).reshaped([batch, context.dim(1), 2, heads, headWidth])
        var k = kv[0..., 0..., 0]
        if let queryNorm, let keyNorm {
            q = queryNorm(q)
            k = keyNorm(k)
        }
        return toOut(NFKTrellisAttention.attend(q, k, kv[0..., 0..., 1]))
    }
}

/// `mlp`: linear, tanh-approximated GELU, linear.
final class NFKTrellisFeedForward: Module {
    @ModuleInfo(key: "mlp") var layers: [Module]

    init(width: Int, hidden: Int) {
        _layers.wrappedValue = [Linear(width, hidden), NFKTrellisSlot(), Linear(hidden, width)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let first = layers[0] as! Linear, last = layers[2] as! Linear
        return last(geluApproximate(first(x)))
    }
}

/// A transformer block modulated by the timestep: adaptive shift, scale, and gate around self-attention
/// and the feed-forward, with an affine-normalized cross-attention to the condition between them.
final class NFKTrellisModulatedCrossBlock: Module {
    @ModuleInfo(key: "norm2") var crossNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: NFKTrellisSelfAttention
    @ModuleInfo(key: "cross_attn") var crossAttention: NFKTrellisCrossAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKTrellisFeedForward
    @ModuleInfo(key: "adaLN_modulation") var modulation: [Module]

    init(width: Int, contextWidth: Int, heads: Int, hidden: Int, crossHeadNorm: Bool) {
        _crossNorm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        _selfAttention.wrappedValue = NFKTrellisSelfAttention(width: width, heads: heads, headNorm: true)
        _crossAttention.wrappedValue = NFKTrellisCrossAttention(width: width, contextWidth: contextWidth, heads: heads,
                                                                headNorm: crossHeadNorm)
        _feedForward.wrappedValue = NFKTrellisFeedForward(width: width, hidden: hidden)
        _modulation.wrappedValue = [NFKTrellisSlot(), Linear(width, width * 6)]
    }

    func callAsFunction(_ x: MLXArray, timestep: MLXArray, context: MLXArray) -> MLXArray {
        let parts = split((modulation[1] as! Linear)(silu(timestep)), parts: 6, axis: 1).map { expandedDimensions($0, axis: 1) }
        var x = x
        var h = NFKTrellisAttention.normalized(x, eps: 1e-6) * (1 + parts[1]) + parts[0]
        x = x + selfAttention(h) * parts[2]
        x = x + crossAttention(crossNorm(x), context: context)
        h = NFKTrellisAttention.normalized(x, eps: 1e-6) * (1 + parts[4]) + parts[3]
        return x + feedForward(h) * parts[5]
    }
}

/// `t_embedder`: the sinusoidal embedding of a timestep (cosines then sines over `frequencyWidth / 2`
/// frequencies from 1 down to 1/10000) through linear, SiLU, linear.
final class NFKTrellisTimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var layers: [Module]
    let frequencyWidth: Int

    init(width: Int, frequencyWidth: Int) {
        self.frequencyWidth = frequencyWidth
        _layers.wrappedValue = [Linear(frequencyWidth, width), NFKTrellisSlot(), Linear(width, width)]
    }

    static func sinusoid(_ t: MLXArray, width: Int) -> MLXArray {
        let half = width / 2
        let frequencies = exp(MLXArray(-Float(log(10000.0))) * MLXArray(0 ..< half).asType(.float32) / Float(half))
        let angles = expandedDimensions(t.asType(.float32), axis: 1) * expandedDimensions(frequencies, axis: 0)
        return concatenated([cos(angles), sin(angles)], axis: -1)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let first = layers[0] as! Linear, last = layers[2] as! Linear
        return last(silu(first(Self.sinusoid(t, width: frequencyWidth))))
    }
}

enum NFKTrellisPositions {
    /// Absolute sinusoidal embeddings of integer positions `[M, axes]`: per axis, the sines then the cosines
    /// of the position times `width / axes / 2` frequencies from 1 down to 1/10000, the axes side by side,
    /// zero-padded to `width`.
    static func absolute(_ positions: MLXArray, width: Int) -> MLXArray {
        let count = positions.dim(0), axes = positions.dim(1)
        let frequencyCount = width / axes / 2
        let frequencies = 1 / pow(MLXArray(Float(10000)), MLXArray(0 ..< frequencyCount).asType(.float32) / Float(frequencyCount))
        let angles = expandedDimensions(positions.reshaped([-1]).asType(.float32), axis: 1) * expandedDimensions(frequencies, axis: 0)
        var embedding = concatenated([sin(angles), cos(angles)], axis: -1).reshaped([count, axes * 2 * frequencyCount])
        if embedding.dim(1) < width {
            embedding = concatenated([embedding, MLXArray.zeros([count, width - embedding.dim(1)])], axis: -1)
        }
        return embedding
    }

    /// The `[side³, 3]` integer positions of a dense cube in x-major order.
    static func cube(side: Int) -> MLXArray {
        let range = MLXArray(0 ..< Int32(side))
        let x = broadcast(range.reshaped([side, 1, 1]), to: [side, side, side])
        let y = broadcast(range.reshaped([1, side, 1]), to: [side, side, side])
        let z = broadcast(range.reshaped([1, 1, side]), to: [side, side, side])
        return stacked([x, y, z], axis: -1).reshaped([side * side * side, 3])
    }
}

// MARK: - Image encoder

final class NFKTrellisLayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    init(width: Int) { _gamma.wrappedValue = MLXArray.ones([width]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

final class NFKTrellisEncoderAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var projection: Linear
    let heads: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        _qkv.wrappedValue = Linear(width, width * 3)
        _projection.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let fused = qkv(x).reshaped([batch, length, 3, heads, width / heads])
        return projection(NFKTrellisAttention.attend(fused[0..., 0..., 0], fused[0..., 0..., 1], fused[0..., 0..., 2]))
    }
}

final class NFKTrellisEncoderMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(width: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(width, hidden)
        _fc2.wrappedValue = Linear(hidden, width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class NFKTrellisEncoderBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: NFKTrellisEncoderAttention
    @ModuleInfo(key: "ls1") var scale1: NFKTrellisLayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKTrellisEncoderMLP
    @ModuleInfo(key: "ls2") var scale2: NFKTrellisLayerScale

    init(_ configuration: NFKMLXTrellisConfiguration) {
        let width = configuration.encoderWidth
        _norm1.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        _attention.wrappedValue = NFKTrellisEncoderAttention(width: width, heads: configuration.encoderHeads)
        _scale1.wrappedValue = NFKTrellisLayerScale(width: width)
        _norm2.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        _mlp.wrappedValue = NFKTrellisEncoderMLP(width: width, hidden: configuration.encoderHidden)
        _scale2.wrappedValue = NFKTrellisLayerScale(width: width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = x + scale1(attention(norm1(x)))
        return x + scale2(mlp(norm2(x)))
    }
}

final class NFKTrellisPatchEmbedding: Module {
    @ModuleInfo(key: "proj") var projection: Conv2d
    init(_ configuration: NFKMLXTrellisConfiguration) {
        _projection.wrappedValue = Conv2d(inputChannels: 3, outputChannels: configuration.encoderWidth,
                                          kernelSize: .init(configuration.patchSize), stride: .init(configuration.patchSize))
    }
}

/// DINOv2 ViT-L/14 with four register tokens, read as TRELLIS reads it: the last block's tokens before the
/// final norm, then normalized without an affine transform. Build one with
/// `NFKMLXTrellis.imageEncoder(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisImageEncoder: Module {
    @ParameterInfo(key: "cls_token") var classToken: MLXArray
    @ParameterInfo(key: "register_tokens") var registers: MLXArray
    @ParameterInfo(key: "pos_embed") var positions: MLXArray
    @ModuleInfo(key: "patch_embed") var patchEmbedding: NFKTrellisPatchEmbedding
    @ModuleInfo(key: "blocks") var blocks: [NFKTrellisEncoderBlock]

    public let configuration: NFKMLXTrellisConfiguration
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let deviation: [Float] = [0.229, 0.224, 0.225]

    init(_ configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        let width = configuration.encoderWidth
        _classToken.wrappedValue = MLXArray.zeros([1, 1, width])
        _registers.wrappedValue = MLXArray.zeros([1, configuration.registerTokens, width])
        let grid = configuration.encoderPositionGrid
        _positions.wrappedValue = MLXArray.zeros([1, grid * grid + 1, width])
        _patchEmbedding.wrappedValue = NFKTrellisPatchEmbedding(configuration)
        _blocks.wrappedValue = (0 ..< configuration.encoderDepth).map { _ in NFKTrellisEncoderBlock(configuration) }
    }

    /// The condition tokens `[N, 1 + registers + grid², width]` (class, registers, patches) for images
    /// `[N, imageSize, imageSize, 3]` in 0...1.
    public func callAsFunction(_ images: MLXArray) -> MLXArray {
        let batch = images.dim(0)
        let width = configuration.encoderWidth
        let normalized = (images - MLXArray(Self.mean)) / MLXArray(Self.deviation)
        let patches = patchEmbedding.projection(normalized).reshaped([batch, -1, width])
        precondition(patches.dim(1) + 1 == positions.dim(1), "the encoder reads the grid its positions were trained at")
        var x = concatenated([broadcast(classToken, to: [batch, 1, width]), patches], axis: 1) + positions
        x = concatenated([x[0..., 0 ..< 1], broadcast(registers, to: [batch, registers.dim(1), width]), x[0..., 1...]], axis: 1)
        for block in blocks {
            x = block(x)
        }
        return NFKTrellisAttention.normalized(x, eps: 1e-5)
    }
}

// MARK: - Sparse-structure flow

/// The flow transformer over the dense structure latent: one token per voxel, absolute position
/// embeddings, and modulated cross blocks conditioned on the image tokens. Its output is the velocity.
/// Build one with `NFKMLXTrellis.structureFlow(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisStructureFlow: Module {
    @ModuleInfo(key: "t_embedder") var timestepEmbedder: NFKTrellisTimestepEmbedder
    @ParameterInfo(key: "pos_emb") var positions: MLXArray
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "blocks") var blocks: [NFKTrellisModulatedCrossBlock]
    @ModuleInfo(key: "out_layer") var outputLayer: Linear

    public let configuration: NFKMLXTrellisConfiguration

    init(_ configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        let width = configuration.flowWidth
        _timestepEmbedder.wrappedValue = NFKTrellisTimestepEmbedder(width: width, frequencyWidth: configuration.frequencyWidth)
        _positions.wrappedValue = NFKTrellisPositions.absolute(NFKTrellisPositions.cube(side: configuration.structureResolution), width: width)
        _inputLayer.wrappedValue = Linear(configuration.structureChannels, width)
        _blocks.wrappedValue = (0 ..< configuration.flowBlocks).map { _ in
            NFKTrellisModulatedCrossBlock(width: width, contextWidth: configuration.conditionWidth, heads: configuration.flowHeads,
                                          hidden: configuration.flowHidden, crossHeadNorm: configuration.crossAttentionHeadNorm)
        }
        _outputLayer.wrappedValue = Linear(width, configuration.structureChannels)
    }

    /// The velocity `[N, R, R, R, C]` at latent `x` `[N, R, R, R, C]`, timestep `t` `[N]` on the
    /// 0...1000 scale, and condition tokens `[N, L, conditionWidth]`.
    public func callAsFunction(_ x: MLXArray, t: MLXArray, condition: MLXArray) -> MLXArray {
        let batch = x.dim(0), side = x.dim(1)
        var h = inputLayer(x.reshaped([batch, side * side * side, x.dim(4)])) + positions
        let timestep = timestepEmbedder(t)
        for block in blocks {
            h = block(h, timestep: timestep, context: condition)
        }
        return outputLayer(NFKTrellisAttention.normalized(h, eps: 1e-5)).reshaped([batch, side, side, side, -1])
    }
}

// MARK: - Sparse-structure decoder

/// Channel LayerNorm, SiLU, 3³ convolution, twice, plus the input.
final class NFKTrellisResidualBlock3d: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "conv1") var conv1: Conv3d
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "conv2") var conv2: Conv3d

    init(channels: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: channels)
        _conv1.wrappedValue = Conv3d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = LayerNorm(dimensions: channels)
        _conv2.wrappedValue = Conv3d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(silu(norm2(conv1(silu(norm1(x))))))
    }
}

/// A 3³ convolution to eight times the output channels, then a 3D pixel shuffle that doubles each side.
final class NFKTrellisUpsample3d: Module {
    @ModuleInfo(key: "conv") var conv: Conv3d
    let outputChannels: Int

    init(inputChannels: Int, outputChannels: Int) {
        self.outputChannels = outputChannels
        _conv.wrappedValue = Conv3d(inputChannels: inputChannels, outputChannels: outputChannels * 8, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        Self.shuffle(conv(x), channels: outputChannels)
    }

    /// `[N, X, Y, Z, 8C]` to `[N, 2X, 2Y, 2Z, C]`: output voxel (2x + a, 2y + b, 2z + d) reads channel
    /// c · 8 + a · 4 + b · 2 + d.
    static func shuffle(_ h: MLXArray, channels: Int) -> MLXArray {
        let (batch, sx, sy, sz) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        return h.reshaped([batch, sx, sy, sz, channels, 2, 2, 2])
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped([batch, sx * 2, sy * 2, sz * 2, channels])
    }
}

/// The decoder from the dense structure latent to occupancy logits: an input convolution, residual
/// blocks, a pixel-shuffle upsample between widths, and a normalized, activated output convolution.
/// A voxel is occupied where its logit is positive. Build one with
/// `NFKMLXTrellis.structureDecoder(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisStructureDecoder: Module {
    @ModuleInfo(key: "input_layer") var inputLayer: Conv3d
    @ModuleInfo(key: "middle_block") var middle: [NFKTrellisResidualBlock3d]
    @ModuleInfo(key: "blocks") var blocks: [Module]
    @ModuleInfo(key: "out_layer") var outputLayers: [Module]

    public let configuration: NFKMLXTrellisConfiguration

    init(_ configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        let channels = configuration.decoderChannels
        _inputLayer.wrappedValue = Conv3d(inputChannels: configuration.structureChannels, outputChannels: channels[0],
                                          kernelSize: 3, padding: 1)
        _middle.wrappedValue = (0 ..< configuration.decoderMiddleBlocks).map { _ in NFKTrellisResidualBlock3d(channels: channels[0]) }
        var sequence = [Module]()
        for (index, width) in channels.enumerated() {
            sequence += (0 ..< configuration.decoderResidualBlocks).map { _ in NFKTrellisResidualBlock3d(channels: width) as Module }
            if index + 1 < channels.count {
                sequence.append(NFKTrellisUpsample3d(inputChannels: width, outputChannels: channels[index + 1]))
            }
        }
        _blocks.wrappedValue = sequence
        _outputLayers.wrappedValue = [LayerNorm(dimensions: channels[channels.count - 1]), NFKTrellisSlot(),
                                      Conv3d(inputChannels: channels[channels.count - 1], outputChannels: 1, kernelSize: 3, padding: 1)]
    }

    /// The occupancy logits `[N, 4R, 4R, 4R, 1]` for structure latents `[N, R, R, R, C]`.
    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var h = inputLayer(latent)
        for block in middle {
            h = block(h)
        }
        for block in blocks {
            if let residual = block as? NFKTrellisResidualBlock3d {
                h = residual(h)
            } else {
                h = (block as! NFKTrellisUpsample3d)(h)
            }
        }
        let norm = outputLayers[0] as! LayerNorm, conv = outputLayers[2] as! Conv3d
        return conv(silu(norm(h)))
    }

    /// The occupied voxels `[M, 3]` (x, y, z) of one logit volume `[R, R, R, 1]`, in x-major order.
    public static func occupiedVoxels(_ logits: MLXArray) -> MLXArray {
        let side = logits.dim(0)
        let flags = (logits.reshaped([-1]) .> 0).asArray(Bool.self)
        var coordinates = [Int32]()
        for (index, occupied) in flags.enumerated() where occupied {
            coordinates += [Int32(index / (side * side)), Int32(index / side % side), Int32(index % side)]
        }
        return MLXArray(coordinates).reshaped([-1, 3])
    }
}

// MARK: - Loading

/// TRELLIS image-to-3D networks and their loaders. Introduced in InferKit 0.4.0.
public enum NFKMLXTrellis {
    /// The image encoder with the DINOv2 checkpoint at `weightsURL` (`dinov2_vitl14_reg4_pretrain.pth` or a
    /// safetensors copy) loaded; nil builds random weights.
    public static func imageEncoder(weightsURL: URL?, configuration: NFKMLXTrellisConfiguration = .released) throws
        -> NFKMLXTrellisImageEncoder {
        let encoder = NFKMLXTrellisImageEncoder(configuration)
        if let weightsURL {
            try load(weightsURL, into: encoder) { key in key != "mask_token" && !key.hasPrefix("norm.") }
        }
        return encoder
    }

    /// The structure flow with `ss_flow_img_dit_L_16l8_fp16.safetensors` at `weightsURL` loaded, at float32;
    /// nil builds random weights.
    public static func structureFlow(weightsURL: URL?, configuration: NFKMLXTrellisConfiguration = .released) throws
        -> NFKMLXTrellisStructureFlow {
        let flow = NFKMLXTrellisStructureFlow(configuration)
        if let weightsURL {
            try load(weightsURL, into: flow) { _ in true }
        }
        return flow
    }

    /// The structure decoder with `ss_dec_conv3d_16l8_fp16.safetensors` at `weightsURL` loaded, at float32;
    /// nil builds random weights.
    public static func structureDecoder(weightsURL: URL?, configuration: NFKMLXTrellisConfiguration = .released) throws
        -> NFKMLXTrellisStructureDecoder {
        let decoder = NFKMLXTrellisStructureDecoder(configuration)
        if let weightsURL {
            try load(weightsURL, into: decoder) { _ in true }
        }
        return decoder
    }

    /// Reads a checkpoint, keeps the tensors `reading` accepts, moves dense convolution weights channels
    /// last (sparse convolution weights are stored `[out, k, k, k, in]` already), widens half-precision
    /// tensors to float32, and applies the result with full coverage.
    static func load(_ url: URL, into module: Module, transposingConvolutions: Bool = true,
                     reading: @escaping (String) -> Bool) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url, reading: reading)
        let transposing = transposingConvolutions && checkpoint.needsConvTranspose
        let arrays: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard reading(key) else { return nil }
            var value = value
            if transposing && (value.ndim == 4 || value.ndim == 5) && key.hasSuffix("weight") {
                value = value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value.transposed(0, 2, 3, 4, 1)
            }
            if value.dtype == .float16 || value.dtype == .bfloat16 {
                value = value.asType(.float32)
            }
            return (key, value)
        }
        try NFKMLXWeights.apply(arrays, to: module)
    }
}
