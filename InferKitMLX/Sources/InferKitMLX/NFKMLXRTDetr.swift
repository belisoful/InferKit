//
//  NFKMLXRTDetr.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXFast
import MLXNN

// RT-DETR (Real-Time DEtection TRansformer, PekingU/lyuwenyu), the license-clean (Apache-2.0)
// object detector: a ResNet-D backbone, a hybrid encoder (an AIFI transformer on the deepest feature
// plus a CSP-RepVGG FPN/PAN), query selection over generated anchors, and a deformable-attention
// decoder with iterative box refinement. Ported from transformers' own RTDetrForObjectDetection.
//
// The decoder's cross-attention is multi-scale deformable ATTENTION — a `grid_sample` bilinear gather
// at learned offset locations, which MLX expresses with `takeAlong` — NOT DCNv2 deformable
// CONVOLUTION (which has no MLX op). So unlike BiRefNet, RT-DETR is portable.

// MARK: - Configuration

public struct NFKMLXRTDetrConfiguration: Sendable {
    // Backbone (ResNet-D).
    public var embeddingSize: Int
    public var hiddenSizes: [Int]                 // four stage widths
    public var depths: [Int]                      // four stage depths
    // Hybrid encoder.
    public var encoderInChannels: [Int]           // the backbone out-feature widths
    public var featStrides: [Int]
    public var encoderHiddenDim: Int
    public var encoderFFNDim: Int
    public var numAttentionHeads: Int
    public var encoderLayers: Int
    public var encodeProjLayers: [Int]
    public var positionalEncodingTemperature: Float
    public var hiddenExpansion: Float
    // Decoder.
    public var dModel: Int
    public var decoderAttentionHeads: Int
    public var decoderFFNDim: Int
    public var decoderLayers: Int
    public var decoderNPoints: Int
    public var numFeatureLevels: Int
    public var numQueries: Int
    public var numLabels: Int
    public var batchNormEps: Float
    public var layerNormEps: Float
    /// The square input the image processor resizes to (RT-DETR squashes, no aspect-preserving pad).
    public var inputResolution: Int = 640
    /// The minimum per-query class probability a detection is emitted at.
    public var confidenceThreshold: Float = 0.3

    public init(embeddingSize: Int, hiddenSizes: [Int], depths: [Int], encoderInChannels: [Int],
                featStrides: [Int], encoderHiddenDim: Int, encoderFFNDim: Int, numAttentionHeads: Int,
                encoderLayers: Int, encodeProjLayers: [Int], positionalEncodingTemperature: Float,
                hiddenExpansion: Float, dModel: Int, decoderAttentionHeads: Int, decoderFFNDim: Int,
                decoderLayers: Int, decoderNPoints: Int, numFeatureLevels: Int, numQueries: Int,
                numLabels: Int, batchNormEps: Float, layerNormEps: Float) {
        self.embeddingSize = embeddingSize
        self.hiddenSizes = hiddenSizes
        self.depths = depths
        self.encoderInChannels = encoderInChannels
        self.featStrides = featStrides
        self.encoderHiddenDim = encoderHiddenDim
        self.encoderFFNDim = encoderFFNDim
        self.numAttentionHeads = numAttentionHeads
        self.encoderLayers = encoderLayers
        self.encodeProjLayers = encodeProjLayers
        self.positionalEncodingTemperature = positionalEncodingTemperature
        self.hiddenExpansion = hiddenExpansion
        self.dModel = dModel
        self.decoderAttentionHeads = decoderAttentionHeads
        self.decoderFFNDim = decoderFFNDim
        self.decoderLayers = decoderLayers
        self.decoderNPoints = decoderNPoints
        self.numFeatureLevels = numFeatureLevels
        self.numQueries = numQueries
        self.numLabels = numLabels
        self.batchNormEps = batchNormEps
        self.layerNormEps = layerNormEps
    }

    /// The parity configuration: a shrunk RT-DETR that carries every structural form.
    public static let tiny = NFKMLXRTDetrConfiguration(
        embeddingSize: 16, hiddenSizes: [16, 32, 64, 128], depths: [1, 1, 1, 1],
        encoderInChannels: [32, 64, 128], featStrides: [8, 16, 32], encoderHiddenDim: 32,
        encoderFFNDim: 48, numAttentionHeads: 2, encoderLayers: 1, encodeProjLayers: [2],
        positionalEncodingTemperature: 10000, hiddenExpansion: 1.0, dModel: 32,
        decoderAttentionHeads: 2, decoderFFNDim: 48, decoderLayers: 2, decoderNPoints: 4,
        numFeatureLevels: 3, numQueries: 10, numLabels: 4, batchNormEps: 1e-5, layerNormEps: 1e-5)

    /// The released `PekingU/rtdetr_r50vd` geometry (ResNet-50-vd backbone, 80 COCO classes).
    public static let r50vd = NFKMLXRTDetrConfiguration(
        embeddingSize: 64, hiddenSizes: [256, 512, 1024, 2048], depths: [3, 4, 6, 3],
        encoderInChannels: [512, 1024, 2048], featStrides: [8, 16, 32], encoderHiddenDim: 256,
        encoderFFNDim: 1024, numAttentionHeads: 8, encoderLayers: 1, encodeProjLayers: [2],
        positionalEncodingTemperature: 10000, hiddenExpansion: 1.0, dModel: 256,
        decoderAttentionHeads: 8, decoderFFNDim: 1024, decoderLayers: 6, decoderNPoints: 4,
        numFeatureLevels: 3, numQueries: 300, numLabels: 80, batchNormEps: 1e-5, layerNormEps: 1e-5)
}

// MARK: - Backbone (ResNet-D)

/// A backbone convolution: `convolution` (bias-free) + `normalization` (BatchNorm) + optional ReLU.
final class NFKRTDetrResNetConvNorm: Module {
    @ModuleInfo(key: "convolution") var conv: Conv2d
    @ModuleInfo(key: "normalization") var norm: BatchNorm
    let activate: Bool

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, stride: Int, activate: Bool, eps: Float) {
        self.activate = activate
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair(kernel / 2), bias: false)
        _norm.wrappedValue = BatchNorm(featureCount: outChannels, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = norm(conv(x))
        return activate ? relu(y) : y
    }
}

/// The ResNet-D shortcut projection: a 1×1 convolution + BatchNorm.
final class NFKRTDetrShortCut: Module {
    @ModuleInfo(key: "convolution") var conv: Conv2d
    @ModuleInfo(key: "normalization") var norm: BatchNorm

    init(_ inChannels: Int, _ outChannels: Int, stride: Int, eps: Float) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: 1, stride: IntOrPair(stride), bias: false)
        _norm.wrappedValue = BatchNorm(featureCount: outChannels, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(conv(x)) }
}

/// A ResNet-D bottleneck: three convolutions (1×1 reduce, 3×3, 1×1 expand) with an avgpool-in-shortcut
/// downsample at stride 2, added back and activated.
final class NFKRTDetrBottleneck: Module {
    @ModuleInfo(key: "layer") var layer: [NFKRTDetrResNetConvNorm]     // three convs
    @ModuleInfo(key: "shortcut") var shortcut: [Module]               // [] identity, [ShortCut], or [marker, ShortCut]
    let stride: Int

    init(_ inChannels: Int, _ outChannels: Int, stride: Int, eps: Float) {
        self.stride = stride
        let reduces = outChannels / 4
        // downsample_in_bottleneck is false: the stride lands on the 3×3, not the first 1×1.
        _layer.wrappedValue = [
            NFKRTDetrResNetConvNorm(inChannels, reduces, kernel: 1, stride: 1, activate: true, eps: eps),
            NFKRTDetrResNetConvNorm(reduces, reduces, kernel: 3, stride: stride, activate: true, eps: eps),
            NFKRTDetrResNetConvNorm(reduces, outChannels, kernel: 1, stride: 1, activate: false, eps: eps),
        ]
        let applyShortcut = inChannels != outChannels || stride != 1
        if stride == 2 {
            // Sequential[AvgPool2d, ShortCut] — the marker occupies index 0 so keys read `shortcut.1.*`.
            _shortcut.wrappedValue = applyShortcut
                ? [Module(), NFKRTDetrShortCut(inChannels, outChannels, stride: 1, eps: eps)]
                : []
        } else {
            _shortcut.wrappedValue = applyShortcut
                ? [NFKRTDetrShortCut(inChannels, outChannels, stride: stride, eps: eps)]
                : []
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var hidden = x
        for conv in layer { hidden = conv(hidden) }
        if stride == 2, shortcut.count == 2 {
            let pooled = NFKMLXResample.averagePooled(residual, kernel: 2, stride: 2)
            residual = (shortcut[1] as! NFKRTDetrShortCut)(pooled)
        } else if let sc = shortcut.first as? NFKRTDetrShortCut {
            residual = sc(residual)
        }
        return relu(hidden + residual)
    }
}

/// A ResNet-D stage: a first (possibly downsampling) bottleneck followed by `depth - 1` plain ones.
final class NFKRTDetrResNetStage: Module {
    @ModuleInfo(key: "layers") var layers: [NFKRTDetrBottleneck]

    init(_ inChannels: Int, _ outChannels: Int, stride: Int, depth: Int, eps: Float) {
        var built = [NFKRTDetrBottleneck(inChannels, outChannels, stride: stride, eps: eps)]
        for _ in 1 ..< depth {
            built.append(NFKRTDetrBottleneck(outChannels, outChannels, stride: 1, eps: eps))
        }
        _layers.wrappedValue = built
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for layer in layers { hidden = layer(hidden) }
        return hidden
    }
}

/// The ResNet-D stem: a deep 3-conv embedder and a max pool.
final class NFKRTDetrEmbeddings: Module {
    @ModuleInfo(key: "embedder") var embedder: [NFKRTDetrResNetConvNorm]

    init(_ config: NFKMLXRTDetrConfiguration) {
        let half = config.embeddingSize / 2
        _embedder.wrappedValue = [
            NFKRTDetrResNetConvNorm(3, half, kernel: 3, stride: 2, activate: true, eps: config.batchNormEps),
            NFKRTDetrResNetConvNorm(half, half, kernel: 3, stride: 1, activate: true, eps: config.batchNormEps),
            NFKRTDetrResNetConvNorm(half, config.embeddingSize, kernel: 3, stride: 1, activate: true, eps: config.batchNormEps),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for conv in embedder { hidden = conv(hidden) }
        return NFKMLXResample.maxPooled(hidden, kernel: 3, stride: 2, padding: 1)
    }
}

final class NFKRTDetrResNetEncoder: Module {
    @ModuleInfo(key: "stages") var stages: [NFKRTDetrResNetStage]
    let outStages: [Int]

    init(_ config: NFKMLXRTDetrConfiguration) {
        // out_features = stage2/stage3/stage4 -> the last three of the four stages.
        outStages = [1, 2, 3]
        var built = [NFKRTDetrResNetStage(config.embeddingSize, config.hiddenSizes[0], stride: 1,
                                          depth: config.depths[0], eps: config.batchNormEps)]
        for i in 1 ..< 4 {
            built.append(NFKRTDetrResNetStage(config.hiddenSizes[i - 1], config.hiddenSizes[i], stride: 2,
                                              depth: config.depths[i], eps: config.batchNormEps))
        }
        _stages.wrappedValue = built
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var hidden = x
        var outputs = [MLXArray]()
        for (index, stage) in stages.enumerated() {
            hidden = stage(hidden)
            if outStages.contains(index) { outputs.append(hidden) }
        }
        return outputs
    }
}

final class NFKRTDetrResNetBackbone: Module {
    @ModuleInfo(key: "embedder") var embedder: NFKRTDetrEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKRTDetrResNetEncoder

    init(_ config: NFKMLXRTDetrConfiguration) {
        _embedder.wrappedValue = NFKRTDetrEmbeddings(config)
        _encoder.wrappedValue = NFKRTDetrResNetEncoder(config)
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] { encoder(embedder(x)) }
}

final class NFKRTDetrConvEncoder: Module {
    @ModuleInfo(key: "model") var model: NFKRTDetrResNetBackbone

    init(_ config: NFKMLXRTDetrConfiguration) {
        _model.wrappedValue = NFKRTDetrResNetBackbone(config)
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] { model(x) }
}

// MARK: - Hybrid encoder

/// A hybrid-encoder convolution: `conv` (bias-free) + `norm` (BatchNorm) + SiLU.
final class NFKRTDetrConvNorm: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    let activate: Bool

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, stride: Int, activate: Bool, eps: Float) {
        self.activate = activate
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair((kernel - 1) / 2), bias: false)
        _norm.wrappedValue = BatchNorm(featureCount: outChannels, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = norm(conv(x))
        return activate ? silu(y) : y
    }
}

/// A RepVGG block: a 3×3 and a 1×1 convolution summed, then SiLU.
final class NFKRTDetrRepVgg: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKRTDetrConvNorm
    @ModuleInfo(key: "conv2") var conv2: NFKRTDetrConvNorm

    init(_ channels: Int, eps: Float) {
        _conv1.wrappedValue = NFKRTDetrConvNorm(channels, channels, kernel: 3, stride: 1, activate: false, eps: eps)
        _conv2.wrappedValue = NFKRTDetrConvNorm(channels, channels, kernel: 1, stride: 1, activate: false, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { silu(conv1(x) + conv2(x)) }
}

/// A CSP layer with RepVGG bottlenecks. `conv3` is the identity when `hiddenChannels == outChannels`.
final class NFKRTDetrCSPRep: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKRTDetrConvNorm
    @ModuleInfo(key: "conv2") var conv2: NFKRTDetrConvNorm
    @ModuleInfo(key: "bottlenecks") var bottlenecks: [NFKRTDetrRepVgg]
    @ModuleInfo(key: "conv3") var conv3: NFKRTDetrConvNorm?

    init(_ config: NFKMLXRTDetrConfiguration) {
        let outChannels = config.encoderHiddenDim
        let inChannels = outChannels * 2
        let hidden = Int(Float(outChannels) * config.hiddenExpansion)
        _conv1.wrappedValue = NFKRTDetrConvNorm(inChannels, hidden, kernel: 1, stride: 1, activate: true, eps: config.batchNormEps)
        _conv2.wrappedValue = NFKRTDetrConvNorm(inChannels, hidden, kernel: 1, stride: 1, activate: true, eps: config.batchNormEps)
        _bottlenecks.wrappedValue = (0 ..< 3).map { _ in NFKRTDetrRepVgg(hidden, eps: config.batchNormEps) }
        _conv3.wrappedValue = hidden == outChannels
            ? nil
            : NFKRTDetrConvNorm(hidden, outChannels, kernel: 1, stride: 1, activate: true, eps: config.batchNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var branch1 = conv1(x)
        for block in bottlenecks { branch1 = block(branch1) }
        let branch2 = conv2(x)
        let merged = branch1 + branch2
        return conv3?(merged) ?? merged
    }
}

/// The multi-head attention shared by the AIFI encoder and the decoder self-attention. Position
/// embeddings are added to the queries and keys, not the values.
final class NFKRTDetrMultiheadAttention: Module {
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int
    let headDim: Int

    init(_ embedDim: Int, heads: Int) {
        self.heads = heads
        self.headDim = embedDim / heads
        _kProj.wrappedValue = Linear(embedDim, embedDim)
        _vProj.wrappedValue = Linear(embedDim, embedDim)
        _qProj.wrappedValue = Linear(embedDim, embedDim)
        _outProj.wrappedValue = Linear(embedDim, embedDim)
    }

    private func split(_ t: MLXArray) -> MLXArray {
        let (b, n) = (t.dim(0), t.dim(1))
        return t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ hidden: MLXArray, position: MLXArray?) -> MLXArray {
        let (b, n) = (hidden.dim(0), hidden.dim(1))
        let withPos = position == nil ? hidden : hidden + position!
        let q = split(qProj(withPos))
        let k = split(kProj(withPos))
        let v = split(vProj(hidden))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: .none)
        let merged = attn.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim])
        return outProj(merged)
    }
}

/// The AIFI encoder layer: post-norm self-attention + feed-forward (GELU).
final class NFKRTDetrEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKRTDetrMultiheadAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(_ config: NFKMLXRTDetrConfiguration) {
        _selfAttn.wrappedValue = NFKRTDetrMultiheadAttention(config.encoderHiddenDim, heads: config.numAttentionHeads)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: config.encoderHiddenDim, eps: config.layerNormEps)
        _fc1.wrappedValue = Linear(config.encoderHiddenDim, config.encoderFFNDim)
        _fc2.wrappedValue = Linear(config.encoderFFNDim, config.encoderHiddenDim)
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.encoderHiddenDim, eps: config.layerNormEps)
    }

    func callAsFunction(_ hidden: MLXArray, position: MLXArray) -> MLXArray {
        var x = hidden + selfAttn(hidden, position: position)
        x = selfAttnNorm(x)
        let residual = x
        x = fc2(gelu(fc1(x)))
        return finalNorm(residual + x)
    }
}

/// One `RTDetrEncoder` — a stack of AIFI layers over a single feature level.
final class NFKRTDetrEncoderStack: Module {
    @ModuleInfo(key: "layers") var layers: [NFKRTDetrEncoderLayer]

    init(_ config: NFKMLXRTDetrConfiguration) {
        _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in NFKRTDetrEncoderLayer(config) }
    }

    func callAsFunction(_ hidden: MLXArray, position: MLXArray) -> MLXArray {
        var x = hidden
        for layer in layers { x = layer(x, position: position) }
        return x
    }
}

final class NFKRTDetrHybridEncoder: Module {
    @ModuleInfo(key: "encoder") var encoder: [NFKRTDetrEncoderStack]
    @ModuleInfo(key: "lateral_convs") var lateralConvs: [NFKRTDetrConvNorm]
    @ModuleInfo(key: "fpn_blocks") var fpnBlocks: [NFKRTDetrCSPRep]
    @ModuleInfo(key: "downsample_convs") var downsampleConvs: [NFKRTDetrConvNorm]
    @ModuleInfo(key: "pan_blocks") var panBlocks: [NFKRTDetrCSPRep]
    let config: NFKMLXRTDetrConfiguration

    init(_ config: NFKMLXRTDetrConfiguration) {
        self.config = config
        let dim = config.encoderHiddenDim
        let stages = config.encoderInChannels.count - 1
        _encoder.wrappedValue = config.encodeProjLayers.map { _ in NFKRTDetrEncoderStack(config) }
        _lateralConvs.wrappedValue = (0 ..< stages).map { _ in
            NFKRTDetrConvNorm(dim, dim, kernel: 1, stride: 1, activate: true, eps: config.batchNormEps)
        }
        _fpnBlocks.wrappedValue = (0 ..< stages).map { _ in NFKRTDetrCSPRep(config) }
        _downsampleConvs.wrappedValue = (0 ..< stages).map { _ in
            NFKRTDetrConvNorm(dim, dim, kernel: 3, stride: 2, activate: true, eps: config.batchNormEps)
        }
        _panBlocks.wrappedValue = (0 ..< stages).map { _ in NFKRTDetrCSPRep(config) }
    }

    /// The 2-D sin-cos position embedding, built in the reference's flatten order (width outer,
    /// height inner). For a square feature this matches the row-major feature order; the reference
    /// itself adds it to the height-major tokens, so this reproduces that exactly.
    private func positionEmbedding(height: Int, width: Int, dtype: DType) -> MLXArray {
        let dim = config.encoderHiddenDim
        let posDim = dim / 4
        var omega = [Float](repeating: 0, count: posDim)
        for i in 0 ..< posDim {
            omega[i] = 1.0 / pow(config.positionalEncodingTemperature, Float(i) / Float(posDim))
        }
        var gw = [Float](), gh = [Float]()
        for w in 0 ..< width {
            for h in 0 ..< height {
                gw.append(Float(w))
                gh.append(Float(h))
            }
        }
        let omegaArray = MLXArray(omega).reshaped([1, posDim])
        let gwArray = MLXArray(gw).reshaped([width * height, 1])
        let ghArray = MLXArray(gh).reshaped([width * height, 1])
        let outW = gwArray * omegaArray
        let outH = ghArray * omegaArray
        let pos = concatenated([sin(outW), cos(outW), sin(outH), cos(outH)], axis: 1)
        return pos.reshaped([1, width * height, dim]).asType(dtype)
    }

    func callAsFunction(_ features: [MLXArray]) -> [MLXArray] {
        var hidden = features
        // AIFI on each named projection level.
        for (encoderIndex, level) in config.encodeProjLayers.enumerated() {
            let feature = hidden[level]
            let (h, w, c) = (feature.dim(1), feature.dim(2), feature.dim(3))
            let flat = feature.reshaped([1, h * w, c])
            let pos = positionEmbedding(height: h, width: w, dtype: feature.dtype)
            let encoded = encoder[encoderIndex](flat, position: pos)
            hidden[level] = encoded.reshaped([1, h, w, c])
        }
        // Top-down FPN.
        let stages = config.encoderInChannels.count - 1
        var fpn = [hidden[hidden.count - 1]]
        for idx in 0 ..< stages {
            let backboneFeature = hidden[stages - idx - 1]
            let top = lateralConvs[idx](fpn[fpn.count - 1])
            fpn[fpn.count - 1] = top
            let upsampled = NFKMLXResample.upsampleNearest(top, scale: 2)
            let fused = concatenated([upsampled, backboneFeature], axis: 3)
            fpn.append(fpnBlocks[idx](fused))
        }
        fpn.reverse()
        // Bottom-up PAN.
        var pan = [fpn[0]]
        for idx in 0 ..< stages {
            let top = pan[pan.count - 1]
            let fpnMap = fpn[idx + 1]
            let downsampled = downsampleConvs[idx](top)
            let fused = concatenated([downsampled, fpnMap], axis: 3)
            pan.append(panBlocks[idx](fused))
        }
        return pan
    }
}

// MARK: - Deformable attention

/// Multi-scale deformable attention: bilinear sampling of the flattened encoder features at learned
/// per-head, per-level, per-point offset locations.
final class NFKRTDetrDeformableAttention: Module {
    @ModuleInfo(key: "sampling_offsets") var samplingOffsets: Linear
    @ModuleInfo(key: "attention_weights") var attentionWeights: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    let heads: Int
    let levels: Int
    let points: Int
    let dModel: Int

    init(_ config: NFKMLXRTDetrConfiguration) {
        heads = config.decoderAttentionHeads
        levels = config.numFeatureLevels
        points = config.decoderNPoints
        dModel = config.dModel
        _samplingOffsets.wrappedValue = Linear(dModel, heads * levels * points * 2)
        _attentionWeights.wrappedValue = Linear(dModel, heads * levels * points)
        _valueProj.wrappedValue = Linear(dModel, dModel)
        _outputProj.wrappedValue = Linear(dModel, dModel)
    }

    /// Bilinear gather of `value` `[heads, H, W, hd]` at normalized `coords` `[heads, n, 2]` in `0...1`,
    /// with `align_corners=false` and zero padding (the reference's `grid_sample`).
    private func sample(_ value: MLXArray, coords: MLXArray, height: Int, width: Int) -> MLXArray {
        let (headCount, n, hd) = (value.dim(0), coords.dim(1), value.dim(3))
        let flat = value.reshaped([headCount, height * width, hd])
        let px: MLXArray = coords[0..., 0..., 0] * Float(width) - Float(0.5)   // [heads, n]
        let py: MLXArray = coords[0..., 0..., 1] * Float(height) - Float(0.5)
        let x0 = floor(px), y0 = floor(py)
        let x1 = x0 + Float(1), y1 = y0 + Float(1)
        let wx = px - x0, wy = py - y0
        let corners: [(MLXArray, MLXArray, MLXArray)] = [
            (x0, y0, (Float(1) - wx) * (Float(1) - wy)),
            (x1, y0, wx * (Float(1) - wy)),
            (x0, y1, (Float(1) - wx) * wy),
            (x1, y1, wx * wy),
        ]
        var output = MLXArray.zeros([headCount, n, hd])
        let maxX = Float(width - 1)
        let maxY = Float(height - 1)
        for (xc, yc, weight) in corners {
            let insideX = logicalAnd(xc .>= Float(0), xc .<= maxX)
            let insideY = logicalAnd(yc .>= Float(0), yc .<= maxY)
            let valid = logicalAnd(insideX, insideY).asType(value.dtype) // [heads, n]
            let xClamped = clip(xc, min: Float(0), max: maxX).asType(.int32)
            let yClamped = clip(yc, min: Float(0), max: maxY).asType(.int32)
            let flatIndex = yClamped * Int32(width) + xClamped           // [heads, n]
            let indices = broadcast(flatIndex.reshaped([headCount, n, 1]), to: [headCount, n, hd])
            let gathered = takeAlong(flat, indices, axis: 1)             // [heads, n, hd]
            let contribution = weight * valid                           // [heads, n]
            output = output + gathered * contribution.reshaped([headCount, n, 1])
        }
        return output
    }

    /// `hidden` `[1, Q, dModel]` (query already carrying its position), `value` the flattened encoder
    /// features `[1, S, dModel]`, `referencePoints` `[1, Q, levels, 4]`, `shapes` per-level `(h, w)`.
    func callAsFunction(_ hidden: MLXArray, value valueInput: MLXArray, referencePoints: MLXArray,
                        shapes: [(Int, Int)]) -> MLXArray {
        let q = hidden.dim(1)
        let hd = dModel / heads
        let value = valueProj(valueInput).reshaped([valueInput.dim(1), heads, hd])   // [S, heads, hd]

        let offsets = samplingOffsets(hidden).reshaped([q, heads, levels, points, 2])
        var weights = attentionWeights(hidden).reshaped([q, heads, levels * points])
        weights = softmax(weights, axis: -1).reshaped([q, heads, levels, points])

        // 4-coordinate reference points: location = ref_xy + offset / n_points * ref_wh * 0.5.
        let ref = referencePoints.reshaped([q, 1, levels, 1, 4])                     // [Q, 1, levels, 1, 4]
        let refXY = ref[0..., 0..., 0..., 0..., 0 ..< 2]
        let refWH = ref[0..., 0..., 0..., 0..., 2 ..< 4]
        let scaledOffsets = offsets / Float(points)
        let offsetLocations = scaledOffsets * refWH * Float(0.5)
        let locations = refXY + offsetLocations                                     // [Q, heads, levels, points, 2]

        var levelStart = 0
        var perLevel = [MLXArray]()                                                  // each [heads, Q, points, hd]
        for (levelIndex, (h, w)) in shapes.enumerated() {
            let count = h * w
            let levelValue = value[levelStart ..< (levelStart + count)]              // [count, heads, hd]
            levelStart += count
            let reshaped = levelValue.transposed(1, 0, 2).reshaped([heads, h, w, hd])
            let levelLoc = locations[0..., 0..., levelIndex, 0..., 0...]             // [Q, heads, points, 2]
            let coords = levelLoc.transposed(1, 0, 2, 3).reshaped([heads, q * points, 2])
            let sampled = sample(reshaped, coords: coords, height: h, width: w)      // [heads, Q*points, hd]
            perLevel.append(sampled.reshaped([heads, q, points, hd]))
        }
        // Combine levels and points, weighted.
        let stacked = stacked(perLevel, axis: 2).reshaped([heads, q, levels * points, hd])
        let weightHead = weights.transposed(1, 0, 2, 3).reshaped([heads, q, levels * points, 1])
        let combined = (stacked * weightHead).sum(axis: 2)                           // [heads, Q, hd]
        let output = combined.transposed(1, 0, 2).reshaped([1, q, heads * hd])
        return outputProj(output)
    }
}

// MARK: - Decoder

/// A simple MLP (relu between hidden layers), used by the query-position head and the box heads.
final class NFKRTDetrMLP: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(inputDim: Int, hiddenDim: Int, outputDim: Int, numLayers: Int) {
        var dims = [inputDim]
        dims.append(contentsOf: Array(repeating: hiddenDim, count: numLayers - 1))
        dims.append(outputDim)
        _layers.wrappedValue = (0 ..< numLayers).map { Linear(dims[$0], dims[$0 + 1]) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for (index, layer) in layers.enumerated() {
            hidden = index < layers.count - 1 ? relu(layer(hidden)) : layer(hidden)
        }
        return hidden
    }
}

final class NFKRTDetrDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKRTDetrMultiheadAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: NFKRTDetrDeformableAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(_ config: NFKMLXRTDetrConfiguration) {
        _selfAttn.wrappedValue = NFKRTDetrMultiheadAttention(config.dModel, heads: config.decoderAttentionHeads)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _encoderAttn.wrappedValue = NFKRTDetrDeformableAttention(config)
        _encoderAttnNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _fc1.wrappedValue = Linear(config.dModel, config.decoderFFNDim)
        _fc2.wrappedValue = Linear(config.decoderFFNDim, config.dModel)
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
    }

    func callAsFunction(_ hidden: MLXArray, position: MLXArray, value: MLXArray,
                        referencePoints: MLXArray, shapes: [(Int, Int)]) -> MLXArray {
        var x = hidden + selfAttn(hidden, position: position)
        x = selfAttnNorm(x)
        let secondResidual = x
        let cross = encoderAttn(x + position, value: value, referencePoints: referencePoints, shapes: shapes)
        x = encoderAttnNorm(secondResidual + cross)
        let residual = x
        x = fc2(relu(fc1(x)))
        return finalNorm(residual + x)
    }
}

final class NFKRTDetrDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKRTDetrDecoderLayer]
    @ModuleInfo(key: "query_pos_head") var queryPosHead: NFKRTDetrMLP
    @ModuleInfo(key: "class_embed") var classEmbed: [Linear]
    @ModuleInfo(key: "bbox_embed") var bboxEmbed: [NFKRTDetrMLP]

    init(_ config: NFKMLXRTDetrConfiguration) {
        _layers.wrappedValue = (0 ..< config.decoderLayers).map { _ in NFKRTDetrDecoderLayer(config) }
        _queryPosHead.wrappedValue = NFKRTDetrMLP(inputDim: 4, hiddenDim: 2 * config.dModel,
                                                  outputDim: config.dModel, numLayers: 2)
        _classEmbed.wrappedValue = (0 ..< config.decoderLayers).map { _ in Linear(config.dModel, config.numLabels) }
        _bboxEmbed.wrappedValue = (0 ..< config.decoderLayers).map { _ in
            NFKRTDetrMLP(inputDim: config.dModel, hiddenDim: config.dModel, outputDim: 4, numLayers: 3)
        }
    }

    /// Returns the per-layer logits and reference points (each `[layers, Q, *]`).
    func callAsFunction(_ target: MLXArray, value: MLXArray, referencePointsUnact: MLXArray,
                        shapes: [(Int, Int)]) -> (logits: [MLXArray], references: [MLXArray]) {
        var hidden = target
        var referencePoints = sigmoid(referencePointsUnact)               // [1, Q, 4]
        var logitsPerLayer = [MLXArray]()
        var referencesPerLayer = [MLXArray]()
        for (index, layer) in layers.enumerated() {
            let referenceInput = referencePoints.expandedDimensions(axis: 2)   // [1, Q, 1, 4]
            let referenceLevels = broadcast(referenceInput, to: [1, referencePoints.dim(1), shapes.count, 4])
            let position = queryPosHead(referencePoints)
            hidden = layer(hidden, position: position, value: value,
                           referencePoints: referenceLevels, shapes: shapes)
            let predictedCorners = bboxEmbed[index](hidden)
            let newReference = sigmoid(predictedCorners + NFKRTDetrDecoder.inverseSigmoid(referencePoints))
            referencePoints = newReference
            referencesPerLayer.append(newReference)
            logitsPerLayer.append(classEmbed[index](hidden))
        }
        return (logitsPerLayer, referencesPerLayer)
    }

    static func inverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let clamped = clip(x, min: 0, max: 1)
        let x1 = clip(clamped, min: eps, max: Float.greatestFiniteMagnitude)
        let x2 = clip(1 - clamped, min: eps, max: Float.greatestFiniteMagnitude)
        return log(x1 / x2)
    }
}

// MARK: - Full model

public final class NFKMLXRTDetrNet: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKRTDetrConvEncoder
    @ModuleInfo(key: "encoder_input_proj") var encoderInputProj: [[Module]]     // each [Conv2d, BatchNorm]
    @ModuleInfo(key: "encoder") var encoder: NFKRTDetrHybridEncoder
    @ModuleInfo(key: "enc_output") var encOutput: [Module]                      // [Linear, LayerNorm]
    @ModuleInfo(key: "enc_score_head") var encScoreHead: Linear
    @ModuleInfo(key: "enc_bbox_head") var encBboxHead: NFKRTDetrMLP
    @ModuleInfo(key: "decoder_input_proj") var decoderInputProj: [[Module]]
    @ModuleInfo(key: "decoder") var decoder: NFKRTDetrDecoder
    public let config: NFKMLXRTDetrConfiguration

    public init(_ config: NFKMLXRTDetrConfiguration) {
        self.config = config
        _backbone.wrappedValue = NFKRTDetrConvEncoder(config)
        _encoderInputProj.wrappedValue = config.encoderInChannels.map { channels in
            [Conv2d(inputChannels: channels, outputChannels: config.encoderHiddenDim, kernelSize: 1, bias: false),
             BatchNorm(featureCount: config.encoderHiddenDim, eps: 1e-5)]
        }
        _encoder.wrappedValue = NFKRTDetrHybridEncoder(config)
        _encOutput.wrappedValue = [Linear(config.dModel, config.dModel),
                                   LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)]
        _encScoreHead.wrappedValue = Linear(config.dModel, config.numLabels)
        _encBboxHead.wrappedValue = NFKRTDetrMLP(inputDim: config.dModel, hiddenDim: config.dModel,
                                                 outputDim: 4, numLayers: 3)
        _decoderInputProj.wrappedValue = (0 ..< config.numFeatureLevels).map { _ in
            [Conv2d(inputChannels: config.encoderHiddenDim, outputChannels: config.dModel, kernelSize: 1, bias: false),
             BatchNorm(featureCount: config.dModel, eps: config.batchNormEps)]
        }
        _decoder.wrappedValue = NFKRTDetrDecoder(config)
    }

    private func project(_ proj: [Module], _ x: MLXArray) -> MLXArray {
        (proj[1] as! BatchNorm)((proj[0] as! Conv2d)(x))
    }

    /// Generates the flattened anchors and the validity mask for the feature-level shapes.
    private func anchors(shapes: [(Int, Int)]) -> (anchors: MLXArray, validMask: MLXArray) {
        let gridSize: Float = 0.05
        var all = [MLXArray]()
        for (level, (h, w)) in shapes.enumerated() {
            var xy = [Float]()
            for i in 0 ..< h {
                for j in 0 ..< w {
                    xy.append((Float(j) + 0.5) / Float(w))
                    xy.append((Float(i) + 0.5) / Float(h))
                }
            }
            let grid = MLXArray(xy).reshaped([h * w, 2])
            let wh = MLXArray.ones([h * w, 2]) * (gridSize * pow(2.0, Float(level)))
            all.append(concatenated([grid, wh], axis: 1))
        }
        var anchors = concatenated(all, axis: 0)                          // [S, 4]
        let eps: Float = 1e-2
        let valid = logicalAnd(anchors .> eps, anchors .< (1 - eps)).all(axis: -1, keepDims: true)   // [S, 1]
        anchors = log(anchors / (1 - anchors))
        anchors = MLX.where(valid, anchors, MLXArray(Float.greatestFiniteMagnitude))
        return (anchors, valid.asType(anchors.dtype))
    }

    /// The staged outputs of one forward, for parity localization and the public detection.
    public struct Detection {
        public let backboneFeatures: [MLXArray]       // per-level NHWC
        public let encLastPAN: MLXArray               // deepest PAN map NHWC
        public let encClass: MLXArray                 // [1, S, num_labels]
        public let encCoord: MLXArray                 // [1, S, 4]
        public let initReference: MLXArray            // [1, Q, 4]
        public let topkIndices: MLXArray              // [Q]
        public let outputMemory: MLXArray             // [1, S, dModel], the query-selection features
        public let sourceFlatten: MLXArray            // [1, S, dModel], the deformable cross-attn value
        public let shapes: [(Int, Int)]               // per-level feature-map sizes
        public let logitsPerLayer: [MLXArray]         // each [1, Q, num_labels]
        public let referencesPerLayer: [MLXArray]     // each [1, Q, 4]
        public var logits: MLXArray { logitsPerLayer[logitsPerLayer.count - 1][0] }   // [Q, num_labels]
        public var boxes: MLXArray { referencesPerLayer[referencesPerLayer.count - 1][0] }  // [Q, 4]
    }

    /// Runs the decoder over an explicit query selection. `indices` `[Q]` picks the query tokens from
    /// `outputMemory` and `encCoord`; the deformable cross-attention reads `sourceFlatten`. This is the
    /// seam that verifies the decoder in isolation of the (float-tie-sensitive) top-k selection.
    public func decode(indices: MLXArray, detection: Detection) -> (logits: MLXArray, boxes: MLXArray) {
        let target = detection.outputMemory[0][indices].expandedDimensions(axis: 0)
        let referenceUnact = detection.encCoord[0][indices].expandedDimensions(axis: 0)
        let (logits, references) = decoder(target, value: detection.sourceFlatten,
                                           referencePointsUnact: referenceUnact, shapes: detection.shapes)
        return (logits[logits.count - 1][0], references[references.count - 1][0])
    }

    /// `pixels` `[1, H, W, 3]` NHWC → the full staged detection.
    public func callAsFunction(_ pixels: MLXArray) -> Detection {
        let features = backbone(pixels)
        let projected = features.enumerated().map { project(encoderInputProj[$0.offset], $0.element) }
        let panMaps = encoder(projected)

        // Flatten the PAN feature maps into one token sequence.
        var sources = [MLXArray]()
        var shapes = [(Int, Int)]()
        for level in 0 ..< config.numFeatureLevels {
            let source = project(decoderInputProj[level], panMaps[level])
            let (h, w, c) = (source.dim(1), source.dim(2), source.dim(3))
            shapes.append((h, w))
            sources.append(source.reshaped([1, h * w, c]))
        }
        let sourceFlatten = concatenated(sources, axis: 1)               // [1, S, dModel]

        let (anchorTensor, validMask) = anchors(shapes: shapes)
        let memory = validMask.expandedDimensions(axis: 0) * sourceFlatten
        var outputMemory = (encOutput[0] as! Linear)(memory)
        outputMemory = (encOutput[1] as! LayerNorm)(outputMemory)

        let encClass = encScoreHead(outputMemory)                        // [1, S, num_labels]
        let encCoord = encBboxHead(outputMemory) + anchorTensor.expandedDimensions(axis: 0)  // [1, S, 4]

        // Top-k query selection by the best class score.
        let scores = encClass[0].max(axis: -1)                           // [S]
        let order = argSort(-scores, axis: 0)
        let topk = order[0 ..< config.numQueries]                        // [Q]
        let referenceUnact = encCoord[0][topk].expandedDimensions(axis: 0)   // [1, Q, 4]
        let target = outputMemory[0][topk].expandedDimensions(axis: 0)       // [1, Q, dModel]

        let (logitsPerLayer, referencesPerLayer) = decoder(
            target, value: sourceFlatten, referencePointsUnact: referenceUnact, shapes: shapes)
        return Detection(backboneFeatures: features, encLastPAN: panMaps[panMaps.count - 1],
                         encClass: encClass, encCoord: encCoord, initReference: referenceUnact,
                         topkIndices: topk, outputMemory: outputMemory, sourceFlatten: sourceFlatten,
                         shapes: shapes, logitsPerLayer: logitsPerLayer,
                         referencesPerLayer: referencesPerLayer)
    }
}

// MARK: - Detection (post-processing) and backend

extension NFKMLXRTDetrNet {
    /// Detects objects in a bridged image `[H, W, 3]` (`0...1`): resize to the square input, forward,
    /// sigmoid the logits, threshold per query, and map the center-format boxes to normalized corners.
    /// RT-DETR is a DETR-family detector, so there is NO non-max suppression — the one-to-one training
    /// makes the queries already distinct. The boxes are normalized in the resized frame, and the resize
    /// squashes the whole frame, so a normalized box maps to the original frame unchanged.
    public func detect(_ image: MLXArray, labels: [String]?) -> [NFKDetection] {
        let batched = image.ndim == 3
            ? image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) : image
        let resized = NFKMLXResample.resizeBilinear(batched, height: config.inputResolution, width: config.inputResolution)
        let detection = self(resized)
        let scores = sigmoid(detection.logits); eval(scores)              // [Q, num_labels]
        let boxes = detection.boxes; eval(boxes)                          // [Q, 4] cxcywh normalized
        let scoreValues = scores.asArray(Float.self)
        let boxValues = boxes.asArray(Float.self)
        let queries = detection.logits.dim(0)
        let classes = config.numLabels

        var results = [NFKDetection]()
        for query in 0 ..< queries {
            var bestClass = 0
            var bestScore: Float = 0
            for k in 0 ..< classes {
                let score = scoreValues[query * classes + k]
                if score > bestScore {
                    bestScore = score
                    bestClass = k
                }
            }
            if bestScore < config.confidenceThreshold { continue }
            let cx = boxValues[query * 4], cy = boxValues[query * 4 + 1]
            let w = boxValues[query * 4 + 2], h = boxValues[query * 4 + 3]
            let minX = min(max(cx - w / 2, 0), 1), maxX = min(max(cx + w / 2, 0), 1)
            let minY = min(max(cy - h / 2, 0), 1), maxY = min(max(cy + h / 2, 0), 1)
            let rect = CGRect(x: CGFloat(minX), y: CGFloat(minY),
                              width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))
            let label = labels.flatMap { bestClass < $0.count ? $0[bestClass] : nil }
            results.append(NFKDetection(label: label, classIndex: bestClass,
                                        confidence: Double(bestScore), boundingBox: rect))
        }
        return results
    }
}

/// Holds the network and class labels for capture in the backend's `@Sendable` closure.
private final class NFKRTDetrHolder: @unchecked Sendable {
    let net: NFKMLXRTDetrNet
    let labels: [String]?
    init(_ net: NFKMLXRTDetrNet, labels: [String]?) {
        self.net = net
        self.labels = labels
    }
}

/// An object-detection backend: an image under `NFKInputImage` produces detections under
/// `NFKOutputDetections`. The license-clean (Apache-2.0) alternative to the AGPL YOLO.
@objc(NFKMLXRTDetrBackend)
public final class NFKMLXRTDetrBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKRTDetrHolder
    private let identifier: String

    init(net: NFKMLXRTDetrNet, identifier: String, labels: [String]?) {
        holder = NFKRTDetrHolder(net, labels: labels)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let value = request.input(forKey: NFKInputImage) else {
                    throw NFKMLXError.unsupportedInput
                }
                let image = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
                let detections = holder.net.detect(image, labels: holder.labels)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputDetections: detections]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// RT-DETR object detection as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXRTDetrNet` is the reference detector, at reference parity against transformers'
/// RTDetrForObjectDetection. Random weights run (proving the pipeline); the released `PekingU/rtdetr_r50vd`
/// safetensors detects accurately. The Apache-2.0 license makes it the clean swap for the AGPL YOLO.
@objc(NFKMLXRTDetr)
public final class NFKMLXRTDetr: NSObject {
    /// The registry name the model builds under.
    @objc public static let modelName = "rtdetr"

    /// Builds a detection backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). `labels` names classes when available.
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:labels:error:)
    public static func backend(weightsURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let net = NFKMLXRTDetrNet(.r50vd)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)                                                  // BatchNorm running statistics
        return NFKMLXRTDetrBackend(net: net, identifier: modelName, labels: labels)
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend. Blocking on the network; run
    /// off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url, labels: labels)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0, labels: labels) },
                               completionHandler: completionHandler)
    }

    /// Registers RT-DETR (`rtdetr`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:labels:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL, labels: nil)
        }
    }

    /// Maps the released checkpoint's names onto the module's. The net root is HF's `RTDetrModel`, so the
    /// `model.` prefix is stripped; the tied top-level `class_embed`/`bbox_embed` (duplicates of the
    /// decoder's) and every `num_batches_tracked` counter are dropped. A ResNet-D stage-1 bottleneck's
    /// stride-1 shortcut is a single `ShortCut` (`shortcut.convolution`), which the module holds as a
    /// one-element array, so it takes a `.0.` index; the stride-2 form is already `shortcut.1.*`.
    /// Returns nil for a key to drop.
    static func remapReferenceKey(_ key: String) -> String? {
        guard key.hasPrefix("model.") else { return nil }               // drops the tied top-level heads
        if key.hasSuffix("num_batches_tracked") { return nil }
        var name = String(key.dropFirst("model.".count))
        name = name.replacingOccurrences(of: ".shortcut.convolution.", with: ".shortcut.0.convolution.")
        name = name.replacingOccurrences(of: ".shortcut.normalization.", with: ".shortcut.0.normalization.")
        return name
    }

    /// Loads a safetensors checkpoint into `net`, remapping the reference's names and transposing 4-D
    /// convolution weights from PyTorch's `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXRTDetrNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard let name = remapReferenceKey(key) else { return nil }
            return (name, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
